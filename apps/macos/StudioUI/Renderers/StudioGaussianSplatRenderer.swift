import CoreGraphics
import Metal
import StudioKit
import simd

// The Metal half of the splat viewer: each Gaussian is one instanced quad, sized in the vertex
// shader from its 3D covariance projected to the screen (the EWA splatting of Zwicker et al., as
// 3DGS rasterizes), shaded with a Gaussian falloff, and composited back to front with
// premultiplied "over" blending. The back-to-front order is a 16-bit counting sort of view depth
// on the CPU, which `StudioGaussianSplatMetalView` runs off the main thread whenever the eye moves.

/// One Gaussian as the vertex shader reads it: position and opacity, the six distinct terms of
/// its 3D covariance, and its color. Matches `Splat` in `shaderSource`.
private struct StudioSplatGPU {
    var position: SIMD4<Float>
    var covarianceA: SIMD4<Float>
    var covarianceB: SIMD4<Float>
    var color: SIMD4<Float>
}

/// Matches `Uniforms` in `shaderSource`.
private struct StudioSplatUniforms {
    var view: simd_float4x4
    var focal: SIMD2<Float>
    var viewport: SIMD2<Float>
    var tanHalf: SIMD2<Float>
    var padding: SIMD2<Float> = .zero
}

/// A back-to-front draw order for one eye position, held in a GPU buffer.
struct StudioSplatOrder: @unchecked Sendable {
    let buffer: MTLBuffer
    let eye: SIMD3<Float>
}

/// A splat uploaded to the GPU, with the pipeline that draws it. Immutable after `init`, so the
/// sort can run on any queue while the view draws on the main thread.
final class StudioGaussianSplatRenderer: @unchecked Sendable {
    static let pixelFormat = MTLPixelFormat.bgra8Unorm

    let device: MTLDevice
    let queue: MTLCommandQueue
    let count: Int
    private let pipeline: MTLRenderPipelineState
    private let splats: MTLBuffer
    private let positions: [SIMD3<Float>]

    /// Uploads `splat`, or nil when there is no Metal device or the shaders don't compile.
    init?(splat: StudioGaussianSplat, device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let queue = device.makeCommandQueue(),
              let pipeline = Self.pipeline(for: device), splat.count > 0 else { return nil }
        var gpu = [StudioSplatGPU]()
        gpu.reserveCapacity(splat.count)
        for index in 0..<splat.count {
            let rotation = simd_matrix3x3(splat.rotations[index])
            let scale = splat.scales[index]
            let scaled = simd_float3x3(rotation.columns.0 * scale.x, rotation.columns.1 * scale.y, rotation.columns.2 * scale.z)
            let sigma = scaled * scaled.transpose
            gpu.append(StudioSplatGPU(
                position: SIMD4(splat.positions[index], splat.opacities[index]),
                covarianceA: SIMD4(sigma[0, 0], sigma[0, 1], sigma[0, 2], sigma[1, 1]),
                covarianceB: SIMD4(sigma[1, 2], sigma[2, 2], 0, 0),
                color: SIMD4(splat.colors[index], splat.opacities[index])
            ))
        }
        guard let buffer = gpu.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }) else {
            return nil
        }
        self.device = device
        self.queue = queue
        self.pipeline = pipeline
        splats = buffer
        positions = splat.positions
        count = splat.count
    }

    /// The indices of every Gaussian, farthest from `eye` along `forward` first. A counting sort
    /// on 16-bit quantized depth: one pass to histogram, one to scatter.
    func order(for orbit: StudioGaussianSplatOrbit) -> StudioSplatOrder? {
        let eye = orbit.fileEye, forward = orbit.fileForward
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<UInt32>.stride, options: .storageModeShared) else {
            return nil
        }
        let output = UnsafeMutableBufferPointer(start: buffer.contents().bindMemory(to: UInt32.self, capacity: count), count: count)
        let depths = UnsafeMutableBufferPointer<Float>.allocate(capacity: count)
        let keys = UnsafeMutableBufferPointer<UInt16>.allocate(capacity: count)
        let histogram = UnsafeMutableBufferPointer<UInt32>.allocate(capacity: 65_536)
        defer {
            depths.deallocate()
            keys.deallocate()
            histogram.deallocate()
        }
        var nearest = Float.greatestFiniteMagnitude
        var farthest = -Float.greatestFiniteMagnitude
        positions.withUnsafeBufferPointer { points in
            for index in 0..<count {
                let depth = simd_dot(points[index] - eye, forward)
                depths[index] = depth
                nearest = min(nearest, depth)
                farthest = max(farthest, depth)
            }
        }
        let scale = farthest > nearest ? 65_535 / (farthest - nearest) : 0
        histogram.initialize(repeating: 0)
        for index in 0..<count {
            // Farthest first: the key grows as depth shrinks.
            let key = UInt16(min(max((farthest - depths[index]) * scale, 0), 65_535))
            keys[index] = key
            histogram[Int(key)] &+= 1
        }
        var running: UInt32 = 0
        for bucket in 0..<65_536 {
            let size = histogram[bucket]
            histogram[bucket] = running
            running &+= size
        }
        for index in 0..<count {
            let key = Int(keys[index])
            output[Int(histogram[key])] = UInt32(index)
            histogram[key] &+= 1
        }
        return StudioSplatOrder(buffer: buffer, eye: eye)
    }

    /// Draws every Gaussian in `order` from `orbit` into `texture`, cleared to transparent so
    /// the view's own background shows through.
    func encode(_ commandBuffer: MTLCommandBuffer, into texture: MTLTexture, orbit: StudioGaussianSplatOrbit, order: StudioSplatOrder) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        let width = Float(texture.width), height = Float(texture.height)
        let tanY = tan(orbit.fieldOfView / 2)
        let focal = (height / 2) / tanY
        var uniforms = StudioSplatUniforms(
            view: orbit.fileViewMatrix, focal: SIMD2(focal, focal), viewport: SIMD2(width, height),
            tanHalf: SIMD2(tanY * width / max(height, 1), tanY)
        )
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(splats, offset: 0, index: 0)
        encoder.setVertexBuffer(order.buffer, offset: 0, index: 1)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<StudioSplatUniforms>.stride, index: 2)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: count)
        encoder.endEncoding()
    }

    /// One frame drawn offscreen and read back, for a view drawn into a bitmap rather than to
    /// the screen (`cacheDisplay`, printing) and for measuring. Premultiplied BGRA in sRGB.
    func image(orbit: StudioGaussianSplatOrbit, width: Int, height: Int, order: StudioSplatOrder? = nil) -> CGImage? {
        guard width > 0, height > 0,
              let order = order ?? self.order(for: orbit) else { return nil }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: Self.pixelFormat, width: width, height: height, mipmapped: false)
        descriptor.usage = [.renderTarget]
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor),
              let commandBuffer = queue.makeCommandBuffer() else { return nil }
        encode(commandBuffer, into: texture, orbit: orbit, order: order)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)
        texture.getBytes(&pixels, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bytesPerRow, space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent
        )
    }

    // MARK: Pipeline

    nonisolated(unsafe) private static var pipelines: [ObjectIdentifier: MTLRenderPipelineState] = [:]
    private static let pipelineLock = NSLock()

    /// The splat pipeline for `device`, compiled from source once per device: SwiftPM builds no
    /// `.metallib` for a library target without a resource bundle, and the source is small.
    private static func pipeline(for device: MTLDevice) -> MTLRenderPipelineState? {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }
        if let known = pipelines[ObjectIdentifier(device)] { return known }
        guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else { return nil }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "Gaussian splats"
        descriptor.vertexFunction = library.makeFunction(name: "splatVertex")
        descriptor.fragmentFunction = library.makeFunction(name: "splatFragment")
        let attachment = descriptor.colorAttachments[0]!
        attachment.pixelFormat = pixelFormat
        attachment.isBlendingEnabled = true
        attachment.rgbBlendOperation = .add
        attachment.alphaBlendOperation = .add
        attachment.sourceRGBBlendFactor = .one
        attachment.sourceAlphaBlendFactor = .one
        attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: descriptor) else { return nil }
        pipelines[ObjectIdentifier(device)] = pipeline
        return pipeline
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Splat { float4 position; float4 covA; float4 covB; float4 color; };
    struct Uniforms { float4x4 view; float2 focal; float2 viewport; float2 tanHalf; float2 padding; };
    struct Fragment { float4 position [[position]]; float2 local; half4 color; };

    constant float2 corners[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };

    vertex Fragment splatVertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                const device Splat *splats [[buffer(0)]],
                                const device uint *order [[buffer(1)]],
                                constant Uniforms &u [[buffer(2)]]) {
        Fragment out;
        out.position = float4(0, 0, 2, 1);
        out.local = float2(0);
        out.color = half4(0);
        Splat s = splats[order[iid]];
        float4 p = u.view * float4(s.position.xyz, 1.0);
        if (p.z < 0.01) { return out; }
        float limx = 1.3 * u.tanHalf.x, limy = 1.3 * u.tanHalf.y;
        float tx = clamp(p.x / p.z, -limx, limx) * p.z;
        float ty = clamp(p.y / p.z, -limy, limy) * p.z;
        float3x3 sigma = float3x3(float3(s.covA.x, s.covA.y, s.covA.z),
                                  float3(s.covA.y, s.covA.w, s.covB.x),
                                  float3(s.covA.z, s.covB.x, s.covB.y));
        float3x3 w = float3x3(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
        float z2 = p.z * p.z;
        float3x3 j = float3x3(float3(u.focal.x / p.z, 0, 0),
                              float3(0, u.focal.y / p.z, 0),
                              float3(-u.focal.x * tx / z2, -u.focal.y * ty / z2, 0));
        float3x3 t = j * w;
        float3x3 cov = t * sigma * transpose(t);
        float a = cov[0][0] + 0.3, b = cov[0][1], c = cov[1][1] + 0.3;
        float mid = 0.5 * (a + c);
        float spread = sqrt(max(0.0, mid * mid - (a * c - b * b)));
        float major = mid + spread, minor = max(mid - spread, 0.09);
        float2 axis = abs(b) > 1e-7 ? normalize(float2(b, major - a)) : (a >= c ? float2(1, 0) : float2(0, 1));
        float2 across = float2(-axis.y, axis.x);
        float2 corner = corners[vid];
        float2 offset = corner.x * min(3.0 * sqrt(major), 2048.0) * axis + corner.y * min(3.0 * sqrt(minor), 2048.0) * across;
        float2 center = u.focal * p.xy / p.z;
        out.position = float4((center + offset) / (0.5 * u.viewport), 0.5, 1.0);
        out.local = corner * 3.0;
        out.color = half4(s.color);
        return out;
    }

    fragment half4 splatFragment(Fragment in [[stage_in]]) {
        float alpha = min(0.99, float(in.color.a) * exp(-0.5 * dot(in.local, in.local)));
        if (alpha < 1.0 / 255.0) { discard_fragment(); }
        return half4(in.color.rgb * half(alpha), half(alpha));
    }
    """
}
