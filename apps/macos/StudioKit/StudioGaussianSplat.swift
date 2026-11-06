import Foundation
import simd

// A 3D Gaussian splat as the INRIA 3DGS PLY layout stores it, read for Studio's splat viewer.
// IDOL (`vision image-to-3d-idol`, `IdolGaussians.plyData`) writes it: one binary little-endian
// `vertex` element of float properties x y z, nx ny nz, f_dc_0..2, opacity, scale_0..2, and
// rot_0..3, with opacity as a logit, scales as logs, and rot_0 as the quaternion's real part.
// Other 3DGS exporters add f_rest_* (higher-order SH) or drop the normals; the reader takes the
// properties it needs by name from the header, in whatever order and type the header declares.

/// Why a PLY file could not be read as a Gaussian splat.
package enum StudioGaussianSplatError: Error, Equatable, LocalizedError {
    case notPLY
    case unterminatedHeader
    case malformedHeader(String)
    case unsupportedFormat(String)
    case missingVertexElement
    case missingProperties([String])
    case listProperty(String)
    case truncated(expectedBytes: Int, actualBytes: Int)
    case empty

    package var errorDescription: String? {
        switch self {
        case .notPLY:
            return "The file isn't a PLY file."
        case .unterminatedHeader:
            return "The PLY header has no end_header line."
        case .malformedHeader(let line):
            return "The PLY header has a line Studio can't read: \(line)"
        case .unsupportedFormat(let format):
            return "Studio reads binary little-endian splat PLY files, not \(format)."
        case .missingVertexElement:
            return "The PLY file has no vertex element."
        case .missingProperties(let names):
            return "The PLY file isn't a Gaussian splat. It has no \(names.joined(separator: ", ")) property."
        case .listProperty(let name):
            return "The PLY file has a list property (\(name)) where a splat has fixed-size rows."
        case .truncated(let expected, let actual):
            return "The PLY file is truncated: its header promises \(expected.formatted()) bytes of rows, "
                + "but \(actual.formatted()) follow it."
        case .empty:
            return "The PLY file has no Gaussians."
        }
    }
}

/// A PLY header: its format, its elements in file order, and where the rows start.
package struct StudioPLYHeader: Equatable, Sendable {
    package enum Format: String, Sendable {
        case ascii
        case binaryLittleEndian = "binary_little_endian"
        case binaryBigEndian = "binary_big_endian"
    }

    package enum ScalarType: Sendable {
        case int8, uint8, int16, uint16, int32, uint32, float32, float64

        init?(plyName: String) {
            switch plyName {
            case "char", "int8": self = .int8
            case "uchar", "uint8": self = .uint8
            case "short", "int16": self = .int16
            case "ushort", "uint16": self = .uint16
            case "int", "int32": self = .int32
            case "uint", "uint32": self = .uint32
            case "float", "float32": self = .float32
            case "double", "float64": self = .float64
            default: return nil
            }
        }

        package var byteCount: Int {
            switch self {
            case .int8, .uint8: return 1
            case .int16, .uint16: return 2
            case .int32, .uint32, .float32: return 4
            case .float64: return 8
            }
        }
    }

    package struct Property: Equatable, Sendable {
        package let name: String
        /// Nil for a list property, whose rows have no fixed size.
        package let type: ScalarType?
    }

    package struct Element: Equatable, Sendable {
        package let name: String
        package let count: Int
        package var properties: [Property]

        /// Bytes per row, or nil when a list property makes rows variable-length.
        package var rowByteCount: Int? {
            properties.reduce(0 as Int?) { total, property in
                guard let total, let type = property.type else { return nil }
                return total + type.byteCount
            }
        }
    }

    /// The properties a splat needs: position, degree-0 SH color, opacity, scale, and rotation.
    package static let splatProperties = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"
    ]

    /// The longest header the reader looks through for `end_header`. 3DGS headers with 45
    /// `f_rest_*` properties run about 1.5 KiB.
    package static let maximumHeaderByteCount = 64 * 1024

    package let format: Format
    package let elements: [Element]
    /// The byte offset of the first row, just past `end_header\n`.
    package let dataOffset: Int

    package var vertex: Element? { elements.first { $0.name == "vertex" } }

    /// The splat properties the vertex element lacks; empty for a Gaussian splat.
    package var missingSplatProperties: [String] {
        let names = Set(vertex?.properties.map(\.name) ?? [])
        return Self.splatProperties.filter { !names.contains($0) }
    }

    /// Whether the vertex rows carry every property a Gaussian splat needs. A mesh or a point
    /// cloud PLY (x y z, maybe colors and normals) does not.
    package var isGaussianSplat: Bool { vertex != nil && missingSplatProperties.isEmpty }

    package static func parse(_ data: Data) throws -> StudioPLYHeader {
        guard data.starts(with: Array("ply\n".utf8)) || data.starts(with: Array("ply\r\n".utf8)) else {
            throw StudioGaussianSplatError.notPLY
        }
        let marker = Array("end_header".utf8)
        let searched = data.prefix(maximumHeaderByteCount)
        guard let markerRange = searched.firstRange(of: marker) else {
            throw StudioGaussianSplatError.unterminatedHeader
        }
        guard let newline = searched[markerRange.upperBound...].firstIndex(of: UInt8(ascii: "\n")) else {
            throw StudioGaussianSplatError.unterminatedHeader
        }
        let text = String(decoding: data[data.startIndex..<markerRange.lowerBound], as: UTF8.self)
        var format: Format?
        var elements: [Element] = []
        for rawLine in text.split(whereSeparator: \.isNewline).dropFirst() {
            let words = rawLine.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            guard let keyword = words.first else { continue }
            switch keyword {
            case "format":
                guard words.count >= 2 else { throw StudioGaussianSplatError.malformedHeader(String(rawLine)) }
                guard let parsed = Format(rawValue: words[1]) else {
                    throw StudioGaussianSplatError.unsupportedFormat(words[1])
                }
                format = parsed
            case "element":
                guard words.count == 3, let count = Int(words[2]), count >= 0 else {
                    throw StudioGaussianSplatError.malformedHeader(String(rawLine))
                }
                elements.append(Element(name: words[1], count: count, properties: []))
            case "property":
                guard !elements.isEmpty else { throw StudioGaussianSplatError.malformedHeader(String(rawLine)) }
                if words.count == 5, words[1] == "list" {
                    elements[elements.count - 1].properties.append(Property(name: words[4], type: nil))
                } else if words.count == 3, let type = ScalarType(plyName: words[1]) {
                    elements[elements.count - 1].properties.append(Property(name: words[2], type: type))
                } else {
                    throw StudioGaussianSplatError.malformedHeader(String(rawLine))
                }
            case "comment", "obj_info":
                continue
            default:
                throw StudioGaussianSplatError.malformedHeader(String(rawLine))
            }
        }
        guard let format else { throw StudioGaussianSplatError.malformedHeader("no format line") }
        return StudioPLYHeader(format: format, elements: elements, dataOffset: newline + 1 - data.startIndex)
    }

    /// The header at the front of `url`, reading no more than `maximumHeaderByteCount` bytes.
    package static func read(from url: URL) throws -> StudioPLYHeader {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        return try parse(handle.readData(ofLength: maximumHeaderByteCount))
    }

    /// Whether the PLY at `url` is a Gaussian splat, judged by its header's properties rather
    /// than its name. The feed asks on every render, so the answer is kept per file until the
    /// file changes.
    package static func isGaussianSplat(at url: URL) -> Bool {
        guard url.pathExtension.lowercased() == "ply" else { return false }
        let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        let stamp = FileStamp(modified: values?.contentModificationDate, size: values?.fileSize)
        sniffLock.lock()
        defer { sniffLock.unlock() }
        if let known = sniffCache[url.path], known.stamp == stamp { return known.isSplat }
        let isSplat = (try? read(from: url))?.isGaussianSplat == true
        if sniffCache.count > 256 { sniffCache.removeAll() }
        sniffCache[url.path] = (stamp, isSplat)
        return isSplat
    }

    private struct FileStamp: Equatable {
        let modified: Date?
        let size: Int?
    }

    private static let sniffLock = NSLock()
    nonisolated(unsafe) private static var sniffCache: [String: (stamp: FileStamp, isSplat: Bool)] = [:]
}

/// A Gaussian splat with its activations applied: positions in the file's frame, RGB from the
/// degree-0 spherical-harmonic coefficients, opacity through a sigmoid, scales through `exp`, and
/// unit rotations. Higher-order SH (`f_rest_*`) is not read; the viewer draws view-independent
/// color, as IDOL's own inspection renderer does.
package struct StudioGaussianSplat: Sendable {
    /// The degree-0 spherical-harmonic basis constant, 1 / (2√π).
    package static let shC0: Float = 0.282_094_79

    /// How many Gaussians a load keeps by default. IDOL writes 262,144 unless asked for more;
    /// a bigger file is read at a stride so loading and sorting stay interactive.
    package static let defaultMaximumCount = 300_000

    package let positions: [SIMD3<Float>]
    package let colors: [SIMD3<Float>]
    package let opacities: [Float]
    package let scales: [SIMD3<Float>]
    package let rotations: [simd_quatf]
    /// How many Gaussians the file holds.
    package let fileCount: Int
    /// Every `stride`-th row was read; 1 when the file fit under the cap.
    package let stride: Int
    package let boundsMin: SIMD3<Float>
    package let boundsMax: SIMD3<Float>

    package var count: Int { positions.count }

    package init(
        positions: [SIMD3<Float>], colors: [SIMD3<Float>], opacities: [Float], scales: [SIMD3<Float>],
        rotations: [simd_quatf], fileCount: Int? = nil, stride: Int = 1
    ) {
        self.positions = positions
        self.colors = colors
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.fileCount = fileCount ?? positions.count
        self.stride = stride
        var lower = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var upper = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for position in positions {
            lower = simd_min(lower, position)
            upper = simd_max(upper, position)
        }
        boundsMin = positions.isEmpty ? .zero : lower
        boundsMax = positions.isEmpty ? .zero : upper
    }

    /// Reads the splat at `url`, memory-mapped. Runs on the caller's thread; views use the async
    /// overload so a 17 MB IDOL file never decodes on the main thread.
    package static func load(from url: URL, maximumCount: Int = defaultMaximumCount) throws -> StudioGaussianSplat {
        try decode(Data(contentsOf: url, options: .mappedIfSafe), maximumCount: maximumCount)
    }

    package static func load(
        from url: URL, maximumCount: Int = defaultMaximumCount, priority: TaskPriority = .userInitiated
    ) async throws -> StudioGaussianSplat {
        try await Task.detached(priority: priority) {
            try load(from: url, maximumCount: maximumCount)
        }.value
    }

    package static func decode(_ data: Data, maximumCount: Int = defaultMaximumCount) throws -> StudioGaussianSplat {
        let header = try StudioPLYHeader.parse(data)
        guard header.format == .binaryLittleEndian else {
            throw StudioGaussianSplatError.unsupportedFormat(header.format == .ascii ? "ASCII" : "big-endian binary")
        }
        guard let vertexIndex = header.elements.firstIndex(where: { $0.name == "vertex" }) else {
            throw StudioGaussianSplatError.missingVertexElement
        }
        let missing = header.missingSplatProperties
        guard missing.isEmpty else { throw StudioGaussianSplatError.missingProperties(missing) }
        var vertexOffset = header.dataOffset
        for element in header.elements[..<vertexIndex] {
            guard let rowBytes = element.rowByteCount else {
                throw StudioGaussianSplatError.listProperty(element.properties.first { $0.type == nil }?.name ?? element.name)
            }
            vertexOffset += rowBytes * element.count
        }
        let vertex = header.elements[vertexIndex]
        guard let rowBytes = vertex.rowByteCount else {
            throw StudioGaussianSplatError.listProperty(vertex.properties.first { $0.type == nil }?.name ?? "vertex")
        }
        guard vertex.count > 0 else { throw StudioGaussianSplatError.empty }
        let expected = rowBytes * vertex.count
        let available = data.count - vertexOffset
        guard available >= expected else {
            throw StudioGaussianSplatError.truncated(expectedBytes: expected, actualBytes: max(available, 0))
        }

        // Byte offset and type of each property the splat needs, in `splatProperties` order.
        var fields: [(offset: Int, type: StudioPLYHeader.ScalarType)] = []
        for name in StudioPLYHeader.splatProperties {
            var offset = 0
            for property in vertex.properties {
                guard let type = property.type else { continue }
                if property.name == name {
                    fields.append((offset, type))
                    break
                }
                offset += type.byteCount
            }
        }
        let allFloat = fields.allSatisfy { $0.type == .float32 }
        let stride = max(1, (vertex.count + maximumCount - 1) / max(maximumCount, 1))
        let capacity = (vertex.count + stride - 1) / stride

        var positions: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []
        var opacities: [Float] = []
        var scales: [SIMD3<Float>] = []
        var rotations: [simd_quatf] = []
        positions.reserveCapacity(capacity)
        colors.reserveCapacity(capacity)
        opacities.reserveCapacity(capacity)
        scales.reserveCapacity(capacity)
        rotations.reserveCapacity(capacity)

        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress! + vertexOffset
            var values = [Float](repeating: 0, count: fields.count)
            var row = 0
            while row < vertex.count {
                let record = base + row * rowBytes
                if allFloat {
                    for index in fields.indices {
                        values[index] = Float(bitPattern: UInt32(littleEndian: record.loadUnaligned(fromByteOffset: fields[index].offset, as: UInt32.self)))
                    }
                } else {
                    for index in fields.indices {
                        values[index] = Self.scalar(at: record + fields[index].offset, type: fields[index].type)
                    }
                }
                row += stride
                guard values.allSatisfy(\.isFinite) else { continue }
                positions.append(SIMD3(values[0], values[1], values[2]))
                colors.append(simd_clamp(SIMD3(repeating: 0.5) + shC0 * SIMD3(values[3], values[4], values[5]),
                                         SIMD3(repeating: 0), SIMD3(repeating: 1)))
                opacities.append(1 / (1 + exp(-min(max(values[6], -80), 80))))
                scales.append(SIMD3(exp(min(values[7], 20)), exp(min(values[8], 20)), exp(min(values[9], 20))))
                let quaternion = simd_quatf(ix: values[11], iy: values[12], iz: values[13], r: values[10])
                let length = simd_length(quaternion.vector)
                rotations.append(length > 1e-12 ? simd_quatf(vector: quaternion.vector / length) : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
            }
        }
        guard !positions.isEmpty else { throw StudioGaussianSplatError.empty }
        return StudioGaussianSplat(
            positions: positions, colors: colors, opacities: opacities, scales: scales, rotations: rotations,
            fileCount: vertex.count, stride: stride
        )
    }

    private static func scalar(at pointer: UnsafeRawPointer, type: StudioPLYHeader.ScalarType) -> Float {
        switch type {
        case .int8: return Float(pointer.loadUnaligned(as: Int8.self))
        case .uint8: return Float(pointer.loadUnaligned(as: UInt8.self))
        case .int16: return Float(Int16(littleEndian: pointer.loadUnaligned(as: Int16.self)))
        case .uint16: return Float(UInt16(littleEndian: pointer.loadUnaligned(as: UInt16.self)))
        case .int32: return Float(Int32(littleEndian: pointer.loadUnaligned(as: Int32.self)))
        case .uint32: return Float(UInt32(littleEndian: pointer.loadUnaligned(as: UInt32.self)))
        case .float32: return Float(bitPattern: UInt32(littleEndian: pointer.loadUnaligned(as: UInt32.self)))
        case .float64: return Float(Double(bitPattern: UInt64(littleEndian: pointer.loadUnaligned(as: UInt64.self))))
        }
    }

    /// The sphere the viewer frames: centered on the per-axis median and wide enough for 98% of
    /// the Gaussians, so a few stray splats far from the object don't shrink it to a dot.
    package var framingSphere: (center: SIMD3<Float>, radius: Float) {
        guard !positions.isEmpty else { return (.zero, 1) }
        let step = max(1, positions.count / 8_192)
        let sample = Swift.stride(from: 0, to: positions.count, by: step).map { positions[$0] }
        func median(_ axis: Int) -> Float {
            let sorted = sample.map { $0[axis] }.sorted()
            return sorted[sorted.count / 2]
        }
        let center = SIMD3(median(0), median(1), median(2))
        let distances = sample.map { simd_distance($0, center) }.sorted()
        let radius = distances[min(distances.count - 1, Int(Float(distances.count) * 0.98))]
        return (center, max(radius, 1e-4))
    }
}

/// The input view IDOL estimated, from the `camera.json` it writes beside `object.ply`
/// (`IdolCamera`): azimuth and elevation in radians in IDOL's z-up frame, the camera's distance,
/// and its vertical field of view in degrees.
package struct StudioGaussianSplatCamera: Codable, Equatable, Sendable {
    package let yaw: Float
    package let pitch: Float
    package let radius: Float
    package let fov: Float

    package init(yaw: Float, pitch: Float, radius: Float, fov: Float) {
        self.yaw = yaw
        self.pitch = pitch
        self.radius = radius
        self.fov = fov
    }

    package static func load(from url: URL) -> StudioGaussianSplatCamera? {
        guard let data = try? Data(contentsOf: url),
              let camera = try? JSONDecoder().decode(StudioGaussianSplatCamera.self, from: data),
              camera.yaw.isFinite, camera.pitch.isFinite else { return nil }
        return camera
    }

    /// The orbit angles that put the viewer where the input picture was taken from. IDOL's
    /// direction (sin yaw cos pitch, cos yaw cos pitch, sin pitch) points from the object toward
    /// the camera in its z-up frame, which is (x, z, −y) in the viewer's upright world.
    package var orbitAngles: (yaw: Float, pitch: Float) {
        (yaw: .pi - yaw, pitch: pitch)
    }
}

/// The orbit camera the splat viewer draws from: angles around a target, a distance, and a
/// vertical field of view, in an upright world with +y up. Pure math, so framing and gestures
/// are testable without Metal.
///
/// A splat PLY puts +y down, as 3DGS's COLMAP-derived frames do; IDOL's export does too, since
/// `IdolGaussians.plyData` writes its z-up (x, y, z) as (x, −z, y). The viewer stands the file
/// upright with a half turn about x (`fileToWorld`), which keeps handedness, so a Gaussian's
/// covariance needs no change beyond the view matrix.
package struct StudioGaussianSplatOrbit: Equatable, Sendable {
    /// File coordinates to the upright world: (x, y, z) → (x, −y, −z). Its own inverse.
    package static let fileToWorld = SIMD3<Float>(1, -1, -1)

    /// Azimuth around +y in radians; 0 looks from +z toward −z.
    package var yaw: Float
    /// Elevation in radians, positive above the target.
    package var pitch: Float
    package var distance: Float
    package var target: SIMD3<Float>
    /// Vertical field of view in radians.
    package var fieldOfView: Float

    package static let pitchLimit: Float = 1.45
    package static let defaultFieldOfView: Float = 35 * .pi / 180
    /// A three-quarter view from slightly above, for a splat without an input camera.
    package static let defaultAngles: (yaw: Float, pitch: Float) = (0.6, 0.25)

    package init(yaw: Float, pitch: Float, distance: Float, target: SIMD3<Float>, fieldOfView: Float = defaultFieldOfView) {
        self.yaw = yaw
        self.pitch = min(max(pitch, -Self.pitchLimit), Self.pitchLimit)
        self.distance = distance
        self.target = target
        self.fieldOfView = fieldOfView
    }

    /// The starting view: the input camera's direction when there is one, else a three-quarter
    /// view, far enough back that the framing sphere fills the narrower side with a margin.
    package static func framing(
        _ splat: StudioGaussianSplat, camera: StudioGaussianSplatCamera?, aspect: Float = 1
    ) -> StudioGaussianSplatOrbit {
        framing(sphere: splat.framingSphere, camera: camera, aspect: aspect)
    }

    /// `sphere` is in file coordinates, as `StudioGaussianSplat.framingSphere` reports it.
    package static func framing(
        sphere: (center: SIMD3<Float>, radius: Float), camera: StudioGaussianSplatCamera?, aspect: Float = 1
    ) -> StudioGaussianSplatOrbit {
        let angles = camera?.orbitAngles ?? defaultAngles
        let fieldOfView = defaultFieldOfView
        let horizontal = 2 * atan(tan(fieldOfView / 2) * max(aspect, 1e-3))
        let narrowest = min(fieldOfView, horizontal)
        let distance = sphere.radius / sin(narrowest / 2) * 1.15
        return StudioGaussianSplatOrbit(
            yaw: angles.yaw, pitch: angles.pitch, distance: distance, target: sphere.center * fileToWorld,
            fieldOfView: fieldOfView
        )
    }

    package var eye: SIMD3<Float> {
        target + distance * SIMD3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw))
    }

    /// Unit vector from the eye toward the target.
    package var forward: SIMD3<Float> { simd_normalize(target - eye) }

    /// The eye and view direction in file coordinates, for sorting the file's positions by depth.
    package var fileEye: SIMD3<Float> { eye * Self.fileToWorld }
    package var fileForward: SIMD3<Float> { forward * Self.fileToWorld }

    /// File coordinates to camera, for drawing the file's positions and covariances.
    package var fileViewMatrix: simd_float4x4 {
        viewMatrix * simd_float4x4(diagonal: SIMD4(Self.fileToWorld, 1))
    }

    /// World to camera: x right, y up, z forward (away from the viewer).
    package var viewMatrix: simd_float4x4 {
        let forward = forward
        let right = simd_normalize(simd_cross(forward, SIMD3(0, 1, 0)))
        let up = simd_cross(right, forward)
        let eye = eye
        return simd_float4x4(rows: [
            SIMD4(right, -simd_dot(right, eye)),
            SIMD4(up, -simd_dot(up, eye)),
            SIMD4(forward, -simd_dot(forward, eye)),
            SIMD4(0, 0, 0, 1)
        ])
    }

    /// Orbits by a drag of `delta` points in a view `height` points tall: a full-height drag
    /// turns the object about 180°.
    package mutating func orbit(by delta: CGSize, height: CGFloat) {
        let scale = Float.pi / Float(max(height, 1))
        yaw -= Float(delta.width) * scale
        pitch = min(max(pitch + Float(delta.height) * scale, -Self.pitchLimit), Self.pitchLimit)
    }

    /// Moves toward the target by `factor` (below 1 zooms in), within `range` of the framed distance.
    package mutating func zoom(by factor: Float, framedDistance: Float) {
        distance = min(max(distance * factor, framedDistance * 0.05), framedDistance * 8)
    }

    /// Slides the target across the view plane by a drag of `delta` points in a view `height` tall.
    package mutating func pan(by delta: CGSize, height: CGFloat) {
        let perPoint = 2 * distance * tan(fieldOfView / 2) / Float(max(height, 1))
        let forward = forward
        let right = simd_normalize(simd_cross(forward, SIMD3(0, 1, 0)))
        let up = simd_cross(right, forward)
        target += (-Float(delta.width) * right + Float(delta.height) * up) * perPoint
    }

    /// The view between `self` and `other` at `progress` in 0...1, for an animated reset.
    package func interpolated(to other: StudioGaussianSplatOrbit, progress: Float) -> StudioGaussianSplatOrbit {
        let t = min(max(progress, 0), 1)
        var turn = (other.yaw - yaw).truncatingRemainder(dividingBy: 2 * .pi)
        if turn > .pi { turn -= 2 * .pi } else if turn < -.pi { turn += 2 * .pi }
        return StudioGaussianSplatOrbit(
            yaw: yaw + turn * t, pitch: pitch + (other.pitch - pitch) * t,
            distance: distance + (other.distance - distance) * t,
            target: target + (other.target - target) * t,
            fieldOfView: fieldOfView + (other.fieldOfView - fieldOfView) * t
        )
    }
}

/// What a run wrote that the splat viewer draws: the splat PLY, found by its header, and the
/// input camera and prepared picture IDOL writes beside it.
package struct StudioGaussianSplatResult: Equatable, Sendable {
    package let splatURL: URL
    package let cameraURL: URL?
    package let preparedInputURL: URL?

    package init(splatURL: URL, cameraURL: URL? = nil, preparedInputURL: URL? = nil) {
        self.splatURL = splatURL
        self.cameraURL = cameraURL
        self.preparedInputURL = preparedInputURL
    }

    /// The first PLY among `files` whose header declares a Gaussian splat, with the
    /// `camera.json` and `input.png` from the same folder when the run wrote them. Nil when no
    /// PLY is a splat, so a mesh or point cloud keeps Quick Look.
    package static func find(in files: [URL]) -> StudioGaussianSplatResult? {
        guard let splat = files.first(where: StudioPLYHeader.isGaussianSplat(at:)) else { return nil }
        let folder = splat.deletingLastPathComponent().standardizedFileURL
        func sibling(_ name: String) -> URL? {
            files.first { $0.lastPathComponent == name && $0.deletingLastPathComponent().standardizedFileURL == folder }
        }
        return StudioGaussianSplatResult(splatURL: splat, cameraURL: sibling("camera.json"), preparedInputURL: sibling("input.png"))
    }
}
