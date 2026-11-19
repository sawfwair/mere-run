#if os(macOS)
import Foundation
import Metal
import MLX

/// An encoder-owned Qwen 27B FFN. The caller orders inputs and retains all
/// referenced buffers until its command buffer completes. Encoding never waits.
@available(macOS 26.0, *)
package final class M5NativeFFN {
    enum Failure: Error {
        case missingFunction(String), allocation, incompatibleWeights
    }

    package struct Projection {
        package var inputSize: Int { owner.originalWeight.dim(1) * 8 }
        package var outputSize: Int { owner.originalWeight.dim(0) }
        let owner: M5TiledAffineQ4.Weights
        let codes: any MTLBuffer
        let scales: any MTLBuffer
        let biases: any MTLBuffer
        let codesOffset: Int
        let scalesOffset: Int
        let biasesOffset: Int

        init(_ weights: M5TiledAffineQ4.Weights, device: any MTLDevice) throws {
            guard weights.groupAffineLayout else { throw Failure.incompatibleWeights }
            owner = weights
            let codes = try M5NativeArrayBuffer(weights.codes, device: device)
            let scales = try M5NativeArrayBuffer(weights.scales, device: device)
            let biases = try M5NativeArrayBuffer(weights.biases, device: device)
            self.codes = codes.buffer; self.scales = scales.buffer; self.biases = biases.buffer
            codesOffset = codes.offset; scalesOffset = scales.offset; biasesOffset = biases.offset
        }
    }

    package struct Weights {
        let gate: Projection
        let up: Projection
        let down: Projection

        init(gate: M5TiledAffineQ4.Weights, up: M5TiledAffineQ4.Weights,
             down: M5TiledAffineQ4.Weights, device: any MTLDevice) throws {
            try self.init(gate: Projection(gate, device: device), up: Projection(up, device: device),
                down: Projection(down, device: device))
        }

        package init(gate: Projection, up: Projection, down: Projection) throws {
            guard gate.owner.originalWeight.shape == [17_408, 640], up.owner.originalWeight.shape == [17_408, 640],
                  down.owner.originalWeight.shape == [5_120, 2_176] else { throw Failure.incompatibleWeights }
            self.gate = gate; self.up = up; self.down = down
        }
    }

    /// Request-owned scratch; reusing a workspace requires ordered GPU access.
    package struct Workspace {
        package let inputSums: any MTLBuffer
        let gateUp: any MTLBuffer
        let activated: any MTLBuffer
        let activationSums: any MTLBuffer

        package init(device: any MTLDevice, untracked: Bool = false, rows: Int = 16) throws {
            precondition(rows > 0 && rows.isMultiple(of: 16))
            let options: MTLResourceOptions = untracked ? [.storageModePrivate, .hazardTrackingModeUntracked] : .storageModePrivate
            func buffer(_ bytes: Int) throws -> any MTLBuffer {
                guard let value = device.makeBuffer(length: bytes, options: options) else {
                    throw Failure.allocation
                }
                return value
            }
            inputSums = try buffer(80 * rows * 4)
            gateUp = try buffer(rows * 17_408 * 2 * 2)
            activated = try buffer(rows * 17_408 * 2)
            activationSums = try buffer(272 * rows * 4)
        }
    }

    enum Stage: Int, CaseIterable { case inputSums, gateUp, activation, activationSums, down }
    package enum InputSums { case compute, provided }
    private let pipelines: [any MTLComputePipelineState]

    private let fuseActivationSums: Bool

    package init(device: any MTLDevice, fuseActivationSums: Bool = false) throws {
        self.fuseActivationSums = fuseActivationSums
        let activation = fuseActivationSums ? "mere_native_ffn_activation_with_sums" : "mere_native_ffn_activation"
        let library = try Self.makeLibrary(device: device, source: Self.source)
        pipelines = try ["mere_native_ffn_input_sums", "mere_native_ffn_gate_up", activation,
                         "mere_native_ffn_activation_sums", "mere_native_ffn_down"].map { name in
            guard let function = library.makeFunction(name: name) else { throw Failure.missingFunction(name) }
            return try device.makeComputePipelineState(function: function)
        }
    }

    package static func makeLibrary(device: any MTLDevice, source: String) throws -> any MTLLibrary {
        let options = MTLCompileOptions()
        if #available(macOS 27.0, *) {
            options.languageVersion = MTLLanguageVersion(rawValue: 0x40001)!
        } else {
            options.languageVersion = .version4_0
        }
        options.mathMode = .safe
        return try device.makeLibrary(source: source, options: options)
    }

    package func encode(into encoder: any MTLComputeCommandEncoder, input: any MTLBuffer,
                output: any MTLBuffer, weights: Weights, workspace: Workspace, rows: Int,
                inputSums: InputSums = .compute, inputOffset: Int = 0, hazards: M5NativeHazards? = nil) {
        for stage in Stage.allCases where inputSums == .compute || stage != .inputSums {
            if fuseActivationSums && stage == .activationSums { continue }
            encode(stage, into: encoder, input: input, output: output,
                weights: weights, workspace: workspace, rows: rows, inputOffset: inputOffset, hazards: hazards)
        }
    }

    func encode(_ stage: Stage, into encoder: any MTLComputeCommandEncoder, input: any MTLBuffer,
                output: any MTLBuffer, weights: Weights, workspace: Workspace, rows: Int, inputOffset: Int = 0, hazards: M5NativeHazards? = nil) {
        precondition(rows > 0)
        if let hazards {
            switch stage {
            case .inputSums:
                hazards.prepare(encoder: encoder, reads: [.init(input, offset: inputOffset, count: rows * 5120 * 2)], writes: [.init(workspace.inputSums)])
            case .gateUp:
                hazards.prepare(encoder: encoder, reads: [.init(input, offset: inputOffset, count: rows * 5120 * 2), .init(workspace.inputSums)], writes: [.init(workspace.gateUp)])
            case .activation:
                let writes: [M5NativeHazards.Access] = fuseActivationSums
                    ? [.init(workspace.activated), .init(workspace.activationSums)] : [.init(workspace.activated)]
                hazards.prepare(encoder: encoder, reads: [.init(workspace.gateUp)], writes: writes)
            case .activationSums:
                hazards.prepare(encoder: encoder, reads: [.init(workspace.activated)], writes: [.init(workspace.activationSums)])
            case .down:
                hazards.prepare(encoder: encoder, reads: [.init(workspace.activated), .init(workspace.activationSums)], writes: [.init(output)])
            }
        }
        encoder.setComputePipelineState(pipelines[stage.rawValue])
        let paddedRows = ((rows + 15) / 16) * 16
        var count = Int32(rows)
        switch stage {
        case .inputSums, .activationSums:
            let initial = stage == .inputSums
            encoder.setBuffer(initial ? input : workspace.activated, offset: initial ? inputOffset : 0, index: 0)
            encoder.setBuffer(initial ? workspace.inputSums : workspace.activationSums, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<Int32>.size, index: 2)
            encoder.dispatchThreads(MTLSize(width: (initial ? 80 : 272) * paddedRows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        case .gateUp, .down:
            let initial = stage == .gateUp
            let first = initial ? weights.gate : weights.down, second = initial ? weights.up : weights.down
            let buffers = [(first.codes, first.codesOffset), (first.scales, first.scalesOffset), (first.biases, first.biasesOffset),
                           (second.codes, second.codesOffset), (second.scales, second.scalesOffset), (second.biases, second.biasesOffset)]
            for (index, item) in buffers.enumerated() {
                encoder.setBuffer(item.0, offset: item.1, index: index)
            }
            encoder.setBuffer(initial ? input : workspace.activated, offset: initial ? inputOffset : 0, index: 6)
            encoder.setBuffer(initial ? workspace.inputSums : workspace.activationSums, offset: 0, index: 7)
            encoder.setBuffer(initial ? workspace.gateUp : output, offset: 0, index: 8)
            encoder.setBytes(&count, length: MemoryLayout<Int32>.size, index: 9)
            encoder.dispatchThreadgroups(MTLSize(width: initial ? 1_088 : 160, height: paddedRows / 16, depth: 1),
                threadsPerThreadgroup: MTLSize(width: initial ? 64 : 128, height: 1, depth: 1))
        case .activation:
            encoder.setBuffer(workspace.gateUp, offset: 0, index: 0)
            encoder.setBuffer(workspace.activated, offset: 0, index: 1)
            encoder.setBytes(&count, length: MemoryLayout<Int32>.size, index: 2)
            if fuseActivationSums {
                encoder.setBuffer(workspace.activationSums, offset: 0, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: 272, height: paddedRows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            } else {
                encoder.dispatchThreads(MTLSize(width: rows * 17_408, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }
}
#endif
