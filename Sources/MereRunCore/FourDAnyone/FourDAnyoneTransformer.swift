import Foundation
import MLX
import MLXNN

/// Prepared tensors use upstream channel-first video and pose layouts.
public struct FourDAnyoneTransformerInput {
    public let latents: MLXArray
    public let sources: MLXArray
    public let poseFeatures: MLXArray
    public let nullPoseFeatures: MLXArray
    public let promptContext: MLXArray
    public let timestep: Float

    public init(
        latents: MLXArray,
        sources: MLXArray,
        poseFeatures: MLXArray,
        nullPoseFeatures: MLXArray,
        promptContext: MLXArray,
        timestep: Float
    ) {
        self.latents = latents
        self.sources = sources
        self.poseFeatures = poseFeatures
        self.nullPoseFeatures = nullPoseFeatures
        self.promptContext = promptContext
        self.timestep = timestep
    }

    func validate(configuration: Wan2TransformerConfiguration) throws -> Wan2GridSize {
        guard latents.ndim == 5, latents.shape.allSatisfy({ $0 > 0 }),
              latents.dim(1) == configuration.inputChannels,
              latents.dim(3).isMultiple(of: 2), latents.dim(4).isMultiple(of: 2) else {
            throw FourDAnyoneError.invalidInput("Latents must have shape [views, channels, frames, 2h, 2w].")
        }
        guard sources.ndim == 5, [1, 5].contains(sources.dim(0)),
              Array(sources.shape.dropFirst()) == Array(latents.shape.dropFirst()) else {
            throw FourDAnyoneError.invalidInput("Supply one source video or one source plus four reference videos.")
        }
        let grid = Wan2GridSize(frames: latents.dim(2), height: latents.dim(3) / 2, width: latents.dim(4) / 2)
        let featureShape = [configuration.hiddenSize, grid.frames, grid.height, grid.width]
        let packedViews = sources.dim(0) == 5 ? 2 : 1
        guard poseFeatures.shape == [latents.dim(0)] + featureShape,
              nullPoseFeatures.shape == [packedViews] + featureShape else {
            throw FourDAnyoneError.invalidInput("Pose and encoded null features must match the target and packed grids.")
        }
        guard promptContext.ndim == 3, promptContext.dim(0) == 1,
              promptContext.dim(1) > 0, promptContext.dim(1) <= configuration.textLength,
              promptContext.dim(2) == configuration.textEmbeddingSize,
              timestep.isFinite, (0...1_000).contains(timestep) else {
            throw FourDAnyoneError.invalidInput("Prompt context or timestep does not match the transformer configuration.")
        }
        return grid
    }
}

final class FourDAnyoneViewPack: Module {
    @ModuleInfo(key: "proj_2x") var halfResolution: Linear
    @ModuleInfo(key: "proj_4x") var quarterResolution: Linear
    let computePrecision: FourDAnyoneComputePrecision

    init(_ configuration: Wan2TransformerConfiguration, computePrecision: FourDAnyoneComputePrecision) {
        self.computePrecision = computePrecision
        self._halfResolution.wrappedValue = Linear(configuration.inputChannels * 16, configuration.hiddenSize)
        // Retain and validate the trained partition even though inference uses proj_2x.
        self._quarterResolution.wrappedValue = Linear(configuration.inputChannels * 64, configuration.hiddenSize)
    }

    func callAsFunction(_ references: MLXArray, grid: Wan2GridSize) -> MLXArray {
        var padded = references
        let padHeight = (4 - padded.dim(3) % 4) % 4
        let padWidth = (4 - padded.dim(4) % 4) % 4
        if padHeight > 0 {
            let edge = MLX.repeated(padded[0..., 0..., 0..., (padded.dim(3) - 1)..<padded.dim(3), 0...],
                                    count: padHeight, axis: 3)
            padded = MLX.concatenated([padded, edge], axis: 3)
        }
        if padWidth > 0 {
            let edge = MLX.repeated(padded[0..., 0..., 0..., 0..., (padded.dim(4) - 1)..<padded.dim(4)],
                                    count: padWidth, axis: 4)
            padded = MLX.concatenated([padded, edge], axis: 4)
        }
        let height = padded.dim(3) / 4
        let width = padded.dim(4) / 4
        let patches = (0..<4).map { index in
            Wan2PatchLayout.flatten(padded[index], patchSize: [1, 4, 4]).value
        }
        let projected = halfResolution(
            MLX.stacked(patches, axis: 0).asType(computePrecision.dtype(for: halfResolution.weight.dtype))
        )
        let packed = projected.reshaped(2, 2, grid.frames, height, width, -1)
            .transposed(2, 0, 3, 1, 4, 5)
            .reshaped(1, grid.frames, 2 * height, 2 * width, -1)
        return packed[0..., 0..., 0..<grid.height, 0..<grid.width, 0...]
            .reshaped(1, grid.sequenceLength, -1)
    }
}

final class FourDAnyoneHead: Module {
    @ModuleInfo var norm: Wan2LayerNorm
    @ModuleInfo var head: Linear
    @ModuleInfo var modulation: MLXArray
    let computePrecision: FourDAnyoneComputePrecision

    init(_ configuration: Wan2TransformerConfiguration, computePrecision: FourDAnyoneComputePrecision) {
        self.computePrecision = computePrecision
        self._norm.wrappedValue = Wan2LayerNorm(
            dimensions: configuration.hiddenSize, epsilon: configuration.epsilon
        )
        self._head.wrappedValue = Linear(configuration.hiddenSize, configuration.outputChannels * 4)
        self._modulation.wrappedValue = MLX.zeros([1, 2, configuration.hiddenSize])
    }

    func callAsFunction(_ input: MLXArray, time: MLXArray) -> MLXArray {
        let parts = MLX.split(
            modulation.asType(time.dtype) + time.expandedDimensions(axis: 1), parts: 2, axis: 1
        )
        let normalized = norm(input.asType(.float32)) * (1 + parts[1]) + parts[0]
        return head(normalized.asType(computePrecision.dtype(for: head.weight.dtype)))
    }
}

/// Native inference graph for the Base 4DAnyone checkpoint.
public final class FourDAnyoneTransformerModel: Module {
    public let configuration: Wan2TransformerConfiguration
    public let computePrecision: FourDAnyoneComputePrecision
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "text_embedding") var textEmbedding: FourDAnyoneProjection
    @ModuleInfo(key: "time_embedding") var timeEmbedding: FourDAnyoneProjection
    @ModuleInfo(key: "time_projection") var timeProjection: FourDAnyoneTimeProjection
    @ModuleInfo var blocks: [FourDAnyoneBlock]
    @ModuleInfo var head: FourDAnyoneHead
    @ModuleInfo(key: "viewpack_embedding") var viewPack: FourDAnyoneViewPack

    public init(
        configuration: Wan2TransformerConfiguration = Wan2TransformerConfiguration(),
        computePrecision: FourDAnyoneComputePrecision = .float32
    ) {
        precondition(configuration.patchSize == [1, 2, 2])
        precondition(configuration.inputChannels == configuration.outputChannels)
        precondition(!configuration.projectiveCameraConditioning)
        let headWidth = configuration.hiddenSize / configuration.headCount
        precondition(headWidth >= 6 && (headWidth / 3).isMultiple(of: 2))
        self.configuration = configuration
        self.computePrecision = computePrecision
        self._patchEmbedding.wrappedValue = Linear(configuration.inputChannels * 4, configuration.hiddenSize)
        self._textEmbedding.wrappedValue = FourDAnyoneProjection(
            input: configuration.textEmbeddingSize,
            hidden: configuration.hiddenSize,
            output: configuration.hiddenSize,
            computePrecision: computePrecision
        )
        self._timeEmbedding.wrappedValue = FourDAnyoneProjection(
            input: configuration.timestepFrequencySize,
            hidden: configuration.hiddenSize,
            output: configuration.hiddenSize,
            usesSiLU: true,
            computePrecision: computePrecision
        )
        self._timeProjection.wrappedValue = FourDAnyoneTimeProjection(configuration.hiddenSize)
        self._blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            FourDAnyoneBlock(configuration, computePrecision: computePrecision)
        }
        self._head.wrappedValue = FourDAnyoneHead(configuration, computePrecision: computePrecision)
        self._viewPack.wrappedValue = FourDAnyoneViewPack(configuration, computePrecision: computePrecision)
    }

    func patchify(_ latents: MLXArray) -> MLXArray {
        let patches = (0..<latents.dim(0)).map {
            Wan2PatchLayout.flatten(latents[$0], patchSize: configuration.patchSize).value
        }
        return patchEmbedding(MLX.stacked(patches, axis: 0)
            .asType(computePrecision.dtype(for: patchEmbedding.weight.dtype)))
    }

    func assemble(_ input: FourDAnyoneTransformerInput, grid: Wan2GridSize) -> MLXArray {
        var values = [patchify(input.latents), patchify(input.sources[0..<1])]
        if input.sources.dim(0) == 5 {
            values.append(viewPack(input.sources[1..<5], grid: grid))
        }
        let features = MLX.concatenated([input.poseFeatures, input.nullPoseFeatures], axis: 0)
            .transposed(0, 2, 3, 4, 1).reshaped(-1, grid.sequenceLength, configuration.hiddenSize)
        return MLX.concatenated(values, axis: 0)
            + features.asType(computePrecision.dtype(for: patchEmbedding.weight.dtype))
    }

    func embeddedTime(timestep: Float, targets: Int, packed: Int) -> MLXArray {
        // Round to the selected arithmetic dtype before the reference sinusoid.
        let times = MLXArray(Array(repeating: timestep, count: targets) + Array(repeating: Float(0), count: packed))
            .asType(computePrecision.dtype(for: patchEmbedding.weight.dtype)).asType(.float32).asArray(Float.self)
        let half = configuration.timestepFrequencySize / 2
        let values = times.flatMap { time -> [Float] in
            let angles = (0..<half).map { Double(time) * pow(10_000, -Double($0) / Double(half)) }
            return angles.map { Float(cos($0)) } + angles.map { Float(sin($0)) }
        }
        return timeEmbedding(MLXArray(values, [times.count, half * 2]))
    }

    public func callAsFunction(
        _ input: FourDAnyoneTransformerInput,
        maximumQueryTokens: Int = 512,
        checkCancellation: () throws -> Void = { try Task.checkCancellation() }
    ) throws -> MLXArray {
        let grid = try input.validate(configuration: configuration)
        guard maximumQueryTokens > 0 else {
            throw FourDAnyoneError.invalidInput("The attention query chunk must be positive.")
        }
        try checkCancellation()
        let targets = input.latents.dim(0)
        let packed = input.sources.dim(0) == 5 ? 2 : 1
        let context = textEmbedding(input.promptContext)
        let time = embeddedTime(timestep: input.timestep, targets: targets, packed: packed)
        let modulation = timeProjection(time).reshaped(targets + packed, 6, configuration.hiddenSize)
        let headWidth = configuration.hiddenSize / configuration.headCount
        let spatial = FourDAnyoneRoPE.prepare(grid: grid, headDimension: headWidth)
        let multiview = FourDAnyoneRoPE.prepare(
            grid: Wan2GridSize(frames: targets + packed, height: grid.height, width: grid.width),
            headDimension: headWidth
        )
        var hidden = assemble(input, grid: grid)
        eval(hidden, context, time, modulation)
        for block in blocks {
            try checkCancellation()
            hidden = block(
                hidden, context: context, time: modulation, grid: grid,
                spatialRoPE: spatial, multiviewRoPE: multiview, maximumQueryTokens: maximumQueryTokens
            )
            eval(hidden)
        }
        let patches = head(hidden[0..<targets], time: time[0..<targets])
        return patches.reshaped(
            targets, grid.frames, grid.height, grid.width, 1, 2, 2, configuration.outputChannels
        ).transposed(0, 7, 1, 4, 2, 5, 3, 6)
            .reshaped(targets, configuration.outputChannels, grid.frames, grid.height * 2, grid.width * 2)
    }
}
