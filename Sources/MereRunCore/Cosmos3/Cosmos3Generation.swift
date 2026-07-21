import Foundation
import MLX

public enum Cosmos3GenerationError: LocalizedError, Sendable {
    case invalidVisionShape([Int])
    case invalidConditionedVisionFrame(Int)
    case invalidActionShape([Int])
    case invalidConditionedActionFrame(Int)
    case missingActionDomain
    case invalidRawActionDimension(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidVisionShape(let shape):
            return "Cosmos3 vision latents must have shape [C,T,H,W]; received \(shape)."
        case .invalidConditionedVisionFrame(let index):
            return "Cosmos3 conditioned vision frame \(index) is outside the latent timeline."
        case .invalidActionShape(let shape):
            return "Cosmos3 action latents must have shape [T,action_dim]; received \(shape)."
        case .invalidConditionedActionFrame(let index):
            return "Cosmos3 conditioned action frame \(index) is outside the action timeline."
        case .missingActionDomain:
            return "Cosmos3 action latents require an embodiment domain."
        case .invalidRawActionDimension(let value):
            return "Cosmos3 raw action dimension must be positive and no larger than action_dim; received \(value)."
        }
    }
}

public struct Cosmos3DenoisingInput {
    public let tokenIDs: MLXArray
    public let visionLatents: MLXArray
    public let conditionedVisionFrames: [Int]
    public let timestep: Float
    public let fps: Float
    public let actionLatents: MLXArray?
    public let conditionedActionFrames: [Int]
    public let actionDomain: Cosmos3ActionDomain?
    public let rawActionDimension: Int?

    public init(
        tokenIDs: MLXArray,
        visionLatents: MLXArray,
        conditionedVisionFrames: [Int] = [],
        timestep: Float,
        fps: Float = 24,
        actionLatents: MLXArray? = nil,
        conditionedActionFrames: [Int] = [],
        actionDomain: Cosmos3ActionDomain? = nil,
        rawActionDimension: Int? = nil
    ) throws {
        guard visionLatents.ndim == 4 else {
            throw Cosmos3GenerationError.invalidVisionShape(visionLatents.shape)
        }
        let visionFrames = visionLatents.dim(1)
        if let invalid = conditionedVisionFrames.first(where: { $0 < 0 || $0 >= visionFrames }) {
            throw Cosmos3GenerationError.invalidConditionedVisionFrame(invalid)
        }
        if let actionLatents {
            guard actionLatents.ndim == 2 else {
                throw Cosmos3GenerationError.invalidActionShape(actionLatents.shape)
            }
            guard actionDomain != nil else {
                throw Cosmos3GenerationError.missingActionDomain
            }
            if let invalid = conditionedActionFrames.first(where: {
                $0 < 0 || $0 >= actionLatents.dim(0)
            }) {
                throw Cosmos3GenerationError.invalidConditionedActionFrame(invalid)
            }
            if let rawActionDimension,
               rawActionDimension < 1 || rawActionDimension > actionLatents.dim(1) {
                throw Cosmos3GenerationError.invalidRawActionDimension(rawActionDimension)
            }
        }
        self.tokenIDs = tokenIDs.reshaped(-1)
        self.visionLatents = visionLatents
        self.conditionedVisionFrames = Array(Set(conditionedVisionFrames)).sorted()
        self.timestep = timestep
        self.fps = fps
        self.actionLatents = actionLatents
        self.conditionedActionFrames = Array(Set(conditionedActionFrames)).sorted()
        self.actionDomain = actionDomain
        self.rawActionDimension = rawActionDimension
    }
}

public struct Cosmos3DenoisingPrediction {
    public let visionVelocity: MLXArray
    public let actionVelocity: MLXArray?

    public init(visionVelocity: MLXArray, actionVelocity: MLXArray?) {
        self.visionVelocity = visionVelocity
        self.actionVelocity = actionVelocity
    }
}

public struct Cosmos3VisionDenoisingItem {
    public let latents: MLXArray
    public let conditionedFrames: [Int]

    public init(
        latents: MLXArray,
        conditionedFrames: [Int] = []
    ) throws {
        guard latents.ndim == 4 else {
            throw Cosmos3GenerationError.invalidVisionShape(latents.shape)
        }
        if let invalid = conditionedFrames.first(where: {
            $0 < 0 || $0 >= latents.dim(1)
        }) {
            throw Cosmos3GenerationError.invalidConditionedVisionFrame(invalid)
        }
        self.latents = latents
        self.conditionedFrames = Array(Set(conditionedFrames)).sorted()
    }
}

extension Cosmos3OmniTransformerModel {
    /// NVIDIA's image-edit path packs the clean source and noisy target as
    /// independent vision items in the same generation stream. Every item
    /// advances the shared temporal mRoPE cursor; only the final target item
    /// contributes a velocity prediction.
    public func predictVisionItems(
        tokenIDs: MLXArray,
        items: [Cosmos3VisionDenoisingItem],
        timestep: Float,
        fps: Float = 24
    ) -> [MLXArray] {
        precondition(!items.isEmpty)
        let textTokens = tokenIDs.reshaped(-1)
        let textHidden = embedText(tokenIDs: textTokens)
        let textPositions = Cosmos3SequenceLayout.textPositionIDs(
            tokenCount: textTokens.size
        )
        var temporalOffset = Float(
            textPositions.nextTemporalOffset + configuration.temporalModalityMargin
        )
        var generationSegments: [MLXArray] = []
        var positionSegments = [textPositions.mlxArray]
        var packedItems: [(
            layout: Cosmos3VisionPatchLayout,
            noisyIndexes: [Int],
            generationOffset: Int
        )] = []
        var generationOffset = 0

        for item in items {
            let packed = Cosmos3VisionPatches.pack(
                item.latents,
                patchSize: configuration.latentPatchSize
            )
            let noisyFrames = Self.noisyIndexes(
                count: packed.layout.frames,
                conditioned: item.conditionedFrames
            )
            let tokenStride = packed.layout.patchHeight * packed.layout.patchWidth
            let noisyIndexes = noisyFrames.flatMap { frame in
                let start = frame * tokenStride
                return Array(start..<(start + tokenStride))
            }
            var hidden = visionInputProjection(packed.tokens)
            if !noisyIndexes.isEmpty {
                let values = MLX.full(
                    [noisyIndexes.count],
                    values: MLXArray(timestep * configuration.timestepScale)
                )
                hidden = hidden.at[MLXArray(noisyIndexes)].add(
                    timestepEmbedding(values).asType(hidden.dtype)
                )
            }
            generationSegments.append(hidden)
            let positions = Cosmos3SequenceLayout.vaePositionIDs(
                frames: packed.layout.frames,
                height: packed.layout.patchHeight,
                width: packed.layout.patchWidth,
                temporalOffset: temporalOffset,
                resetSpatialIndices: configuration.resetsSpatialPositionIDs,
                fps: configuration.enablesFPSModulation ? fps : nil,
                baseFPS: Float(configuration.baseFPS),
                temporalCompressionFactor: configuration.temporalCompressionFactor
            )
            positionSegments.append(positions.mlxArray)
            temporalOffset = Float(positions.nextTemporalOffset)
            packedItems.append((
                layout: packed.layout,
                noisyIndexes: noisyIndexes,
                generationOffset: generationOffset
            ))
            generationOffset += packed.layout.tokenCount
        }

        let output = self(
            understanding: textHidden,
            generation: MLX.concatenated(generationSegments, axis: 0),
            positionIDs: MLX.concatenated(positionSegments, axis: 1)
        ).generation
        return packedItems.map { packed in
            var velocity = MLX.zeros(
                [packed.layout.tokenCount, configuration.patchLatentDimension],
                dtype: output.dtype
            )
            if !packed.noisyIndexes.isEmpty {
                let indexes = packed.noisyIndexes.map {
                    packed.generationOffset + $0
                }
                let prediction = visionOutputProjection(output[MLXArray(indexes)])
                velocity = velocity.at[MLXArray(packed.noisyIndexes)].add(prediction)
            }
            return Cosmos3VisionPatches.unpack(
                velocity,
                layout: packed.layout,
                channels: configuration.latentChannels
            )
        }
    }

    public func predict(_ input: Cosmos3DenoisingInput) -> Cosmos3DenoisingPrediction {
        let textHidden = embedText(tokenIDs: input.tokenIDs)
        let vision = Cosmos3VisionPatches.pack(
            input.visionLatents,
            patchSize: configuration.latentPatchSize
        )
        let noisyVisionFrames = Self.noisyIndexes(
            count: vision.layout.frames,
            conditioned: input.conditionedVisionFrames
        )
        let visionTokenStride = vision.layout.patchHeight * vision.layout.patchWidth
        let noisyVisionTokenIndexes = noisyVisionFrames.flatMap { frame in
            let start = frame * visionTokenStride
            return Array(start..<(start + visionTokenStride))
        }
        var visionHidden = visionInputProjection(vision.tokens)
        if !noisyVisionTokenIndexes.isEmpty {
            let timestepValues = MLX.full(
                [noisyVisionTokenIndexes.count],
                values: MLXArray(input.timestep * configuration.timestepScale)
            )
            let timestepHidden = timestepEmbedding(timestepValues).asType(visionHidden.dtype)
            visionHidden = visionHidden.at[MLXArray(noisyVisionTokenIndexes)].add(timestepHidden)
        }

        var generationSegments = [visionHidden]
        var positionSegments: [MLXArray] = []
        let textPositions = Cosmos3SequenceLayout.textPositionIDs(
            tokenCount: input.tokenIDs.size
        )
        positionSegments.append(textPositions.mlxArray)
        let modalityOffset = Float(
            textPositions.nextTemporalOffset + configuration.temporalModalityMargin
        )
        positionSegments.append(Cosmos3SequenceLayout.vaePositionIDs(
            frames: vision.layout.frames,
            height: vision.layout.patchHeight,
            width: vision.layout.patchWidth,
            temporalOffset: modalityOffset,
            resetSpatialIndices: configuration.resetsSpatialPositionIDs,
            fps: configuration.enablesFPSModulation ? input.fps : nil,
            baseFPS: Float(configuration.baseFPS),
            temporalCompressionFactor: configuration.temporalCompressionFactor
        ).mlxArray)

        var noisyActionIndexes: [Int] = []
        var actionCount = 0
        if let actionLatents = input.actionLatents, let actionDomain = input.actionDomain {
            actionCount = actionLatents.dim(0)
            noisyActionIndexes = Self.noisyIndexes(
                count: actionCount,
                conditioned: input.conditionedActionFrames
            )
            let domainIDs = MLXArray(
                Array(repeating: Int32(actionDomain.domainID), count: actionCount)
            )
            var actionHidden = actionInputProjection(actionLatents, domainIDs: domainIDs)
                + actionModalityEmbedding
            if !noisyActionIndexes.isEmpty {
                let timestepValues = MLX.full(
                    [noisyActionIndexes.count],
                    values: MLXArray(input.timestep * configuration.timestepScale)
                )
                let timestepHidden = timestepEmbedding(timestepValues).asType(actionHidden.dtype)
                actionHidden = actionHidden.at[MLXArray(noisyActionIndexes)].add(timestepHidden)
            }
            generationSegments.append(actionHidden)
            positionSegments.append(Cosmos3SequenceLayout.vaePositionIDs(
                frames: actionCount,
                height: 1,
                width: 1,
                temporalOffset: modalityOffset,
                resetSpatialIndices: configuration.resetsSpatialPositionIDs,
                fps: configuration.enablesFPSModulation ? input.fps : nil,
                baseFPS: Float(configuration.baseFPS),
                temporalCompressionFactor: 1,
                baseTemporalCompressionFactor: configuration.temporalCompressionFactor,
                startFrameOffset: 1
            ).mlxArray)
        }

        let generationHidden = MLX.concatenated(generationSegments, axis: 0)
        let positionIDs = MLX.concatenated(positionSegments, axis: 1)
        let output = self(
            understanding: textHidden,
            generation: generationHidden,
            positionIDs: positionIDs
        ).generation

        var packedVisionVelocity = MLX.zeros(
            vision.tokens.shape,
            dtype: output.dtype
        )
        if !noisyVisionTokenIndexes.isEmpty {
            let noisyHidden = output[MLXArray(noisyVisionTokenIndexes)]
            let noisyPredictions = visionOutputProjection(noisyHidden)
            packedVisionVelocity = packedVisionVelocity
                .at[MLXArray(noisyVisionTokenIndexes)]
                .add(noisyPredictions)
        }
        let visionVelocity = Cosmos3VisionPatches.unpack(
            packedVisionVelocity,
            layout: vision.layout,
            channels: configuration.latentChannels
        )

        var actionVelocity: MLXArray?
        if actionCount > 0, let actionDomain = input.actionDomain {
            let actionDimension = configuration.actionDimension!
            var velocity = MLX.zeros([actionCount, actionDimension], dtype: output.dtype)
            if !noisyActionIndexes.isEmpty {
                let generationOffset = vision.layout.tokenCount
                let hiddenIndexes = noisyActionIndexes.map { generationOffset + $0 }
                let noisyHidden = output[MLXArray(hiddenIndexes)]
                let domainIDs = MLXArray(
                    Array(repeating: Int32(actionDomain.domainID), count: noisyActionIndexes.count)
                )
                var predictions = actionOutputProjection(
                    noisyHidden,
                    domainIDs: domainIDs
                )
                if let rawDimension = input.rawActionDimension,
                   rawDimension < actionDimension {
                    let channelMask = MLXArray(
                        Array(repeating: Float(1), count: rawDimension)
                            + Array(repeating: Float(0), count: actionDimension - rawDimension)
                    ).asType(predictions.dtype)
                    predictions = predictions * channelMask
                }
                velocity = velocity.at[MLXArray(noisyActionIndexes)].add(predictions)
            }
            actionVelocity = velocity
        }
        return Cosmos3DenoisingPrediction(
            visionVelocity: visionVelocity,
            actionVelocity: actionVelocity
        )
    }

    private static func noisyIndexes(count: Int, conditioned: [Int]) -> [Int] {
        let conditionedSet = Set(conditioned)
        return (0..<count).filter { !conditionedSet.contains($0) }
    }
}
