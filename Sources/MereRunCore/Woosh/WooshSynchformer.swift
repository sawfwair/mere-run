import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public final class WooshSynchformer {
    private let model: WooshSynchformerModel

    public init(resources: WooshSynchformerResources) throws {
        let missing = resources.missingFiles()
        guard missing.isEmpty else {
            throw WooshError.missingFiles(missing)
        }

        let model = WooshSynchformerModel()
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.weightsURL,
            to: model,
            dtype: .float32,
            verify: .none,
            mapper: Self.mapWeights
        )
        self.model = model
    }

    public func extractFeatures(
        videoURL: URL,
        durationSeconds: Float,
        targetFrameRate: Int = 24,
        segmentBatchSize: Int = 1
    ) throws -> MLXArray {
        guard durationSeconds > 0 else {
            throw WooshError.invalidVideoShape([0])
        }
        guard targetFrameRate > 0, segmentBatchSize > 0 else {
            throw WooshError.invalidVideoShape([targetFrameRate, segmentBatchSize])
        }

        let framesDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-woosh-sync-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: framesDirectory) }
        let sequence = try MediaVideoIO.extractFrames(from: videoURL, into: framesDirectory)
        let frames = try Self.loadSyncFrames(
            sequence: sequence,
            durationSeconds: durationSeconds,
            targetFrameRate: targetFrameRate
        )
        return try extractFeatures(
            framesCHW: frames.values,
            frameCount: frames.frameCount,
            segmentBatchSize: segmentBatchSize
        )
    }

    public func extractFeatures(
        framesCHW: [Float],
        frameCount: Int,
        segmentBatchSize: Int = 1
    ) throws -> MLXArray {
        let spatialSize = WooshSynchformerConfig.imageSize * WooshSynchformerConfig.imageSize
        let frameStride = WooshSynchformerConfig.channels * spatialSize
        guard frameCount > 0, framesCHW.count == frameCount * frameStride else {
            throw WooshError.invalidVideoShape([frameCount, framesCHW.count])
        }
        guard segmentBatchSize > 0 else {
            throw WooshError.invalidVideoShape([segmentBatchSize])
        }

        let paddedFrames = frameCount + WooshSynchformerConfig.segmentSize
        let segmentCount = ((paddedFrames - WooshSynchformerConfig.segmentSize) / WooshSynchformerConfig.stepSize) + 1
        var outputs: [MLXArray] = []
        outputs.reserveCapacity((segmentCount + segmentBatchSize - 1) / segmentBatchSize)
        let zeroChannel = [Float](repeating: 0, count: spatialSize)

        var segmentStart = 0
        while segmentStart < segmentCount {
            let currentBatch = min(segmentBatchSize, segmentCount - segmentStart)
            var segmentValues: [Float] = []
            segmentValues.reserveCapacity(
                currentBatch * WooshSynchformerConfig.channels * WooshSynchformerConfig.segmentSize * spatialSize
            )

            for batchOffset in 0..<currentBatch {
                let segmentIndex = segmentStart + batchOffset
                let paddedStart = segmentIndex * WooshSynchformerConfig.stepSize
                for channel in 0..<WooshSynchformerConfig.channels {
                    for localFrame in 0..<WooshSynchformerConfig.segmentSize {
                        let originalFrame = paddedStart + localFrame - WooshSynchformerConfig.segmentSize
                        if originalFrame < 0 {
                            segmentValues.append(contentsOf: zeroChannel)
                        } else {
                            let sourceOffset = (originalFrame * frameStride) + (channel * spatialSize)
                            segmentValues.append(contentsOf: framesCHW[sourceOffset..<(sourceOffset + spatialSize)])
                        }
                    }
                }
            }

            let input = MLXArray(segmentValues)
                .reshaped(
                    currentBatch,
                    1,
                    WooshSynchformerConfig.channels,
                    WooshSynchformerConfig.segmentSize,
                    WooshSynchformerConfig.imageSize,
                    WooshSynchformerConfig.imageSize
                )
                .asType(.float32)
            let batchFeatures = model(input)
                .reshaped(1, currentBatch * WooshSynchformerConfig.outputFramesPerSegment, WooshSynchformerConfig.dim)
            MLX.eval(batchFeatures)
            outputs.append(batchFeatures.asType(.float32))
            segmentStart += currentBatch
        }

        let features = outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 1)
        return features[0..., 0..<frameCount, 0...]
    }

    private static func mapWeights(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        guard key.hasPrefix("vfeat_extractor.") else {
            return []
        }
        if key.hasPrefix("vfeat_extractor.patch_embed.") {
            return []
        }
        if key == "vfeat_extractor.patch_embed_3d.proj.weight" {
            let transposed = value.transposed(0, 2, 3, 4, 1)
            return [(key, transposed.reshaped(-1).reshaped(transposed.shape))]
        }
        return [(key, value)]
    }

    private static func loadSyncFrames(
        sequence: VideoFrameSequence,
        durationSeconds: Float,
        targetFrameRate: Int
    ) throws -> (values: [Float], frameCount: Int) {
        guard !sequence.frameURLs.isEmpty else {
            throw WooshError.invalidVideoShape([0])
        }

        let indices = downsampledFrameIndices(
            sourceFrameRate: sequence.fps,
            sourceFrameCount: sequence.frameURLs.count,
            durationSeconds: durationSeconds,
            targetFrameRate: Double(targetFrameRate)
        )
        guard !indices.isEmpty else {
            throw WooshError.invalidVideoShape([sequence.frameURLs.count])
        }

        let frameStride = WooshSynchformerConfig.channels
            * WooshSynchformerConfig.imageSize
            * WooshSynchformerConfig.imageSize
        var values: [Float] = []
        values.reserveCapacity(indices.count * frameStride)
        for index in indices {
            let image = try MediaImageIO.decode(sequence.frameURLs[index])
            let cropped = try MediaImageIO.centerCropped(
                image,
                width: WooshSynchformerConfig.imageSize,
                height: WooshSynchformerConfig.imageSize
            )
            values.append(contentsOf: MediaImageIO.rgbCHWFloat(cropped, normalizedToMinusOneToOne: true))
        }
        return (values, indices.count)
    }

    private static func downsampledFrameIndices(
        sourceFrameRate: Double,
        sourceFrameCount: Int,
        durationSeconds: Float,
        targetFrameRate: Double
    ) -> [Int] {
        let sourceFPS = max(1, sourceFrameRate)
        let sourceLimit = min(
            sourceFrameCount,
            max(1, Int((Double(durationSeconds) * sourceFPS).rounded(.up)))
        )
        let roundedSourceFPS = Int(sourceFPS.rounded())
        let roundedTargetFPS = Int(targetFrameRate.rounded())
        if roundedTargetFPS > 0,
           Swift.abs(sourceFPS - Double(roundedSourceFPS)) < 0.01,
           roundedSourceFPS >= roundedTargetFPS,
           roundedSourceFPS % roundedTargetFPS == 0 {
            let factor = max(1, roundedSourceFPS / roundedTargetFPS)
            return stride(from: 0, to: sourceLimit, by: factor).map { $0 }
        }

        var indices: [Int] = []
        var nextPTS = 0.0
        let step = 1.0 / max(1, targetFrameRate)
        for index in 0..<sourceLimit {
            let pts = Double(index) / sourceFPS
            if pts + 1e-9 >= nextPTS {
                indices.append(index)
                nextPTS += step
            }
        }
        if indices.isEmpty {
            indices.append(0)
        }
        return indices
    }
}

private enum WooshSynchformerConfig {
    static let dim = 768
    static let channels = 3
    static let imageSize = 224
    static let patchSize = 16
    static let temporalPatchSize = 2
    static let temporalResolution = 8
    static let spatialPatches = 14 * 14
    static let segmentSize = 16
    static let stepSize = 8
    static let outputFramesPerSegment = 8
    static let heads = 12
    static let headDim = 64
}

private final class WooshSynchformerModel: Module {
    @ModuleInfo(key: "vfeat_extractor") var vfeatExtractor: WooshMotionFormer

    override init() {
        self._vfeatExtractor.wrappedValue = WooshMotionFormer()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        vfeatExtractor(x)
    }
}

private final class WooshMotionFormer: Module {
    @ModuleInfo(key: "patch_embed_3d") var patchEmbed3D: WooshMotionFormerPatchEmbed3D
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ParameterInfo(key: "temp_embed") var tempEmbed: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [WooshMotionFormerBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "spatial_attn_agg") var spatialAttentionAggregator: WooshSynchformerSpatialAggregator

    override init() {
        self._patchEmbed3D.wrappedValue = WooshMotionFormerPatchEmbed3D()
        self._clsToken.wrappedValue = MLXArray.zeros([1, 1, WooshSynchformerConfig.dim], dtype: .float32)
        self._posEmbed.wrappedValue = MLXArray.zeros([
            1,
            WooshSynchformerConfig.spatialPatches + 1,
            WooshSynchformerConfig.dim,
        ], dtype: .float32)
        self._tempEmbed.wrappedValue = MLXArray.zeros([
            1,
            WooshSynchformerConfig.temporalResolution,
            WooshSynchformerConfig.dim,
        ], dtype: .float32)
        self._blocks.wrappedValue = (0..<12).map { _ in WooshMotionFormerBlock() }
        self._norm.wrappedValue = LayerNorm(dimensions: WooshSynchformerConfig.dim, eps: 1e-6)
        self._spatialAttentionAggregator.wrappedValue = WooshSynchformerSpatialAggregator()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard x.ndim == 6 else {
            preconditionFailure("Expected Synchformer input [batch, segments, channels, frames, height, width].")
        }
        let batch = x.dim(0)
        let segments = x.dim(1)
        var hidden = x.reshaped(
            batch * segments,
            WooshSynchformerConfig.channels,
            WooshSynchformerConfig.segmentSize,
            WooshSynchformerConfig.imageSize,
            WooshSynchformerConfig.imageSize
        )
        hidden = patchEmbed3D(hidden)
        let flatBatch = hidden.dim(0)
        let cls = MLX.broadcast(clsToken, to: [flatBatch, 1, WooshSynchformerConfig.dim])
        hidden = MLX.concatenated([cls, hidden], axis: 1)
        hidden = hidden + positionalEmbedding().asType(hidden.dtype)

        for block in blocks {
            hidden = block(hidden)
        }

        hidden = hidden[0..., 1..., 0...]
        hidden = norm(hidden)
        hidden = hidden
            .transposed(0, 2, 1)
            .reshaped(
                flatBatch,
                WooshSynchformerConfig.dim,
                WooshSynchformerConfig.temporalResolution,
                WooshSynchformerConfig.imageSize / WooshSynchformerConfig.patchSize,
                WooshSynchformerConfig.imageSize / WooshSynchformerConfig.patchSize
            )
        hidden = spatialAttentionAggregator(hidden)
        return hidden.reshaped(batch, segments, WooshSynchformerConfig.outputFramesPerSegment, WooshSynchformerConfig.dim)
    }

    private func positionalEmbedding() -> MLXArray {
        let cls = posEmbed[0..., 0..<1, 0...]
        let spatial = posEmbed[0..., 1..., 0...]
        let tiledSpatial = MLX.tiled(spatial, repetitions: [1, WooshSynchformerConfig.temporalResolution, 1])
        let temporal = MLX.broadcast(
            tempEmbed.expandedDimensions(axis: 2),
            to: [
                1,
                WooshSynchformerConfig.temporalResolution,
                WooshSynchformerConfig.spatialPatches,
                WooshSynchformerConfig.dim,
            ]
        ).reshaped(1, WooshSynchformerConfig.temporalResolution * WooshSynchformerConfig.spatialPatches, WooshSynchformerConfig.dim)
        return MLX.concatenated([cls, tiledSpatial + temporal], axis: 1)
    }
}

private final class WooshMotionFormerPatchEmbed3D: Module {
    @ModuleInfo(key: "proj") var projection: Conv3d

    override init() {
        self._projection.wrappedValue = Conv3d(
            inputChannels: WooshSynchformerConfig.channels,
            outputChannels: WooshSynchformerConfig.dim,
            kernelSize: IntOrTriple([
                WooshSynchformerConfig.temporalPatchSize,
                WooshSynchformerConfig.patchSize,
                WooshSynchformerConfig.patchSize,
            ]),
            stride: IntOrTriple([
                WooshSynchformerConfig.temporalPatchSize,
                WooshSynchformerConfig.patchSize,
                WooshSynchformerConfig.patchSize,
            ]),
            padding: .init(0),
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let hidden = projection(x.transposed(0, 2, 3, 4, 1))
        return hidden.reshaped(
            batch,
            WooshSynchformerConfig.temporalResolution * WooshSynchformerConfig.spatialPatches,
            WooshSynchformerConfig.dim
        )
    }
}

private enum WooshMotionFormerAttentionMode {
    case time
    case space
}

private final class WooshMotionFormerDividedAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    private let scale: Float

    override init() {
        self.scale = 1.0 / sqrt(Float(WooshSynchformerConfig.headDim))
        self._qkv.wrappedValue = Linear(WooshSynchformerConfig.dim, WooshSynchformerConfig.dim * 3, bias: true)
        self._proj.wrappedValue = Linear(WooshSynchformerConfig.dim, WooshSynchformerConfig.dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, mode: WooshMotionFormerAttentionMode) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let qkvParts = MLX.split(qkv(x), parts: 3, axis: -1)
        let q = qkvParts[0]
            .reshaped(batch, sequence, WooshSynchformerConfig.heads, WooshSynchformerConfig.headDim)
            .transposed(0, 2, 1, 3)
        let k = qkvParts[1]
            .reshaped(batch, sequence, WooshSynchformerConfig.heads, WooshSynchformerConfig.headDim)
            .transposed(0, 2, 1, 3)
        let v = qkvParts[2]
            .reshaped(batch, sequence, WooshSynchformerConfig.heads, WooshSynchformerConfig.headDim)
            .transposed(0, 2, 1, 3)

        let clsQ = q[0..., 0..., 0..<1, 0...]
        let clsK = k[0..., 0..., 0..<1, 0...]
        let clsV = v[0..., 0..., 0..<1, 0...]
        let patchQ = q[0..., 0..., 1..., 0...]
        let patchK = k[0..., 0..., 1..., 0...]
        let patchV = v[0..., 0..., 1..., 0...]
        let clsOut = MLXFast.scaledDotProductAttention(
            queries: clsQ,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )

        let patchOut: MLXArray
        switch mode {
        case .time:
            patchOut = dividedAttentionOverTime(
                q: patchQ,
                k: patchK,
                v: patchV,
                clsK: clsK,
                clsV: clsV
            )
        case .space:
            patchOut = dividedAttentionOverSpace(
                q: patchQ,
                k: patchK,
                v: patchV,
                clsK: clsK,
                clsV: clsV
            )
        }

        let attended = MLX.concatenated([clsOut, patchOut], axis: 2)
        return proj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, WooshSynchformerConfig.dim))
    }

    private func dividedAttentionOverTime(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        clsK: MLXArray,
        clsV: MLXArray
    ) -> MLXArray {
        let batch = q.dim(0)
        let heads = q.dim(1)
        let frames = WooshSynchformerConfig.temporalResolution
        let patches = WooshSynchformerConfig.spatialPatches
        let headDim = WooshSynchformerConfig.headDim
        let query = q
            .reshaped(batch, heads, frames, patches, headDim)
            .transposed(0, 1, 3, 2, 4)
            .reshaped(batch * heads * patches, frames, headDim)
        let key = k
            .reshaped(batch, heads, frames, patches, headDim)
            .transposed(0, 1, 3, 2, 4)
            .reshaped(batch * heads * patches, frames, headDim)
        let value = v
            .reshaped(batch, heads, frames, patches, headDim)
            .transposed(0, 1, 3, 2, 4)
            .reshaped(batch * heads * patches, frames, headDim)
        let repeatedK = repeatCLS(clsK, repeatCount: patches)
        let repeatedV = repeatCLS(clsV, repeatCount: patches)
        let attended = attention3D(
            query,
            key: MLX.concatenated([repeatedK, key], axis: 1),
            value: MLX.concatenated([repeatedV, value], axis: 1)
        )
        return attended
            .reshaped(batch, heads, patches, frames, headDim)
            .transposed(0, 1, 3, 2, 4)
            .reshaped(batch, heads, frames * patches, headDim)
    }

    private func dividedAttentionOverSpace(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        clsK: MLXArray,
        clsV: MLXArray
    ) -> MLXArray {
        let batch = q.dim(0)
        let heads = q.dim(1)
        let frames = WooshSynchformerConfig.temporalResolution
        let patches = WooshSynchformerConfig.spatialPatches
        let headDim = WooshSynchformerConfig.headDim
        let query = q.reshaped(batch * heads * frames, patches, headDim)
        let key = k.reshaped(batch * heads * frames, patches, headDim)
        let value = v.reshaped(batch * heads * frames, patches, headDim)
        let repeatedK = repeatCLS(clsK, repeatCount: frames)
        let repeatedV = repeatCLS(clsV, repeatCount: frames)
        let attended = attention3D(
            query,
            key: MLX.concatenated([repeatedK, key], axis: 1),
            value: MLX.concatenated([repeatedV, value], axis: 1)
        )
        return attended.reshaped(batch, heads, frames * patches, headDim)
    }

    private func repeatCLS(_ cls: MLXArray, repeatCount: Int) -> MLXArray {
        let batch = cls.dim(0)
        let heads = cls.dim(1)
        return MLX.broadcast(
            cls.expandedDimensions(axis: 2),
            to: [batch, heads, repeatCount, 1, WooshSynchformerConfig.headDim]
        ).reshaped(batch * heads * repeatCount, 1, WooshSynchformerConfig.headDim)
    }

    private func attention3D(_ query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: query.expandedDimensions(axis: 1),
            keys: key.expandedDimensions(axis: 1),
            values: value.expandedDimensions(axis: 1),
            scale: scale,
            mask: .none
        ).squeezed(axis: 1)
    }
}

private final class WooshMotionFormerMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    override init() {
        self._fc1.wrappedValue = Linear(WooshSynchformerConfig.dim, WooshSynchformerConfig.dim * 4, bias: true)
        self._fc2.wrappedValue = Linear(WooshSynchformerConfig.dim * 4, WooshSynchformerConfig.dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

private final class WooshMotionFormerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: WooshMotionFormerDividedAttention
    @ModuleInfo(key: "timeattn") var timeAttention: WooshMotionFormerDividedAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: WooshMotionFormerMLP
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    override init() {
        self._norm1.wrappedValue = LayerNorm(dimensions: WooshSynchformerConfig.dim, eps: 1e-6)
        self._attn.wrappedValue = WooshMotionFormerDividedAttention()
        self._timeAttention.wrappedValue = WooshMotionFormerDividedAttention()
        self._norm2.wrappedValue = LayerNorm(dimensions: WooshSynchformerConfig.dim, eps: 1e-6)
        self._mlp.wrappedValue = WooshMotionFormerMLP()
        self._norm3.wrappedValue = LayerNorm(dimensions: WooshSynchformerConfig.dim, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let timeResidual = x + timeAttention(norm3(x), mode: .time)
        let spaceResidual = timeResidual + attn(norm1(timeResidual), mode: .space)
        return spaceResidual + mlp(norm2(spaceResidual))
    }
}

private final class WooshSynchformerMultiheadAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    private let scale: Float

    override init() {
        self.scale = 1.0 / sqrt(Float(WooshSynchformerConfig.headDim))
        self._inProjWeight.wrappedValue = MLXArray.zeros([
            WooshSynchformerConfig.dim * 3,
            WooshSynchformerConfig.dim,
        ], dtype: .float32)
        self._inProjBias.wrappedValue = MLXArray.zeros([WooshSynchformerConfig.dim * 3], dtype: .float32)
        self._outProj.wrappedValue = Linear(WooshSynchformerConfig.dim, WooshSynchformerConfig.dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let projected = MLX.matmul(x, inProjWeight.T) + inProjBias
        let qkv = MLX.split(projected, parts: 3, axis: -1)
        let q = qkv[0]
            .reshaped(batch, sequence, WooshSynchformerConfig.heads, WooshSynchformerConfig.headDim)
            .transposed(0, 2, 1, 3)
        let k = qkv[1]
            .reshaped(batch, sequence, WooshSynchformerConfig.heads, WooshSynchformerConfig.headDim)
            .transposed(0, 2, 1, 3)
        let v = qkv[2]
            .reshaped(batch, sequence, WooshSynchformerConfig.heads, WooshSynchformerConfig.headDim)
            .transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )
        return outProj(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, WooshSynchformerConfig.dim))
    }
}

private final class WooshSynchformerSpatialAggregator: Module {
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ModuleInfo(key: "self_attn") var selfAttention: WooshSynchformerMultiheadAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    override init() {
        self._clsToken.wrappedValue = MLXArray.zeros([1, 1, WooshSynchformerConfig.dim], dtype: .float32)
        self._selfAttention.wrappedValue = WooshSynchformerMultiheadAttention()
        self._linear1.wrappedValue = Linear(WooshSynchformerConfig.dim, WooshSynchformerConfig.dim * 4, bias: true)
        self._linear2.wrappedValue = Linear(WooshSynchformerConfig.dim * 4, WooshSynchformerConfig.dim, bias: true)
        self._norm1.wrappedValue = LayerNorm(dimensions: WooshSynchformerConfig.dim, eps: 1e-6)
        self._norm2.wrappedValue = LayerNorm(dimensions: WooshSynchformerConfig.dim, eps: 1e-6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSegments = x.dim(0)
        let frames = x.dim(2)
        let height = x.dim(3)
        let width = x.dim(4)
        var hidden = x
            .transposed(0, 2, 3, 4, 1)
            .reshaped(batchSegments * frames, height * width, WooshSynchformerConfig.dim)
        let cls = MLX.broadcast(clsToken, to: [hidden.dim(0), 1, WooshSynchformerConfig.dim])
        hidden = MLX.concatenated([cls, hidden], axis: 1)
        hidden = hidden + selfAttention(norm1(hidden))
        hidden = hidden + linear2(gelu(linear1(norm2(hidden))))
        return hidden[0..., 0, 0...].reshaped(batchSegments, frames, WooshSynchformerConfig.dim)
    }
}
