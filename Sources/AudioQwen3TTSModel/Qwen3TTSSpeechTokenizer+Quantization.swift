import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunKVCache

final class EuclideanCodebook: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    let dim: Int
    let codebookSize: Int

    init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self.codebookSize = codebookSize
        self._embed.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dim)
    }

    func encode(_ vectors: MLXArray) -> MLXArray {
        var targetShape: [Int] = []
        if vectors.ndim > 1 {
            targetShape.reserveCapacity(vectors.ndim - 1)
            for axis in 0..<(vectors.ndim - 1) {
                targetShape.append(vectors.dim(axis))
            }
        }

        let flat = vectors.reshaped(-1, vectors.dim(-1)).asType(.float32)
        let indices = MLXArray((0..<codebookSize).map { Int32($0) }).asType(.int32)
        let table = embed(indices).asType(.float32) // [K, D]
        let c2 = (table * table).sum(axis: -1) / 2
        let dot = MLX.matmul(flat, table.transposed(1, 0))
        let nearest = (c2[.newAxis, 0...] - dot).argMin(axis: -1).asType(.int32)
        return nearest.reshaped(targetShape)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        embed(codes)
    }
}

final class VectorQuantization: Module {
    @ModuleInfo(key: "project_out") var projectOut: Linear?
    @ModuleInfo(key: "codebook") var codebook: EuclideanCodebook
    let codebookSize: Int

    init(dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        let cbDim = codebookDim ?? dim
        if cbDim != dim {
            self._projectIn.wrappedValue = Linear(dim, cbDim)
            self._projectOut.wrappedValue = Linear(cbDim, dim)
        } else {
            self._projectIn.wrappedValue = nil
            self._projectOut.wrappedValue = nil
        }
        self._codebook.wrappedValue = EuclideanCodebook(dim: cbDim, codebookSize: codebookSize)
        self.codebookSize = codebookSize
    }

    @ModuleInfo(key: "project_in") var projectIn: Linear?

    func encode(_ xs: MLXArray) -> MLXArray {
        var inputs = xs.transposed(0, 2, 1) // [B, T, C]
        if let proj = projectIn {
            inputs = proj(inputs)
        }
        return codebook.encode(inputs)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = codebook.decode(codes) // [B, T, codebook_dim]
        if let proj = projectOut {
            quantized = proj(quantized)
        }
        return quantized.transposed(0, 2, 1)
    }
}

final class ResidualVectorQuantization: Module {
    @ModuleInfo(key: "layers") var layers: [VectorQuantization]

    init(numQuantizers: Int, dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        self._layers.wrappedValue = (0..<numQuantizers).map { _ in
            VectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var residual = xs
        var codes: [MLXArray] = []
        codes.reserveCapacity(layers.count)

        for layer in layers {
            let indices = layer.encode(residual) // [B, T]
            let quantized = layer.decode(indices) // [B, C, T]
            residual = (residual.asType(.float32) - quantized.asType(.float32)).asType(residual.dtype)
            codes.append(indices)
        }

        return MLX.stacked(codes, axis: 0) // [Q, B, T]
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = MLXArray.zeros([codes.dim(1), layers[0].codebook.dim, codes.dim(2)])
        for idx in 0..<layers.count {
            let layerCodes = codes[idx]
            quantized = quantized + layers[idx].decode(layerCodes)
        }
        return quantized
    }
}

final class ResidualVectorQuantizer: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Conv1d?
    @ModuleInfo(key: "output_proj") var outputProj: Conv1d?
    @ModuleInfo(key: "vq") var vq: ResidualVectorQuantization

    let nQ: Int
    let dimension: Int
    let inputDimension: Int
    let outputDimension: Int
    let bins: Int

    init(
        dimension: Int,
        inputDimension: Int?,
        outputDimension: Int?,
        nQ: Int,
        bins: Int,
        forceProjection: Bool
    ) {
        self.nQ = nQ
        self.dimension = dimension
        self.inputDimension = inputDimension ?? dimension
        self.outputDimension = outputDimension ?? dimension
        self.bins = bins

        if self.inputDimension == dimension && !forceProjection {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = Conv1d(inputChannels: self.inputDimension, outputChannels: dimension, kernelSize: 1, stride: 1, padding: 0, dilation: 1, groups: 1, bias: false)
        }

        if self.outputDimension == dimension && !forceProjection {
            self._outputProj.wrappedValue = nil
        } else {
            self._outputProj.wrappedValue = Conv1d(inputChannels: dimension, outputChannels: self.outputDimension, kernelSize: 1, stride: 1, padding: 0, dilation: 1, groups: 1, bias: false)
        }

        self._vq.wrappedValue = ResidualVectorQuantization(numQuantizers: nQ, dim: dimension, codebookSize: bins)
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var hidden = xs
        if let inputProj = inputProj {
            hidden = hidden.transposed(0, 2, 1)
            hidden = inputProj(hidden)
            hidden = hidden.transposed(0, 2, 1)
        }
        return vq.encode(hidden).transposed(1, 0, 2) // [B, Q, T]
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let codesTransposed = codes.transposed(1, 0, 2)
        var quantized = vq.decode(codesTransposed)
        if let outProj = outputProj {
            quantized = quantized.transposed(0, 2, 1)
            quantized = outProj(quantized)
            quantized = quantized.transposed(0, 2, 1)
        }
        return quantized
    }
}

final class SplitResidualVectorQuantizer: Module {
    @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

    let nQSemantic: Int
    let nQAcoustic: Int

    init(
        nQ: Int,
        nQSemantic: Int,
        dimension: Int,
        inputDimension: Int?,
        outputDimension: Int?,
        bins: Int
    ) {
        self.nQSemantic = nQSemantic
        self.nQAcoustic = nQ - nQSemantic
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            nQ: nQSemantic,
            bins: bins,
            forceProjection: true
        )
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            nQ: nQ - nQSemantic,
            bins: bins,
            forceProjection: true
        )
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var codes = rvqFirst.encode(xs)
        if nQAcoustic > 0 {
            let restCodes = rvqRest.encode(xs)
            codes = MLX.concatenated([codes, restCodes], axis: 1)
        }
        return codes
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = rvqFirst.decode(codes[0..., 0..<nQSemantic, 0...])
        if codes.dim(1) > nQSemantic {
            let rest = rvqRest.decode(codes[0..., nQSemantic..., 0...])
            quantized = quantized + rest
        }
        return quantized
    }
}

// MARK: - Decoder Blocks
