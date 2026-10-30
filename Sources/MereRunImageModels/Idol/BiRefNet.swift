// Native adaptation of speridlabs/idol-sdk 4702a7f (Apache-2.0).
import MLX
import MLXNN

/// Swin-L BiRefNet with multiscale image features and modulated deformable ASPP.
/// Input and output use NHWC. The production input is ImageNet-normalized 1024px RGB.
public final class BiRefNet {
    public let configuration: BiRefNetConfiguration
    private let weights: BiRefNetWeights
    public init(weights: [String: MLXArray], configuration: BiRefNetConfiguration = .init(), dtype: DType = .float16) throws {
        try configuration.validate()
        guard dtype == .float16 || dtype == .float32 else { throw IdolError.invalidConfiguration("BiRefNet uses FP16 or FP32.") }
        self.configuration = configuration
        self.weights = try BiRefNetWeights(weights, configuration: configuration, dtype: dtype)
    }

    public func callAsFunction(_ input: MLXArray, progress: ((String) -> Void)? = nil) throws -> MLXArray {
        guard input.ndim == 4, input.dim(0) == 1, input.dim(3) == 3,
              input.dim(1) >= 64, input.dim(2) >= 64, input.dim(1) % 64 == 0, input.dim(2) % 64 == 0 else {
            throw IdolError.invalidInput("BiRefNet expects one RGB image with dimensions divisible by 64.")
        }
        let input = input.asType(weights.dtype)
        let swin = BiRefNetSwin(weights: weights, configuration: configuration)
        progress?("BiRefNet Swin-L full-resolution backbone")
        let full = try swin(input)
        progress?("BiRefNet Swin-L half-resolution backbone")
        let half = try swin(BiRefNetOps.resize(input, height: input.dim(1) / 2, width: input.dim(2) / 2))
        let features = full.indices.map { index in
            concatenated([full[index], BiRefNetOps.resize(half[index], height: full[index].dim(1), width: full[index].dim(2))], axis: -1)
        }
        let deepHeight = features[3].dim(1), deepWidth = features[3].dim(2)
        let context = concatenated(features.map { BiRefNetOps.resize($0, height: deepHeight, width: deepWidth) }, axis: -1)
        progress?("BiRefNet deformable context squeeze")
        var hidden = try decoderBlock(context, "squeeze_module.0")
        for stage in 0..<4 {
            try Task.checkCancellation()
            let number = 4 - stage
            progress?("BiRefNet deformable decoder \(number)")
            let patches = BiRefNetOps.imagePatches(input, height: hidden.dim(1), width: hidden.dim(2))
            let injection = simpleConvs(patches, "decoder.ipt_blk\(5 - stage)")
            hidden = try decoderBlock(concatenated([hidden, injection], axis: -1), "decoder.decoder_block\(number)")
            if number > 1 {
                let gradient = relu(weights.batchNorm(weights.conv(hidden, "decoder.gdt_convs_\(number).0", padding: 1),
                                                       "decoder.gdt_convs_\(number).1"))
                hidden = hidden * sigmoid(weights.conv(gradient, "decoder.gdt_convs_attn_\(number).0"))
                let lateral = features[2 - stage]
                hidden = BiRefNetOps.resize(hidden, height: lateral.dim(1), width: lateral.dim(2))
                    + weights.conv(lateral, "decoder.lateral_block\(number).conv")
            }
            eval(hidden)
        }
        hidden = BiRefNetOps.resize(hidden, height: input.dim(1), width: input.dim(2))
        let injection = simpleConvs(input, "decoder.ipt_blk1")
        let result = sigmoid(weights.conv(concatenated([hidden, injection], axis: -1), "decoder.conv_out1.0"))
        eval(result)
        return result
    }

    private func simpleConvs(_ x: MLXArray, _ name: String) -> MLXArray {
        weights.conv(weights.conv(x, name + ".conv1", padding: 1), name + ".conv_out", padding: 1)
    }
    private func deformBranch(_ x: MLXArray, _ name: String, kernel: Int) throws -> MLXArray {
        let prefix = name + ".atrous_conv"
        let offsets = weights.conv(x, prefix + ".offset_conv", padding: kernel / 2)
        let mask = 2 * sigmoid(weights.conv(x, prefix + ".modulator_conv", padding: kernel / 2))
        let convolved = try BiRefNetOps.deformConv(x, offsets: offsets, mask: mask,
            weight: weights[prefix + ".regular_conv.weight"], padding: kernel / 2)
        return relu(weights.batchNorm(convolved, name + ".bn"))
    }
    private func decoderBlock(_ input: MLXArray, _ name: String) throws -> MLXArray {
        let x = relu(weights.batchNorm(weights.conv(input, name + ".conv_in", padding: 1), name + ".bn_in"))
        let prefix = name + ".dec_att"
        var branches = [try deformBranch(x, prefix + ".aspp1", kernel: 1)]
        for (index, kernel) in [1, 3, 7].enumerated() {
            branches.append(try deformBranch(x, prefix + ".aspp_deforms.\(index)", kernel: kernel))
        }
        let average = mean(x.asType(.float32), axes: [1, 2], keepDims: true).asType(x.dtype)
        let pooled = relu(weights.batchNorm(weights.conv(average, prefix + ".global_avg_pool.1"), prefix + ".global_avg_pool.2"))
        branches.append(BiRefNetOps.resize(pooled, height: x.dim(1), width: x.dim(2)))
        let fused = relu(weights.batchNorm(weights.conv(concatenated(branches, axis: -1), prefix + ".conv1"), prefix + ".bn1"))
        return weights.batchNorm(weights.conv(fused, name + ".conv_out", padding: 1), name + ".bn_out")
    }
}
