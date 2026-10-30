import MLX

extension BiRefNetWeights {
    static func shapes(_ config: BiRefNetConfiguration) -> [String: [Int]] {
        var result: [String: [Int]] = [:]
        func norm(_ name: String, _ channels: Int) {
            result[name + ".weight"] = [channels]; result[name + ".bias"] = [channels]
        }
        func bn(_ name: String, _ channels: Int) {
            norm(name, channels)
            result[name + ".running_mean"] = [channels]; result[name + ".running_var"] = [channels]
            result[name + ".num_batches_tracked"] = []
        }
        func linear(_ name: String, _ input: Int, _ output: Int, bias: Bool = true) {
            result[name + ".weight"] = [output, input]
            if bias { result[name + ".bias"] = [output] }
        }
        func conv(_ name: String, _ input: Int, _ output: Int, _ kernel: Int, bias: Bool = true) {
            result[name + ".weight"] = [output, input, kernel, kernel]
            if bias { result[name + ".bias"] = [output] }
        }
        func deform(_ name: String, _ input: Int, _ output: Int, _ kernel: Int) {
            conv(name + ".atrous_conv.offset_conv", input, 2 * kernel * kernel, kernel)
            conv(name + ".atrous_conv.modulator_conv", input, kernel * kernel, kernel)
            conv(name + ".atrous_conv.regular_conv", input, output, kernel, bias: false)
            bn(name + ".bn", output)
        }
        func block(_ name: String, _ input: Int, _ output: Int) {
            let intermediate = config.decoderIntermediate, aspp = config.asppChannels
            conv(name + ".conv_in", input, intermediate, 3); bn(name + ".bn_in", intermediate)
            deform(name + ".dec_att.aspp1", intermediate, aspp, 1)
            for (index, kernel) in [1, 3, 7].enumerated() {
                deform(name + ".dec_att.aspp_deforms.\(index)", intermediate, aspp, kernel)
            }
            conv(name + ".dec_att.global_avg_pool.1", intermediate, aspp, 1, bias: false)
            bn(name + ".dec_att.global_avg_pool.2", aspp)
            conv(name + ".dec_att.conv1", 5 * aspp, intermediate, 1, bias: false)
            bn(name + ".dec_att.bn1", intermediate)
            conv(name + ".conv_out", intermediate, output, 3); bn(name + ".bn_out", output)
        }
        let dim = config.embedDim, window = config.windowSize
        conv("bb.patch_embed.proj", 3, dim, 4); norm("bb.patch_embed.norm", dim)
        for stage in 0..<4 {
            let channels = dim << stage, prefix = "bb.layers.\(stage)"
            norm("bb.norm\(stage)", channels)
            for depth in 0..<config.depths[stage] {
                let name = prefix + ".blocks.\(depth)"
                norm(name + ".norm1", channels); norm(name + ".norm2", channels)
                linear(name + ".attn.qkv", channels, channels * 3)
                linear(name + ".attn.proj", channels, channels)
                result[name + ".attn.relative_position_bias_table"] = [(2 * window - 1) * (2 * window - 1), config.heads[stage]]
                result[name + ".attn.relative_position_index"] = [window * window, window * window]
                linear(name + ".mlp.fc1", channels, channels * 4)
                linear(name + ".mlp.fc2", channels * 4, channels)
            }
            if stage < 3 {
                norm(prefix + ".downsample.norm", 4 * channels)
                linear(prefix + ".downsample.reduction", 4 * channels, 2 * channels, bias: false)
            }
        }
        let channels = config.channels
        block("squeeze_module.0", channels.reduce(0, +), channels[0])
        for (index, input, output) in [
            (5, 3072, channels[0] / 8), (4, 768, channels[0] / 8), (3, 192, channels[1] / 8),
            (2, 48, channels[2] / 8), (1, 3, channels[3] / 8)
        ] {
            conv("decoder.ipt_blk\(index).conv1", input, config.decoderIntermediate, 3)
            conv("decoder.ipt_blk\(index).conv_out", config.decoderIntermediate, output, 3)
        }
        for index in 0..<4 {
            let number = 4 - index
            let extra = index == 0 ? channels[0] / 8 : channels[index - 1] / 8
            let output = index == 3 ? channels[3] / 2 : channels[index + 1]
            block("decoder.decoder_block\(number)", channels[index] + extra, output)
            if number > 1 {
                conv("decoder.lateral_block\(number).conv", output, output, 1)
                conv("decoder.gdt_convs_\(number).0", output, 16, 3)
                bn("decoder.gdt_convs_\(number).1", 16)
                conv("decoder.gdt_convs_attn_\(number).0", 16, 1, 1)
            }
        }
        conv("decoder.conv_out1.0", channels[3] / 2 + channels[3] / 8, 1, 1)
        return result
    }
}
