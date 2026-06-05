import Foundation
import MLX

public enum Ideogram4VAEWeights {
    private static let ignoredPrefix = "__ideogram4_ignored."
    private static let resolutionCount = 4

    public static func mapKey(_ key: String) -> String {
        if key.hasPrefix("bn.") {
            return ignoredPrefix + key
        }
        if key.hasPrefix("encoder.quant_conv.") {
            return key.replacingOccurrences(of: "encoder.quant_conv.", with: "quant_conv.")
        }
        if key.hasPrefix("decoder.post_quant_conv.") {
            return key.replacingOccurrences(of: "decoder.post_quant_conv.", with: "post_quant_conv.")
        }
        if key.hasPrefix("encoder.norm_out.") {
            return key.replacingOccurrences(of: "encoder.norm_out.", with: "encoder.conv_norm_out.")
        }
        if key.hasPrefix("decoder.norm_out.") {
            return key.replacingOccurrences(of: "decoder.norm_out.", with: "decoder.conv_norm_out.")
        }

        if let mapped = mapMidBlock(key) { return mapped }
        if let mapped = mapEncoderDownBlock(key) { return mapped }
        if let mapped = mapDecoderUpBlock(key) { return mapped }
        if key.hasPrefix("encoder.conv_in.")
            || key.hasPrefix("encoder.conv_out.")
            || key.hasPrefix("decoder.conv_in.")
            || key.hasPrefix("decoder.conv_out.") {
            return key
        }
        return ignoredPrefix + key
    }

    public static func mapParameter(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        if key.hasPrefix(ignoredPrefix) {
            return []
        }
        return [(key, value)]
    }

    private static func mapMidBlock(_ key: String) -> String? {
        if let match = firstMatch(key, #"^(encoder|decoder)\.mid\.block_(\d+)\.(.+)$"#) {
            let side = match[0]
            guard let block = Int(match[1]) else { return nil }
            let rest = match[2].replacingOccurrences(of: "nin_shortcut", with: "conv_shortcut")
            return "\(side).mid_block.resnets.\(block - 1).\(rest)"
        }
        if let match = firstMatch(key, #"^(encoder|decoder)\.mid\.attn_1\.(.+)$"#) {
            let side = match[0]
            var rest = match[1]
            rest = rest.replacingOccurrences(of: "norm.", with: "group_norm.")
            rest = rest.replacingOccurrences(of: "q.", with: "to_q.")
            rest = rest.replacingOccurrences(of: "k.", with: "to_k.")
            rest = rest.replacingOccurrences(of: "v.", with: "to_v.")
            rest = rest.replacingOccurrences(of: "proj_out.", with: "to_out.0.")
            return "\(side).mid_block.attentions.0.\(rest)"
        }
        return nil
    }

    private static func mapEncoderDownBlock(_ key: String) -> String? {
        if let match = firstMatch(key, #"^encoder\.down\.(\d+)\.block\.(\d+)\.(.+)$"#) {
            let rest = match[2].replacingOccurrences(of: "nin_shortcut", with: "conv_shortcut")
            return "encoder.down_blocks.\(match[0]).resnets.\(match[1]).\(rest)"
        }
        if let match = firstMatch(key, #"^encoder\.down\.(\d+)\.downsample\.conv\.(.+)$"#) {
            return "encoder.down_blocks.\(match[0]).downsamplers.0.conv.\(match[1])"
        }
        return nil
    }

    private static func mapDecoderUpBlock(_ key: String) -> String? {
        if let match = firstMatch(key, #"^decoder\.up\.(\d+)\.block\.(\d+)\.(.+)$"#) {
            guard let officialIndex = Int(match[0]) else { return nil }
            let localIndex = resolutionCount - 1 - officialIndex
            let rest = match[2].replacingOccurrences(of: "nin_shortcut", with: "conv_shortcut")
            return "decoder.up_blocks.\(localIndex).resnets.\(match[1]).\(rest)"
        }
        if let match = firstMatch(key, #"^decoder\.up\.(\d+)\.upsample\.conv\.(.+)$"#) {
            guard let officialIndex = Int(match[0]) else { return nil }
            let localIndex = resolutionCount - 1 - officialIndex
            return "decoder.up_blocks.\(localIndex).upsamplers.0.conv.\(match[1])"
        }
        return nil
    }

    private static func firstMatch(_ text: String, _ pattern: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range) else { return nil }
        var captures: [String] = []
        for index in 1..<match.numberOfRanges {
            let captureRange = match.range(at: index)
            guard let range = Range(captureRange, in: text) else { return nil }
            captures.append(String(text[range]))
        }
        return captures
    }
}
