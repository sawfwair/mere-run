import Foundation

public enum Krea2LoRAKeyMapper {
    public static func map(baseKey: String) -> String? {
        let normalized = normalizeBaseKey(baseKey)

        if let global = globalMappings[normalized] {
            return global
        }

        if let block = mapBlockKey(normalized) {
            return block
        }

        if let textFusion = mapTextFusionKey(normalized) {
            return textFusion
        }

        return nil
    }

    private static let globalMappings: [String: String] = [
        "first": "img_in",
        "txtmlp.1": "txt_in.linear_1",
        "txtmlp.3": "txt_in.linear_2",
        "txtfusion.projector": "text_fusion.projector",
        "tmlp.0": "time_embed.linear_1",
        "tmlp.2": "time_embed.linear_2",
        "tproj.1": "time_mod_proj",
        "last.linear": "final_layer.linear",
    ]

    private static func normalizeBaseKey(_ key: String) -> String {
        var normalized = key
        let prefixesToRemove = [
            "base_model.model.",
            "transformer.",
            "model.",
        ]

        for prefix in prefixesToRemove where normalized.hasPrefix(prefix) {
            normalized = String(normalized.dropFirst(prefix.count))
            break
        }

        if normalized.hasSuffix(".weight") {
            normalized = String(normalized.dropLast(".weight".count))
        }

        return normalized
    }

    private static func mapBlockKey(_ key: String) -> String? {
        let components = key.split(separator: ".").map(String.init)
        guard components.count >= 4,
              components[0] == "blocks",
              Int(components[1]) != nil else {
            return nil
        }

        guard let suffix = mapKreaProjectionSuffix(components.dropFirst(2).joined(separator: ".")) else {
            return nil
        }
        return "transformer_blocks.\(components[1]).\(suffix)"
    }

    private static func mapTextFusionKey(_ key: String) -> String? {
        let components = key.split(separator: ".").map(String.init)
        guard components.count >= 2,
              components[0] == "txtfusion" else {
            return nil
        }

        if components.count == 2, components[1] == "projector" {
            return "text_fusion.projector"
        }

        guard components.count >= 5,
              (components[1] == "layerwise_blocks" || components[1] == "refiner_blocks"),
              Int(components[2]) != nil else {
            return nil
        }

        guard let suffix = mapKreaProjectionSuffix(components.dropFirst(3).joined(separator: ".")) else {
            return nil
        }
        return "text_fusion.\(components[1]).\(components[2]).\(suffix)"
    }

    private static func mapKreaProjectionSuffix(_ suffix: String) -> String? {
        switch suffix {
        case "attn.wq":
            return "attn.to_q"
        case "attn.wk":
            return "attn.to_k"
        case "attn.wv":
            return "attn.to_v"
        case "attn.gate":
            return "attn.to_gate"
        case "attn.wo":
            return "attn.to_out.0"
        case "mlp.gate":
            return "ff.gate"
        case "mlp.up":
            return "ff.up"
        case "mlp.down":
            return "ff.down"
        default:
            return nil
        }
    }
}
