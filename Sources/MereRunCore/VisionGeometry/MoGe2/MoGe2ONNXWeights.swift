import Foundation
@preconcurrency import MLX
import MLXNN

public enum MoGe2WeightError: Error, Equatable, LocalizedError, Sendable {
    case missingInitializer(String)

    public var errorDescription: String? {
        switch self {
        case .missingInitializer(let name): "MoGe-2 ONNX initializer is missing: \(name)"
        }
    }
}

public enum MoGe2ONNXWeights {
    public static func load(model: MoGe2Model, archive: ONNXInitializerArchive) throws {
        let source = try ONNXMLXInitializerLoader.load(archive: archive)
        var target: [String: MLXArray] = [:]

        func required(_ name: String) throws -> MLXArray {
            guard let value = source[name] else { throw MoGe2WeightError.missingInitializer(name) }
            return value
        }
        func copy(_ sourceName: String, _ targetName: String) throws {
            target[targetName] = try required(sourceName)
        }
        func linear(_ sourceName: String, _ targetName: String) throws {
            target[targetName] = try required(sourceName).T
        }
        func convolution(_ sourceName: String, _ targetName: String) throws {
            target[targetName] = try required(sourceName).transposed(0, 2, 3, 1)
        }
        func transposedConvolution(_ sourceName: String, _ targetName: String) throws {
            target[targetName] = try required(sourceName).transposed(1, 2, 3, 0)
        }

        let cls = try required("encoder.backbone.cls_token")
        target["encoder.backbone.cls_token"] = cls
        target["encoder.backbone.class_position"] = cls
        target["encoder.backbone.patch_position"] = try required("onnx::Reshape_3473")
        try convolution("encoder.backbone.patch_embed.proj.weight", "encoder.backbone.patch_embed.proj.weight")
        try copy("encoder.backbone.patch_embed.proj.bias", "encoder.backbone.patch_embed.proj.bias")
        for block in 0..<12 {
            let sourcePrefix = "encoder.backbone.blocks.\(block)"
            let targetPrefix = sourcePrefix
            let anonymous = 3_480 + block * 6
            try copy("\(sourcePrefix).norm1.weight", "\(targetPrefix).norm1.weight")
            try copy("\(sourcePrefix).norm1.bias", "\(targetPrefix).norm1.bias")
            try linear("onnx::MatMul_\(anonymous)", "\(targetPrefix).attn.qkv.weight")
            try copy("\(sourcePrefix).attn.qkv.bias", "\(targetPrefix).attn.qkv.bias")
            try linear("onnx::MatMul_\(anonymous + 3)", "\(targetPrefix).attn.proj.weight")
            try copy("\(sourcePrefix).attn.proj.bias", "\(targetPrefix).attn.proj.bias")
            try copy("\(sourcePrefix).ls1.gamma", "\(targetPrefix).ls1.gamma")
            try copy("\(sourcePrefix).norm2.weight", "\(targetPrefix).norm2.weight")
            try copy("\(sourcePrefix).norm2.bias", "\(targetPrefix).norm2.bias")
            try linear("onnx::MatMul_\(anonymous + 4)", "\(targetPrefix).mlp.fc1.weight")
            try copy("\(sourcePrefix).mlp.fc1.bias", "\(targetPrefix).mlp.fc1.bias")
            try linear("onnx::MatMul_\(anonymous + 5)", "\(targetPrefix).mlp.fc2.weight")
            try copy("\(sourcePrefix).mlp.fc2.bias", "\(targetPrefix).mlp.fc2.bias")
            try copy("\(sourcePrefix).ls2.gamma", "\(targetPrefix).ls2.gamma")
        }
        try copy("encoder.backbone.norm.weight", "encoder.backbone.norm.weight")
        try copy("encoder.backbone.norm.bias", "encoder.backbone.norm.bias")
        for projection in 0..<2 {
            try convolution(
                "encoder.output_projections.\(projection).weight",
                "encoder.output_projections.\(projection).weight"
            )
            try copy(
                "encoder.output_projections.\(projection).bias",
                "encoder.output_projections.\(projection).bias"
            )
        }

        func mapStack(sourcePrefix: String, targetPrefix: String, hasOutput: Bool) throws {
            for level in 0..<5 {
                try convolution(
                    "\(sourcePrefix).input_blocks.\(level).weight",
                    "\(targetPrefix).input_blocks.\(level).weight"
                )
                try copy(
                    "\(sourcePrefix).input_blocks.\(level).bias",
                    "\(targetPrefix).input_blocks.\(level).bias"
                )
            }
            for level in 0..<3 {
                try transposedConvolution(
                    "\(sourcePrefix).resamplers.\(level).0.weight",
                    "\(targetPrefix).resamplers.\(level).transpose.weight"
                )
                try copy(
                    "\(sourcePrefix).resamplers.\(level).0.bias",
                    "\(targetPrefix).resamplers.\(level).transpose.bias"
                )
                try convolution(
                    "\(sourcePrefix).resamplers.\(level).1.weight",
                    "\(targetPrefix).resamplers.\(level).conv.weight"
                )
                try copy(
                    "\(sourcePrefix).resamplers.\(level).1.bias",
                    "\(targetPrefix).resamplers.\(level).conv.bias"
                )
            }
            try convolution("\(sourcePrefix).resamplers.3.1.weight", "\(targetPrefix).resamplers.3.conv.weight")
            try copy("\(sourcePrefix).resamplers.3.1.bias", "\(targetPrefix).resamplers.3.conv.bias")
            for level in 1...3 {
                try convolution(
                    "\(sourcePrefix).res_blocks.\(level).0.layers.2.weight",
                    "\(targetPrefix).res_blocks.\(level).blocks.0.first.weight"
                )
                try copy(
                    "\(sourcePrefix).res_blocks.\(level).0.layers.2.bias",
                    "\(targetPrefix).res_blocks.\(level).blocks.0.first.bias"
                )
                try convolution(
                    "\(sourcePrefix).res_blocks.\(level).0.layers.5.weight",
                    "\(targetPrefix).res_blocks.\(level).blocks.0.second.weight"
                )
                try copy(
                    "\(sourcePrefix).res_blocks.\(level).0.layers.5.bias",
                    "\(targetPrefix).res_blocks.\(level).blocks.0.second.bias"
                )
            }
            if hasOutput {
                try convolution("\(sourcePrefix).output_blocks.4.weight", "\(targetPrefix).output.weight")
                try copy("\(sourcePrefix).output_blocks.4.bias", "\(targetPrefix).output.bias")
            }
        }

        try mapStack(sourcePrefix: "neck", targetPrefix: "neck", hasOutput: false)
        try mapStack(sourcePrefix: "points_head", targetPrefix: "points_head", hasOutput: true)
        try mapStack(sourcePrefix: "normal_head", targetPrefix: "normal_head", hasOutput: true)
        try mapStack(sourcePrefix: "mask_head", targetPrefix: "mask_head", hasOutput: true)
        try copy("scale_head.0.weight", "scale_head.first.weight")
        try copy("scale_head.0.bias", "scale_head.first.bias")
        try copy("scale_head.2.weight", "scale_head.second.weight")
        try copy("scale_head.2.bias", "scale_head.second.bias")
        try copy("scale_head.4.weight", "scale_head.output.weight")
        try copy("scale_head.4.bias", "scale_head.output.bias")

        try model.update(parameters: ModuleParameters.unflattened(target), verify: .all)
        MLX.eval(model)
    }
}
