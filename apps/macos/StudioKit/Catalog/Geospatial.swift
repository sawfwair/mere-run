import Foundation

// MARK: - Geospatial templates

extension CommandCatalog {
    package static let geospatialTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .geoFlood,
            category: .geospatial,
            title: "Flood inference",
            subtitle: "TerraMind flood logits from a normalized tile batch",
            systemImage: "water.waves",
            inputKind: .file([.data]),
            outputKind: .file("safetensors")
        ),
        CommandTemplate(
            id: .geoFire,
            category: .geospatial,
            title: "Fire inference",
            subtitle: "TerraMind fire logits from a normalized tile batch",
            systemImage: "flame",
            inputKind: .file([.data]),
            outputKind: .file("safetensors")
        ),
        CommandTemplate(
            id: .geoTessera,
            category: .geospatial,
            title: "TESSERA embeddings",
            subtitle: "Encode Sentinel-1/2 time series with TESSERA v2",
            systemImage: "square.stack.3d.down.right",
            inputKind: .file([.data]),
            outputKind: .file("safetensors")
        ),
        CommandTemplate(
            id: .geoOlmoEarth,
            category: .geospatial,
            title: "OlmoEarth embeddings",
            subtitle: "Encode multisensor Earth observations with OlmoEarth v1.2",
            systemImage: "globe.europe.africa",
            inputKind: .file([.data]),
            outputKind: .file("safetensors")
        )
    ]
}

// MARK: - Geospatial arguments

extension CommandArguments {
    package static func geoFlood(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.GeoFlood
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.preflight { args.flag(F.preflight) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func geoFire(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.GeoFire
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if draft.preflight { args.flag(F.preflight) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func geoTessera(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.GeoTessera
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        if !draft.geoDimensions.isBlank { args.option(F.dimensions, draft.geoDimensions) }
        if draft.preflight { args.flag(F.preflight) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }

    package static func geoOlmoEarth(_ draft: CommandDraft) -> [String] {
        typealias F = CommandFlags.GeoOlmoEarth
        var args = ArgumentBuilder(F.self)
        args.value(draft.inputPath)
        if !draft.outputPath.isBlank { args.option(F.output, draft.outputPath) }
        if !draft.model.isBlank { args.option(F.model, draft.model) }
        args.option(F.patchSize, String(draft.geoPatchSize))
        args.option(F.inputResolution, String(draft.geoInputResolution))
        if draft.geoIncludeTokens { args.flag(F.includeTokens) }
        if draft.preflight { args.flag(F.preflight) }
        if draft.json { args.flag(F.json) }
        return args.arguments
    }
}
