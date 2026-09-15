import Foundation

extension MereRunCapabilityCatalog {
    public static let geoFlood = MereRunCommandCapability(
        id: "geo.flood",
        command: ["geo", "flood"],
        title: "Flood inference",
        summary: "Run native TerraMind Flood tile inference with MLX on Apple Silicon.",
        arguments: [
            .init(name: "input", label: "Tile safetensors", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Logits output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let geoFire = MereRunCommandCapability(
        id: "geo.fire",
        command: ["geo", "fire"],
        title: "Fire inference",
        summary: "Run native TerraMind Fire tile inference with MLX on Apple Silicon.",
        arguments: [
            .init(name: "input", label: "Tile safetensors", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Logits output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let geoTessera = MereRunCommandCapability(
        id: "geo.tessera",
        command: ["geo", "tessera"],
        title: "TESSERA embeddings",
        summary: "Encode local Sentinel-1/2 time series with a native TESSERA v2 student.",
        arguments: [
            .init(name: "input", label: "Observation safetensors", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Embedding output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--dimensions", label: "Dimensions", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let geoOlmoEarth = MereRunCommandCapability(
        id: "geo.olmoearth",
        command: ["geo", "olmoearth"],
        title: "OlmoEarth embeddings",
        summary: "Encode multisensor Earth observations with native OlmoEarth v1.2.",
        arguments: [
            .init(name: "input", label: "Observation safetensors", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Embedding output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--patch-size", label: "Patch size", kind: .integer),
            .init(flag: "--input-resolution", label: "Input resolution", kind: .number),
            .init(flag: "--include-tokens", label: "Include tokens", kind: .boolean),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    // MARK: - Model store locations
}
