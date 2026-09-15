import Foundation

extension MereRunCapabilityCatalog {
    public static let guide = MereRunCommandCapability(
        id: "guide",
        command: ["guide"],
        title: "Offline guides",
        summary: "List or read CLI-owned offline workflow guides.",
        arguments: [
            .init(name: "command-path", label: "Command path", kind: .string, required: false, repeatable: true)
        ],
        options: [
            .init(flag: "--list", label: "List topics", kind: .boolean),
            .init(flag: "--list-models", label: "List model guides", kind: .boolean),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--markdown", label: "Markdown index", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let configSet = MereRunCommandCapability(
        id: "config.set",
        command: ["config", "set"],
        title: "Set configuration",
        summary: "Persist a supported configuration value, including a secret-safe environment source.",
        arguments: [
            .init(name: "key", label: "Key", kind: .string, required: true),
            .init(name: "value", label: "Value", kind: .string, required: false)
        ],
        options: [
            .init(flag: "--from-env", label: "Environment variable", kind: .string)
        ],
        output: .init(kind: .text)
    )

    public static let configGet = MereRunCommandCapability(
        id: "config.get",
        command: ["config", "get"],
        title: "Read configuration",
        summary: "Read a persisted configuration value with secrets masked by default.",
        arguments: [
            .init(name: "key", label: "Key", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--reveal", label: "Reveal secret", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let configUnset = MereRunCommandCapability(
        id: "config.unset",
        command: ["config", "unset"],
        title: "Unset configuration",
        summary: "Remove a persisted configuration value.",
        arguments: [
            .init(name: "key", label: "Key", kind: .string, required: true)
        ],
        options: [],
        output: .init(kind: .text)
    )

    // MARK: - Geospatial

    public static let configList = MereRunCommandCapability(
        id: "config.list",
        command: ["config", "list"],
        title: "List configuration",
        summary: "Show all persisted configuration values with secrets masked.",
        options: [],
        output: .init(kind: .text)
    )

    public static let configPath = MereRunCommandCapability(
        id: "config.path",
        command: ["config", "path"],
        title: "Configuration path",
        summary: "Print the path of the persisted configuration file.",
        options: [],
        output: .init(kind: .text)
    )
}
