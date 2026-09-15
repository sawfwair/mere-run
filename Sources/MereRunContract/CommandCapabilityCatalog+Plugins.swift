import Foundation

extension MereRunCapabilityCatalog {
    public static let pluginList = MereRunCommandCapability(
        id: "plugin.list",
        command: ["plugin", "list"],
        title: "List plugins",
        summary: "Inspect the official or an overridden plugin catalog.",
        options: [
            .init(flag: "--catalog-url", label: "Catalog", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let pluginInstall = MereRunCommandCapability(
        id: "plugin.install",
        command: ["plugin", "install"],
        title: "Install plugin",
        summary: "Plan or execute an official plugin installation.",
        arguments: [.init(name: "id", label: "Plugin id", kind: .string, required: true)],
        options: [
            .init(
                flag: "--source", label: "Install from source", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(
                flag: "--bundle-manifest", label: "Signed bundle manifest", kind: .string, group: Group.inputs,
                tier: .expert
            ),
            .init(
                flag: "--bundle-archive", label: "Signed bundle archive", kind: .file, group: Group.inputs, tier: .expert
            ),
            .init(flag: "--catalog-url", label: "Catalog", kind: .string),
            .init(flag: "--channel", label: "Channel", kind: .string),
            .init(flag: "--yes", label: "Execute", kind: .boolean),
            .init(flag: "--force", label: "Force", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let pluginDoctor = MereRunCommandCapability(
        id: "plugin.doctor",
        command: ["plugin", "doctor"],
        title: "Plugin doctor",
        summary: "Run a companion plugin's health check.",
        arguments: [.init(name: "id", label: "Plugin id", kind: .string, required: true)],
        options: [
            .init(flag: "--catalog-url", label: "Catalog", kind: .string)
        ],
        output: .init(kind: .text)
    )

    public static let pluginInfo = MereRunCommandCapability(
        id: "plugin.info",
        command: ["plugin", "info"],
        title: "Plugin details",
        summary: "Show one plugin's catalog entry and install command.",
        arguments: [.init(name: "id", label: "Plugin id", kind: .string, required: true)],
        options: [
            .init(flag: "--catalog-url", label: "Catalog", kind: .string),
            .init(flag: "--channel", label: "Channel", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let pluginRun = MereRunCommandCapability(
        id: "plugin.run",
        command: ["plugin", "run"],
        title: "Run plugin",
        summary: "Run an installed plugin without changing PATH.",
        arguments: [
            .init(name: "entrypoint", label: "Entrypoint", kind: .string, required: true),
            .init(name: "arguments", label: "Plugin arguments", kind: .string, required: false, repeatable: true)
        ],
        options: [],
        output: .init(kind: .text)
    )

    public static let pluginRollback = MereRunCommandCapability(
        id: "plugin.rollback",
        command: ["plugin", "rollback"],
        title: "Roll back plugin",
        summary: "Restore a retained signed plugin bundle.",
        arguments: [.init(name: "id", label: "Plugin id", kind: .string, required: true)],
        options: [
            .init(flag: "--yes", label: "Activate", kind: .boolean)
        ],
        output: .init(kind: .text)
    )
}
