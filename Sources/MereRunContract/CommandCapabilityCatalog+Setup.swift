import Foundation

extension MereRunCapabilityCatalog {
    public static let status = MereRunCommandCapability(
        id: "status",
        command: ["status"],
        title: "Status snapshot",
        summary: "Inspect the API server, loaded models, model store, and local inventory.",
        options: [
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--timeout-seconds", label: "Timeout", kind: .number),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let gate = MereRunCommandCapability(
        id: "gate",
        command: ["gate"],
        title: "Quality gate",
        summary: "Run installed-model correctness, determinism, and performance checks.",
        options: [
            .init(
                flag: "--require-all", label: "Require selected models", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(
                flag: "--all-installed", label: "Check all installed models", kind: .boolean, group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--skip-model", label: "Quarantined models", kind: .string, group: Group.run, tier: .expert
            ),
            .init(flag: "--suite", label: "Suites", kind: .string),
            .init(flag: "--update-baselines", label: "Update baselines", kind: .boolean),
            .init(flag: "--strict-perf", label: "Strict performance", kind: .boolean),
            .init(flag: "--json-output", label: "JSON report", kind: .file),
            .init(flag: "--list", label: "List checks", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--json-output", optional: true)
    )

    public static let setup = MereRunCommandCapability(
        id: "setup",
        command: ["setup"],
        title: "Setup path",
        summary: "Plan or run guided, BYOA, or manual local setup.",
        options: [
            .init(flag: "--mode", label: "Mode", kind: .choice, choices: ["agent", "byoa", "manual"]),
            .init(flag: "--agent-model", label: "Agent tier", kind: .choice, choices: ["small", "tier", "premier"]),
            .init(flag: "--install", label: "Install", kind: .boolean),
            .init(flag: "--start", label: "Start", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--host", label: "API host", kind: .string),
            .init(flag: "--port", label: "API port", kind: .integer),
            .init(flag: "--pi-path", label: "Pi executable", kind: .file),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )
}
