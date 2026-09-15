import Foundation

extension MereRunCapabilityCatalog {
    public static let agentOnboard = MereRunCommandCapability(
        id: "agent.onboard",
        command: ["agent", "onboard"],
        title: "Agent onboarding",
        summary: "Check readiness and prepare the optional Pi integration.",
        options: [
            .init(flag: "--pull-recommended", label: "Pull recommended", kind: .boolean),
            .init(flag: "--accept-model-license", label: "Accept model terms", kind: .boolean),
            .init(flag: "--install-pi", label: "Install Pi", kind: .boolean),
            .init(flag: "--configure-pi", label: "Configure Pi", kind: .boolean),
            .init(flag: "--host", label: "API host", kind: .string),
            .init(flag: "--port", label: "API port", kind: .integer),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let agentStatus = MereRunCommandCapability(
        id: "agent.status",
        command: ["agent", "status"],
        title: "Agent status",
        summary: "Inspect Pi, provider, machine, and local agent-model readiness.",
        options: [
            .init(flag: "--pi-path", label: "Pi executable", kind: .file),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let agentInstallPi = MereRunCommandCapability(
        id: "agent.install-pi",
        command: ["agent", "install-pi"],
        title: "Install Pi",
        summary: "Install or replace the optional Pi setup agent.",
        options: [
            .init(flag: "--force", label: "Force reinstall", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let agentStart = MereRunCommandCapability(
        id: "agent.start",
        command: ["agent", "start"],
        title: "Start setup agent",
        summary: "Start a guided Pi session against the local API.",
        options: [
            .init(
                flag: "--pi-argument", label: "Pi argument", kind: .string, repeatable: true, group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--working-directory", label: "Working directory", kind: .directory, group: Group.run, tier: .expert
            ),
            .init(
                flag: "--inline", label: "Run in active terminal", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(flag: "--host", label: "API host", kind: .string),
            .init(flag: "--port", label: "API port", kind: .integer),
            .init(flag: "--pi-path", label: "Pi executable", kind: .file),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--skip-server", label: "Skip server", kind: .boolean),
            .init(flag: "--allow-unsupported", label: "Allow unsupported", kind: .boolean),
            .init(flag: "--no-bootstrap", label: "No bootstrap", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service)
    )
}
