import Foundation

extension MereRunCapabilityCatalog {
    public static let evaluationPackValidate = MereRunCommandCapability(
        id: "eval.pack.validate",
        command: ["eval", "pack", "validate"],
        title: "Validate evaluation pack",
        summary: "Validate and content-hash a declared-file-only external evaluation pack.",
        arguments: [
            .init(name: "pack", label: "Evaluation pack", kind: .directory, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let evaluationRun = MereRunCommandCapability(
        id: "eval.run",
        command: ["eval", "run"],
        title: "Run external evaluation",
        summary: "Run matched model, prompt, and adapter arms with resumable reports and gates.",
        arguments: [
            .init(name: "pack", label: "Evaluation pack", kind: .directory, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model slot binding", kind: .string, repeatable: true),
            .init(flag: "--adapter", label: "Adapter slot binding", kind: .string, repeatable: true),
            .init(flag: "--trials", label: "Trials", kind: .integer),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context size", kind: .integer),
            .init(
                flag: "--logprobs",
                label: "Logprob capture",
                kind: .choice,
                choices: ["none", "summary", "tokens", "top"]
            ),
            .init(flag: "--top-logprobs", label: "Top logprobs", kind: .integer),
            .init(flag: "--allow-external-scorer", label: "Authorize scorer", kind: .boolean),
            .init(flag: "--log-responses", label: "Include responses", kind: .boolean),
            .init(flag: "--dry-run", label: "Plan only", kind: .boolean),
            .init(flag: "--checkpoint", label: "Checkpoint", kind: .file),
            .init(flag: "--resume", label: "Resume", kind: .boolean),
            .init(flag: "--case-trial-limit", label: "Row limit", kind: .integer),
            .init(flag: "--output", label: "Report", kind: .file),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
    )

    public static let evaluationPromote = MereRunCommandCapability(
        id: "eval.promote",
        command: ["eval", "promote"],
        title: "Promote evaluated artifact",
        summary: "Create a content-addressed receipt for a complete, gate-passing evaluation report.",
        arguments: [
            .init(name: "report", label: "Evaluation report", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Promotion receipt", kind: .file),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
    )
}
