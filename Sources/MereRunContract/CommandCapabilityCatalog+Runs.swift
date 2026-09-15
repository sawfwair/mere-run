import Foundation

extension MereRunCapabilityCatalog {
    public static let runList = MereRunCommandCapability(
        id: "run.list",
        command: ["run", "list"],
        title: "Browse durable runs",
        summary: "Find local run directories and reports or list remote Relay jobs.",
        options: [
            .init(flag: "--root", label: "Local root", kind: .directory),
            .init(flag: "--executor", label: "Remote executor", kind: .string),
            .init(flag: "--limit", label: "Remote limit", kind: .integer),
            .init(flag: "--max-depth", label: "Scan depth", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let runInspect = MereRunCommandCapability(
        id: "run.inspect",
        command: ["run", "inspect"],
        title: "Inspect durable run",
        summary: "Inspect a run directory, report, plan, or remote job reference.",
        arguments: [
            .init(name: "path", label: "Run path or reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let runWatch = MereRunCommandCapability(
        id: "run.watch",
        command: ["run", "watch"],
        title: "Watch remote run",
        summary: "Poll a remote graph job and stream worker events until completion.",
        arguments: [
            .init(name: "reference", label: "Remote run reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--poll-interval", label: "Poll interval", kind: .number),
            .init(flag: "--json-stream", label: "NDJSON events", kind: .boolean),
            .init(flag: "--json", label: "Final JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let runFetch = MereRunCommandCapability(
        id: "run.fetch",
        command: ["run", "fetch"],
        title: "Fetch remote run",
        summary: "Verify and materialize a remote run and selected artifacts locally.",
        arguments: [
            .init(name: "reference", label: "Remote run reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--into", label: "Destination", kind: .directory, required: true),
            .init(flag: "--all-artifacts", label: "All artifacts", kind: .boolean),
            .init(flag: "--artifact", label: "Named artifact", kind: .string, repeatable: true),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--into")
    )

    public static let runCancel = MereRunCommandCapability(
        id: "run.cancel",
        command: ["run", "cancel"],
        title: "Cancel run",
        summary: "Request cooperative cancellation for a local or remote graph run.",
        arguments: [
            .init(name: "reference", label: "Run reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let runRetry = MereRunCommandCapability(
        id: "run.retry",
        command: ["run", "retry"],
        title: "Retry run",
        summary: "Retry a recorded image or transcription run, or an immutable Relay graph job.",
        arguments: [
            .init(name: "reference", label: "Run reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )
}
