import Foundation

extension MereRunCapabilityCatalog {
    public static let worldServe = MereRunCommandCapability(
        id: "world.serve",
        command: ["world", "serve"],
        title: "World session",
        summary: "Serve one warm DreamX or Cosmos3 conditioned-video world session.",
        options: [
            .init(
                flag: "--disable-scene-memory", label: "Disable scene memory", kind: .boolean, group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--scene-memory-strength", label: "Scene memory strength", kind: .number, defaultValue: "0.08",
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--scene-memory-max-frames", label: "Scene memory frame limit", kind: .integer, defaultValue: "96",
                group: Group.run, tier: .expert
            ),
            .init(
                flag: "--scene-memory-minimum-gap", label: "Scene memory minimum gap", kind: .integer, defaultValue: "3",
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--scene-memory-max-yaw", label: "Scene memory maximum yaw", kind: .number, defaultValue: "2.0",
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--scene-memory-max-translation", label: "Scene memory maximum translation", kind: .number,
                defaultValue: "0.1", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--scene-memory-exact-yaw", label: "Scene restore yaw tolerance", kind: .number, defaultValue: "0.01",
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--scene-memory-exact-translation", label: "Scene restore translation tolerance", kind: .number,
                defaultValue: "0.001", group: Group.sampling, tier: .expert
            ),
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--backend", label: "Backend", kind: .choice, choices: ["dreamx", "cosmos3"]),
            .init(flag: "--base-model", label: "Base model", kind: .string),
            .init(flag: "--model", label: "World model", kind: .string),
            .init(flag: "--state-directory", label: "State directory", kind: .directory),
            .init(flag: "--prepare", label: "Warm models", kind: .boolean)
        ],
        output: .init(kind: .service)
    )
}
