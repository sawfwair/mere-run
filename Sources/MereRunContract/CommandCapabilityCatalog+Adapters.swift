import Foundation

extension MereRunCapabilityCatalog {
    public static let adapterList = MereRunCommandCapability(
        id: "adapter.list",
        command: ["adapter", "list"],
        title: "Browse adapters",
        summary: "List verified LoRA adapters, compatibility, provenance, and install state.",
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let adapterPull = MereRunCommandCapability(
        id: "adapter.pull",
        command: ["adapter", "pull"],
        title: "Pull adapter",
        summary: "Download and checksum-verify one cataloged LoRA adapter.",
        arguments: [
            .init(name: "target", label: "Adapter id", kind: .string, required: true)
        ],
        options: [
            .init(
                flag: "--accept-license", label: "Accept adapter terms", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(flag: "--force", label: "Replace install", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors")
    )
}
