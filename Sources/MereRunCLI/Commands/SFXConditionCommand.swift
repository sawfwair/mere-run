import ArgumentParser
import Foundation
import MereRunCore
import MLX

struct SFXCondition: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "condition",
        abstract: "Export Woosh conditioning tensors.",
        subcommands: [
            SFXConditionText.self,
        ]
    )
}

struct SFXConditionText: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "text",
        abstract: "Encode a prompt with the Woosh text conditioner.",
        discussion: """
        Writes a safetensors file containing `embeddings` and `mask` arrays.
        Prints the output path to stdout.
        """
    )

    @Argument(help: "Prompt to encode.")
    var prompt: String

    @Option(name: [.customShort("o"), .long], help: "Output safetensors path (default: ./mererun-sfx-condition-<timestamp>.safetensors).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Woosh checkpoints root.")
    var model: String = ModelResolver.ModelID.wooshDFlow.rawValue

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let outputURL = CLIOutput.resolveOutputURL(
            output,
            defaultPrefix: "mererun-sfx-condition",
            defaultExtension: "safetensors"
        )
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let resources = try await SFXWooshRuntime.resolve(model: model, quiet: quiet)
        if !quiet {
            CLIStderr.write("Loading Woosh text conditioner from \(resources.textConditionerURL.path)\n")
        }
        let conditioner = try WooshTextConditioner.load(resources: resources)
        let conditioning = conditioner.encode(prompts: [prompt])
        MLX.eval(conditioning.embeddings, conditioning.attentionMask)
        try MLX.save(
            arrays: [
                "embeddings": conditioning.embeddings,
                "mask": conditioning.attentionMask,
            ],
            url: outputURL
        )
        if !quiet {
            CLIStderr.write("Saved conditioning tensors: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }
}
