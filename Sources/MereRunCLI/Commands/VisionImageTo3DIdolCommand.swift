import ArgumentParser
import Foundation
import MereRunCore

struct VisionImageTo3DIdol: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "image-to-3d-idol",
        abstract: "Generate object Gaussian splats with native MLX IDOL.")

    @Argument(help: "Object image. Native BiRefNet removes opaque backgrounds; transparent alpha is preserved.")
    var input: String
    @Option(help: "Local speridlabs/idol directory containing the four published safetensors checkpoints.")
    var model: String
    @Option(help: "Local DINOv3 ViT-H+/16 directory containing config.json and model.safetensors.")
    var dino: String
    @Option(help: "Local FLUX.2-dev vae directory containing config.json and diffusion_pytorch_model.safetensors.")
    var vae: String
    @Option(name: [.customShort("o"), .long], help: "New output directory for object.ply, input.png, camera.json and manifest.json.")
    var output: String
    @Option(help: "Input framing mode: synthetic or real.")
    var mode: String = "synthetic"
    @Flag(help: "Skip foreground detection and cropping; preserve supplied framing and composite alpha on black.")
    var alreadyFramed = false
    @Option(help: "MLX seed. Reproducible within this backend; differs from PyTorch random streams.")
    var seed: UInt64 = 42
    @Option(help: "Euler sampling steps (1–1000).")
    var steps: Int = 50
    @Option(help: "Guidance w in (1+w)*conditional-w*unconditional.")
    var guidance: Float = 3
    @Option(help: "Flow time shift in (0,100].")
    var timeShift: Float = 3
    @Option(help: "Gaussian count (32768–1048576, multiple of 32).")
    var numGaussians: Int = 262144
    @Flag(help: "Verify checkpoint bytes and image header, then print JSON without model inference.")
    var dryRun = false
    @Flag(help: "Print the completed run manifest as JSON.")
    var json = false

    func settings() throws -> IdolSamplingSettings {
        try IdolSamplingSettings(steps: steps, guidance: guidance, timeShift: timeShift,
                                 gaussianCount: numGaussians, seed: seed)
    }

    mutating func validate() throws {
        guard IdolInputMode(rawValue: mode) != nil else { throw ValidationError("--mode must be synthetic or real.") }
        _ = try settings()
    }

    mutating func run() async throws {
        let settings = try settings()
        guard let inputMode = IdolInputMode(rawValue: mode) else { throw ValidationError("--mode must be synthetic or real.") }
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        let outputURL = URL(fileURLWithPath: output).standardizedFileURL
        let resources = IdolResources(root: URL(fileURLWithPath: model), dinoRoot: URL(fileURLWithPath: dino),
                                      vaeRoot: URL(fileURLWithPath: vae))
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
            throw ValidationError("Output directory already exists: \(outputURL.path)")
        }
        _ = try VFXImageInputValidator.inspectAndValidate([inputURL])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        if dryRun {
                        let checkpoints = try resources.verify()
            let plan = IdolPlan(input: inputURL.path, output: outputURL.path, settings: settings,
                                mode: inputMode, alreadyFramed: alreadyFramed, checkpoints: checkpoints)
            print(String(decoding: try encoder.encode(plan), as: UTF8.self))
            return
        }
        try MLXBundleSupport.ensureAvailable(quiet: true)
        let result = try IdolGenerator.generate(imageURL: inputURL, outputDirectory: outputURL, resources: resources,
            mode: inputMode, alreadyFramed: alreadyFramed, settings: settings) { message in
                CLIStderr.write("[image-to-3d-idol] \(message)\n")
            }
        if json { print(String(decoding: try encoder.encode(result), as: UTF8.self)) }
        else { print(outputURL.appending(path: "manifest.json").path) }
    }
}

private struct IdolPlan: Encodable {
    let status = "planned"
    let backend = "mere.run-native-mlx"
    let input: String
    let output: String
    let settings: IdolSamplingSettings
    let mode: IdolInputMode
    let alreadyFramed: Bool
    let checkpoints: [ModelArtifactPin]
}
