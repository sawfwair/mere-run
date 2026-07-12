import ArgumentParser
import MereRunCore

/// Canonical image-family spelling retained by the managed model catalog.
/// `vision image-to-3d` is an equivalent VFX-oriented entry point.
struct ImageReconstruct3D: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "reconstruct-3d",
        abstract: "Reconstruct a colored object mesh from one image with native TripoSR."
    )

    @Argument(help: "Input object image path. Transparent PNG foregrounds are cropped automatically.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output mesh directory.")
    var output: String?

    @Option(name: [.long], help: "Managed model id, pinned model.ckpt, or verified converted package.")
    var model: String?

    @Option(name: [.long], help: "Native density-grid resolution from 2 through 512.")
    var resolution: Int = 256

    @Option(name: [.long], help: "Activated-density isosurface threshold.")
    var densityThreshold: Float = TripoSRConfiguration.production.densityThreshold

    @Option(name: [.long], help: "Transparent foreground occupancy ratio in (0, 1].")
    var foregroundRatio: Float = 0.85

    @Flag(name: [.long], help: "Skip transparent-foreground crop/pad and treat the image as already framed.")
    var alreadyFramed = false

    @Flag(name: [.long], help: "Skip neural vertex-color queries and export geometry only.")
    var noVertexColors = false

    @Flag(name: [.long], help: "Verify the checkpoint and print the execution plan without loading weights.")
    var dryRun = false

    @Flag(name: [.long], help: "Print structured JSON on stdout.")
    var json = false

    mutating func run() async throws {
        try await VisionImageTo3D.execute(
            input: input,
            output: output,
            model: model,
            resolution: resolution,
            densityThreshold: densityThreshold,
            foregroundRatio: foregroundRatio,
            alreadyFramed: alreadyFramed,
            noVertexColors: noVertexColors,
            dryRun: dryRun,
            json: json
        )
    }
}
