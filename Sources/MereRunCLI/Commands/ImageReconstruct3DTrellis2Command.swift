import ArgumentParser
import MereRunCore

struct ImageReconstruct3DTrellis2: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "reconstruct-3d-trellis2",
        abstract: "Reconstruct a 512-resolution PBR O-Voxel mesh with native MLX TRELLIS.2."
    )

    @Argument(help: "Input object image path. Transparent alpha is required unless --already-framed is used.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output mesh and PBR artifact directory.")
    var output: String?

    @Option(name: [.long], help: "Managed TRELLIS.2 model id or verified checkpoint directory.")
    var model: String?

    @Option(name: [.long], help: "Deterministic MLX random seed.")
    var seed: UInt64 = 42

    @Option(name: [.long], help: "Re-roll only the texture flow with this seed, keeping the geometry of --seed.")
    var textureSeed: UInt64?

    @Option(name: [.long], help: "Safety limit for decoded 512-resolution O-Voxels.")
    var maxTokens: Int = Trellis2Generator.defaultMaximumSparseTokens

    @Flag(name: [.long], help: "Preserve framing and composite alpha over black without foreground cropping.")
    var alreadyFramed = false

    @Flag(name: [.long], help: "Export the raw porous dual-grid crust instead of the watertight narrow-band remesh.")
    var noRemesh = false

    @Option(name: [.long], help: "Narrow-band half-width in voxels for the watertight remesh.")
    var remeshBand: Float = 1

    @Option(name: [.long], help: "Morphological-closing radius in voxels. 0 disables cavity sealing.")
    var sealRadius: Int = 12

    @Flag(name: [.long], help: "Verify inputs and checkpoints, then print the execution plan without loading weights.")
    var dryRun = false

    @Flag(name: [.long], help: "Print structured JSON on stdout.")
    var json = false

    mutating func run() async throws {
        try await VisionImageTo3DTrellis2.execute(
            input: input,
            output: output,
            model: model,
            seed: seed,
            textureSeed: textureSeed,
            maxTokens: maxTokens,
            alreadyFramed: alreadyFramed,
            noRemesh: noRemesh,
            remeshBand: remeshBand,
            sealRadius: sealRadius,
            dryRun: dryRun,
            json: json
        )
    }
}
