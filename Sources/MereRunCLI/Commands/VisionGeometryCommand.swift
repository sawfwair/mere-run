import ArgumentParser
import Foundation
import MereRunCore

struct VisionGeometry: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "geometry",
        abstract: "Generate metric depth, normals, camera intrinsics, and a point cloud with native MoGe-2."
    )

    @Argument(help: "Input image path.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output directory.")
    var output: String?

    @Option(name: [.long], help: "Managed model id, model root, or pinned model.onnx path.")
    var model: String?

    @Option(name: [.long], help: "Quality level from 0 through 9. (default: 9)")
    var resolutionLevel: Int = MoGe2GenerationSettings.defaultResolutionLevel

    @Option(name: [.long], help: "Override the DINO base-token count (1...3600).")
    var tokenCount: Int?

    @Option(name: [.long], help: "Maximum number of points written to PLY.")
    var maxPoints: Int?

    @Flag(name: [.long], help: "Validate and print the execution plan without loading weights.")
    var dryRun = false

    @Flag(name: [.long], help: "Print the structured result on stdout.")
    var json = false

    mutating func run() async throws {
        do {
            let request = try makeGenerationRequest()
            if dryRun {
                let plan = try MoGe2GenerationOperation.prepare(request)
                print(try Self.jsonString(Self.makePlan(plan)))
                return
            }
            let result = try await MoGe2GenerationOperation.execute(
                request, progress: { message in CLIStderr.write("[geometry] \(message)\n") }
            )
            if json {
                print(try Self.jsonString(VisionGeometryRunPayload(result: result)))
            } else {
                print(result.export.manifestURL.path)
            }
        } catch let error as MoGe2GenerationError {
            switch error {
            case .invalidResolutionLevel:
                throw ValidationError("--resolution-level must be between 0 and 9")
            case .invalidTokenCount:
                throw ValidationError("--token-count must be between 1 and 3600")
            case .invalidMaximumPointCount:
                throw ValidationError("--max-points must be positive")
            case .inputNotFound:
                throw ValidationError(error.localizedDescription)
            }
        }
    }

    func makeGenerationRequest() throws -> MoGe2GenerationRequest {
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        return MoGe2GenerationRequest(
            imageURL: inputURL,
            outputDirectory: Self.resolveOutputURL(output, inputURL: inputURL),
            model: model,
            settings: try MoGe2GenerationSettings(
                resolutionLevel: resolutionLevel, tokenCount: tokenCount, maximumPointCount: maxPoints
            )
        )
    }

    static func resolveOutputURL(_ raw: String?, inputURL: URL) -> URL {
        if let raw, !raw.isEmpty {
            return URL(fileURLWithPath: raw).standardizedFileURL
        }
        return inputURL.deletingLastPathComponent().appendingPathComponent(
            "\(inputURL.deletingPathExtension().lastPathComponent)-geometry",
            isDirectory: true
        )
    }

    static func makePlan(_ plan: MoGe2GenerationPlan) -> VisionGeometryPlanPayload {
        let request = plan.request
        let tokenGrid = plan.tokenGrid
        let installed = ManagedModelResolver.resolveInstalledModel(
            id: MoGe2GenerationRequest.defaultModelID
        )
        return VisionGeometryPlanPayload(
            status: "planned",
            inputPath: request.imageURL.path,
            outputDirectory: request.outputDirectory.path,
            model: request.model ?? MoGe2GenerationRequest.defaultModelID,
            managedModelInstalled: installed != nil,
            imageWidth: plan.dimensions.width,
            imageHeight: plan.dimensions.height,
            tokenCount: request.settings.configuration.effectiveTokenCount,
            tokenRows: tokenGrid.rows,
            tokenColumns: tokenGrid.columns,
            networkWidth: tokenGrid.columns * 14,
            networkHeight: tokenGrid.rows * 14,
            outputKinds: [
                "metric-depth-exr", "depth-preview-png", "normal-exr", "normal-preview-png",
                "validity-mask-png", "camera-json", "point-cloud-ply", "manifest-json",
            ]
        )
    }

    private static func jsonString<T: Encodable>(_ payload: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(payload), as: UTF8.self)
    }
}

struct VisionGeometryPlanPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let inputPath: String
    let outputDirectory: String
    let model: String
    let managedModelInstalled: Bool
    let imageWidth: Int
    let imageHeight: Int
    let tokenCount: Int
    let tokenRows: Int
    let tokenColumns: Int
    let networkWidth: Int
    let networkHeight: Int
    let outputKinds: [String]

    init(
        schemaVersion: Int = 1,
        status: String,
        inputPath: String,
        outputDirectory: String,
        model: String,
        managedModelInstalled: Bool,
        imageWidth: Int,
        imageHeight: Int,
        tokenCount: Int,
        tokenRows: Int,
        tokenColumns: Int,
        networkWidth: Int,
        networkHeight: Int,
        outputKinds: [String]
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.inputPath = inputPath
        self.outputDirectory = outputDirectory
        self.model = model
        self.managedModelInstalled = managedModelInstalled
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.tokenCount = tokenCount
        self.tokenRows = tokenRows
        self.tokenColumns = tokenColumns
        self.networkWidth = networkWidth
        self.networkHeight = networkHeight
        self.outputKinds = outputKinds
    }
}

struct VisionGeometryRunPayload: Codable {
    let schemaVersion: Int
    let status: String
    let manifestPath: String
    let modelID: String
    let tokenCount: Int
    let width: Int
    let height: Int
    let units: GeometryValueUnits
    let validPixelCount: Int
    let focal: Double
    let shift: Double
    let metricScale: Double
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let postprocessSeconds: Double
    let artifacts: [GeometryArtifact]

    init(result: MoGe2RunResult, schemaVersion: Int = 1, status: String = "completed") {
        let manifest = result.export.manifest
        self.schemaVersion = schemaVersion
        self.status = status
        self.manifestPath = result.export.manifestURL.path
        self.modelID = manifest.model.modelID
        self.tokenCount = result.tokenCount
        self.width = manifest.width
        self.height = manifest.height
        self.units = manifest.units
        self.validPixelCount = manifest.depthStatistics.validPixelCount
        self.focal = result.focalShift.focal
        self.shift = result.focalShift.shift
        self.metricScale = result.metricScale
        self.modelLoadSeconds = result.modelLoadSeconds
        self.inferenceSeconds = result.inferenceSeconds
        self.postprocessSeconds = result.postprocessSeconds
        self.artifacts = manifest.artifacts
    }
}
