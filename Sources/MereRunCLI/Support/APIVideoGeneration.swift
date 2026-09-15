import Foundation
import MereRunCore

/// Translates the API's typed fields and optional CLI syntax into shared settings.
/// It does not run a command or acquire admission; the HTTP route owns the lease.
enum APIVideoGeneration {
    static func options(
        _ plan: APIServerContract.VideoGenerationPlan,
        outputURL: URL
    ) throws -> VideoGenerationOptions {
        var arguments: VideoGenerate
        do {
            // Parse only the compatibility option array. JSON fields never become argv.
            arguments = try VideoGenerate.parse(["api-video"] + plan.options)
        } catch {
            throw APIRequestValidationError.invalidField("options", VideoGenerate.message(for: error))
        }
        arguments.prompt = plan.prompt
        arguments.model = plan.modelID
        arguments.width = plan.width
        arguments.height = plan.height
        arguments.duration = plan.seconds
        arguments.numFrames = plan.numFrames
        arguments.fps = Double(plan.fps)
        arguments.seed = plan.seed
        arguments.quality = plan.quality
        arguments.outputMode = plan.outputMode
        return arguments.makeGenerationOptions(outputURL: outputURL)
    }

    static func generate(
        _ plan: APIServerContract.VideoGenerationPlan,
        outputURL: URL,
        prepareRuntime: @Sendable () throws -> Void = { try MLXBundleSupport.ensureAvailable(quiet: true) },
        executor: VideoGenerationOperation.Executor = VideoGenerationOperation.generate
    ) async throws -> VideoGenerationOutcome {
        let settings = try options(plan, outputURL: outputURL)
        let outcome = try await VideoGenerationOperation.execute(
            settings, prepareRuntime: prepareRuntime, executor: executor
        )
        guard !outcome.isDirectory, FileManager.default.fileExists(atPath: outputURL.path) else {
            throw APIRequestValidationError.invalidField("output", "video generation completed without an MP4 artifact")
        }
        if let report = outcome.timings {
            try emitLTXVideoTimingReport(report, printToStandardError: settings.timings, outputPath: settings.timingsOutput)
        }
        return outcome
    }
}
