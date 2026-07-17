import ArgumentParser
import Foundation
import MediaIO
import MLX
import MereRunCore

enum LTXVideoSessionResponseStatus: String, Codable, Hashable, Sendable {
    case result
    case error
}

struct LTXVideoSessionRequest: Codable, Hashable, Sendable {
    let id: String?
    let prompt: String
    let output: String
    let width: Int?
    let height: Int?
    let numFrames: Int?
    let fps: Int?
    let seed: Int?
    let image: String?
    let imageStrength: Float?
    let endImage: String?
    let endImageStrength: Float?
}

struct LTXVideoSessionResponse: Codable, Hashable, Sendable {
    let status: LTXVideoSessionResponseStatus
    let id: String?
    let output: String?
    let seed: Int?
    let timings: LTXVideoTimingReport?
    let error: String?

    static func result(
        id: String?,
        output: String,
        seed: Int,
        timings: LTXVideoTimingReport
    ) -> Self {
        Self(
            status: .result,
            id: id,
            output: output,
            seed: seed,
            timings: timings,
            error: nil
        )
    }

    static func failure(id: String?, error: Error) -> Self {
        let message = (error as? LocalizedError)?.errorDescription
            ?? String(describing: error)
        return Self(
            status: .error,
            id: id,
            output: nil,
            seed: nil,
            timings: nil,
            error: message
        )
    }
}

struct VideoSession: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "session",
        abstract: "Keep standalone distilled LTX 2.3 resident for JSONL generation requests.",
        discussion: """
        Loads the standalone video-ltx23-av-mlx bundle once, then reads one JSON
        request per line from stdin and writes one JSON result per line to stdout.
        Diagnostics go to stderr. Requests are processed serially.

        This resident path deliberately rejects the full dev + distilled-LoRA
        bundle: that two-stage lane fuses LoRA weights in place and must currently
        reload before another generation.

        Request keys use snake_case. Required keys are prompt and output. Optional
        keys are id, width, height, num_frames, fps, seed, image,
        image_strength, end_image, and end_image_strength.
        """
    )

    @Option(name: [.customShort("m"), .long], help: "Managed standalone distilled LTX 2.3 model id or local model root.")
    var model: String = ModelResolver.ModelID.ltxVideo23AVMLX.rawValue

    @Option(name: [.customLong("model-root")], help: "Local standalone distilled LTX 2.3 model root. Takes precedence over --model.")
    var modelRoot: String?

    @Flag(name: [.short, .long], help: "Suppress session diagnostics on stderr.")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let rootURL = try await resolveVideoModelRoot(
            explicitModelRoot: modelRoot,
            requestedModel: model,
            variant: .unifiedAV,
            allowAutoDownload: true
        )
        try validateNativeModelRoot(rootURL)
        guard isLTX23SplitModelRoot(rootURL), !isLTX23FullModelRoot(rootURL) else {
            throw ValidationError(
                "video session requires the standalone \(ModelResolver.ModelID.ltxVideo23AVMLX.rawValue) bundle; "
                    + "the full dev + distilled-LoRA lane must reload between generations."
            )
        }

        let generator = LTXUnifiedAVGenerator()
        let loadTimings = try await generator.load(modelRoot: rootURL)
        if !quiet {
            CLIStderr.write("LTX video session ready: \(rootURL.path)\n")
            CLIStderr.write(String(format: "Model load: %.3fs\n", loadTimings.totalSeconds))
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.sortedKeys]
        var completedRequestCount = 0

        while let line = readLine(strippingNewline: true) {
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedLine.isEmpty else { continue }

            let request: LTXVideoSessionRequest
            do {
                request = try decoder.decode(LTXVideoSessionRequest.self, from: Data(trimmedLine.utf8))
            } catch {
                try writeSessionResponse(.failure(id: nil, error: error), encoder: encoder)
                continue
            }

            do {
                let response = try await process(
                    request: request,
                    generator: generator,
                    modelRoot: rootURL,
                    loadTimings: completedRequestCount == 0 ? loadTimings : LTXLoadTimings(),
                    residentModelReused: completedRequestCount > 0
                )
                try writeSessionResponse(response, encoder: encoder)
                completedRequestCount += 1
            } catch {
                try writeSessionResponse(.failure(id: request.id, error: error), encoder: encoder)
            }
            Memory.clearCache()
        }

        await generator.unload()
        if !quiet {
            CLIStderr.write("LTX video session closed after \(completedRequestCount) generation(s).\n")
        }
    }

    private func process(
        request: LTXVideoSessionRequest,
        generator: LTXUnifiedAVGenerator,
        modelRoot: URL,
        loadTimings: LTXLoadTimings,
        residentModelReused: Bool
    ) async throws -> LTXVideoSessionResponse {
        let requestStart = videoMonotonicSeconds()
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw ValidationError("prompt cannot be empty")
        }

        let width = request.width ?? 768
        let height = request.height ?? 512
        let numFrames = request.numFrames ?? 65
        let fps = request.fps ?? 24
        let seed = request.seed ?? 42
        let imageStrength = request.imageStrength ?? 1
        let endImageStrength = request.endImageStrength ?? 1

        guard width >= 64, height >= 64, width % 64 == 0, height % 64 == 0 else {
            throw ValidationError("width and height must be at least 64 and divisible by 64")
        }
        guard numFrames >= 9, numFrames % 8 == 1 else {
            throw ValidationError("num_frames must be at least 9 and satisfy 8n+1")
        }
        guard fps > 0 else {
            throw ValidationError("fps must be positive")
        }
        guard (0...1).contains(imageStrength), (0...1).contains(endImageStrength) else {
            throw ValidationError("image strengths must be between 0 and 1")
        }
        guard request.endImage == nil || request.image != nil else {
            throw ValidationError("end_image requires image")
        }

        let sourceImageURL = try existingFileURL(request.image, field: "image")
        let endImageURL = try existingFileURL(request.endImage, field: "end_image")
        let outputURL = URL(fileURLWithPath: request.output).standardizedFileURL
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let result = try await generator.generate(
            options: LTXUnifiedAVGenerationOptions(
                prompt: prompt,
                width: width,
                height: height,
                numFrames: numFrames,
                fps: fps,
                seed: seed,
                sourceImageURL: sourceImageURL,
                imageStrength: imageStrength,
                endImageURL: endImageURL,
                endImageStrength: endImageStrength
            )
        )

        let writeStart = videoMonotonicSeconds()
        try LTXVideoMP4Writer.writeMP4(
            frames: result.frames,
            fps: fps,
            to: outputURL,
            audioWaveform: result.audioWaveform,
            audioSampleRate: result.audioSampleRate
        )
        guard MediaVideoIO.hasAudioTrack(outputURL) else {
            throw ValidationError("session output has no audio track at \(outputURL.path)")
        }
        let writeSeconds = videoMonotonicSeconds() - writeStart
        let timings = LTXVideoTimingReport(
            mode: "resident-standalone-distilled-unified-av",
            modelRoot: modelRoot.path,
            residentModelReused: residentModelReused,
            load: loadTimings,
            generation: result.timings,
            unloadSeconds: 0,
            mp4WriteSeconds: writeSeconds,
            totalSeconds: loadTimings.totalSeconds + videoMonotonicSeconds() - requestStart
        )
        return .result(id: request.id, output: outputURL.path, seed: seed, timings: timings)
    }

    private func existingFileURL(_ path: String?, field: String) throws -> URL? {
        guard let path else { return nil }
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("\(field) file not found: \(url.path)")
        }
        return url
    }

    private func writeSessionResponse(
        _ response: LTXVideoSessionResponse,
        encoder: JSONEncoder
    ) throws {
        var data = try encoder.encode(response)
        data.append(0x0A)
        FileHandle.standardOutput.write(data)
    }
}
