import ArgumentParser
import AudioCore
import Foundation
import MereRunCore

extension MusicGenerate {
    var isYuE2Request: Bool {
        model == YuE2Resources.modelID || YuE2Resources.looksLikeRoot(resolveUserPath(model))
    }

    func resolvedYuE2Plan(explicitDurationSeconds: Float?) throws -> YuE2GenerationPlan {
        try validateStandaloneMusicOptions(modelName: "YuE2")
        if miniMaxOutputSampleRate != nil || miniMaxLoadingStrategy != nil || miniMaxPerformanceMode != nil
            || miniMaxSamplingTier != nil || miniMaxFlowStrategy != nil || miniMaxFlowSolver != nil
            || miniMaxAutoregressiveGuidanceFrames != nil || miniMaxFlowGuidanceEnd != nil
            || miniMaxSeedStrategy != nil || miniMaxProfileOutput != nil || miniMaxCompose
            || miniMaxComposerModelRoot != nil || miniMaxRequireComposerInstalled || miniMaxCompositionOutput != nil
            || miniMaxLyricPreflightPolicy != .warn || miniMaxComposerModel != TextChat.defaultChatModelId {
            throw ValidationError("MiniMax composer, runtime, and profiling options do not apply to YuE2.")
        }
        guard lrcFile == nil else { throw ValidationError("YuE2 accepts plain lyrics; use --lyrics or --lyrics-file.") }
        guard lyricsFile == nil || lyrics.isEmpty, !instrumental || (lyricsFile == nil && lyrics.isEmpty) else {
            throw ValidationError("Pass only one of --lyrics, --lyrics-file, or --instrumental.")
        }
        if noRecipe && recipeOutput != nil { throw ValidationError("--recipe-output conflicts with --no-recipe.") }
        if yue2Options.planning == .off, yue2Options.abcOutput != nil || yue2Options.abcMaximumTokens != nil {
            throw ValidationError("--abc-output and --abc-max-tokens require full or melody score mode.")
        }
        if yue2Options.abcFile != nil && yue2Options.abcMaximumTokens != nil {
            throw ValidationError("--abc-max-tokens does not apply when --abc-file supplies the score.")
        }
        if let minimum = miniMaxMinimumFrames, !(1...9000).contains(minimum) {
            throw ValidationError("--min-frames must be between 1 and 9000.")
        }
        let durationFrames = try explicitDurationSeconds.map { try Self.yue2Frames(seconds: $0, minimum: false) }
        let minimumDurationFrames = try miniMaxMinimumDurationSeconds.map { try Self.yue2Frames(seconds: $0, minimum: true) }
        if let durationFrames, let frames = miniMaxMaximumFrames, durationFrames != frames {
            throw ValidationError("--duration and --max-frames must describe the same 25 Hz frame limit.")
        }
        if let minimumDurationFrames, let frames = miniMaxMinimumFrames, minimumDurationFrames != frames {
            throw ValidationError("--minimum-duration and --min-frames must describe the same decoded duration floor.")
        }
        var semantic = YuE2Sampling()
        semantic.maximumTokens = miniMaxMaximumFrames ?? durationFrames ?? 9000
        semantic.minimumTokens = miniMaxMinimumFrames ?? minimumDurationFrames ?? min(200, semantic.maximumTokens)
        if let value = yue2Options.temperature { semantic.temperature = value }
        if let value = yue2Options.topP { semantic.topP = value }
        if let value = yue2Options.topK { semantic.topK = value }
        if let value = yue2Options.repetitionPenalty { semantic.repetitionPenalty = value }
        var score = YuE2Sampling.score
        if let maximum = yue2Options.abcMaximumTokens {
            score.maximumTokens = maximum
            score.minimumTokens = min(score.minimumTokens, maximum)
        }
        let text = try lyricsFile.map { try String(contentsOf: resolveUserPath($0), encoding: .utf8) } ?? lyrics
        let abc = try yue2Options.abcFile.map { try String(contentsOf: resolveUserPath($0), encoding: .utf8) }
        return try YuE2GenerationPlan(
            style: caption, lyrics: text, planning: yue2Options.planning ?? .full, abc: abc,
            seed: seed ?? 831001, guidanceScale: guidanceScale, scoreSampling: score,
            semanticSampling: semantic, steps: steps ?? 32
        )
    }

    static func yue2Frames(seconds: Float, minimum: Bool) throws -> Int {
        guard seconds.isFinite, seconds > 0, seconds <= 360 else {
            throw ValidationError("YuE2 duration must be greater than 0 and at most 360 seconds.")
        }
        // The decoder emits 1920*T - 64 samples at 48 kHz.
        let frames = minimum ? Int(ceil((Double(seconds) * 48_000 + 64) / 1920)) : Int(Double(seconds) * 25)
        guard (1...9000).contains(frames) else {
            throw ValidationError("The requested YuE2 duration requires a frame count outside 1...9000.")
        }
        return frames
    }

    func runYuE2(explicitDurationSeconds: Float?, exportPlan: AudioExportPlan) async throws {
        let plan = try resolvedYuE2Plan(explicitDurationSeconds: explicitDurationSeconds)
        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-yue2", defaultExtension: "wav")
        let abcURL = plan.planning == .off ? nil : yue2Options.abcOutput.map(resolveUserPath)
            ?? outputURL.deletingPathExtension().appendingPathExtension("abc")
        let recipeURL = noRecipe ? nil : recipeOutput.map(resolveUserPath)
            ?? outputURL.deletingPathExtension().appendingPathExtension("recipe.json")
        let outputs = [outputURL, abcURL, recipeURL].compactMap { $0?.standardizedFileURL.resolvingSymlinksInPath() }
        let inputs = [lyricsFile, yue2Options.abcFile].compactMap { $0 }.map { resolveUserPath($0).resolvingSymlinksInPath() }
        guard Set(outputs).count == outputs.count, Set(inputs).isDisjoint(with: outputs) else {
            throw ValidationError("YuE2 output paths must be distinct and must not overwrite the lyrics or input score.")
        }
        let rootURL: URL
        if model == YuE2Resources.modelID {
            do { rootURL = try ModelResolver().resolve(.yue2).rootURL }
            catch {
                throw ValidationError("YuE2 is not installed. Review its noncommercial license, then run "
                    + "`mere.run model pull music-yue2 --accept-model-license`.")
            }
        } else { rootURL = resolveUserPath(model) }
        if !quiet { CLIStderr.write("Loading experimental native YuE2 from \(rootURL.path)\n") }
        let operation = YuE2GenerationOperation(resources: YuE2Resources(rootURL: rootURL))
        let progressStream = progressJson ? JSONProgressStream() : nil
        let result = try await operation.generate(plan) { event in
            if let progressStream {
                progressStream.report(stage: event.stage.rawValue, step: max(0, event.completed - 1), totalSteps: event.total)
            } else if !quiet, event.completed == 1 || event.completed == event.total || event.completed % 50 == 0 {
                CLIStderr.write("YuE2 \(event.stage.rawValue): \(event.completed)/\(event.total)\n")
            }
        }
        progressStream?.finish()
        try Task.checkCancellation()
        let exported = try AudioExportService.write(result.waveform, plan: exportPlan, to: outputURL)
        var sidecars: [RunReceipt.Output] = []
        if let abcURL, let abc = result.abc {
            try Self.writeYuE2Artifact(Data(abc.utf8), to: abcURL)
            sidecars.append(.init(url: abcURL, kind: .text, role: "score"))
        }
        if let recipeURL {
            let recipe = YuE2Recipe(plan: plan, modelRoot: rootURL.path, abc: result.abc,
                                   scoreTokens: result.scoreTokens, codecTokens: result.codecTokens,
                                   scoreTruncated: result.scoreTruncated, musicTruncated: result.musicTruncated,
                                   export: exportPlan.options, exportStatistics: exported.statistics,
                                   outputSHA256: try ModelArtifactPin.fileSHA256(outputURL))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try Self.writeYuE2Artifact(encoder.encode(recipe), to: recipeURL)
            sidecars.append(.init(url: recipeURL, kind: .json, role: "recipe"))
        }
        if result.scoreTruncated || result.musicTruncated {
            CLIStderr.write("YuE2 reached a token limit before an end token (score: \(result.scoreTruncated), music: \(result.musicTruncated)).\n")
        }
        print(outputURL.path)
        try RunReceipt.emit(RunReceipt.generatedAudioOutputs(audio: outputURL, sidecars: sidecars), enabled: receipt)
    }

    private static func writeYuE2Artifact(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
}

private struct YuE2Recipe: Encodable {
    let schemaVersion = 1
    let runtime = "native-swift-mlx"
    let randomNumberGenerator = "mlx-request-local-v1"
    let expectedModelRevision = YuE2Resources.revision
    let expectedVAERevision = YuE2Resources.vaeRevision
    let plan: YuE2GenerationPlan
    let modelRoot: String
    let abc: String?
    let scoreTokens: [Int]
    let codecTokens: [Int]
    let scoreTruncated: Bool
    let musicTruncated: Bool
    let export: AudioExportOptions
    let exportStatistics: AudioExportStatistics
    let outputSHA256: String
}
