import Crypto
import Foundation
import MediaIO
import MereRunCore

#if canImport(Darwin)
import Darwin
#endif

#if canImport(CoreGraphics)
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
#endif

// MARK: - Report types

enum GateStatus: String, Codable {
    case passed
    case warned
    case failed
    case skipped
}

struct GateObservation: Codable {
    let hash: String
    let secondRunHash: String?
    let wallSeconds: Double
    let decodeTps: Double?
    let semanticFailure: String?
}

struct GateResult: Codable {
    let id: String
    let status: GateStatus
    let detail: String
    let observation: GateObservation?
}

struct GateReport: Codable {
    let createdAt: String
    let hostModel: String
    let results: [GateResult]
}

struct GateBaseline: Codable {
    let hash: String
    let wallSeconds: Double
    let decodeTps: Double?
}

struct GateBaselines: Codable {
    var entries: [String: GateBaseline] = [:]
}

enum GateBaselineStore {
    static var applicationSupport: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
    }

    static var url: URL {
        applicationSupport
            .appendingPathComponent("gate", isDirectory: true)
            .appendingPathComponent("baselines.json")
    }

    static func load() -> GateBaselines {
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode(GateBaselines.self, from: data) else {
            return GateBaselines()
        }
        return decoded
    }

    static func save(_ baselines: GateBaselines) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(baselines).write(to: url)
    }
}

// MARK: - Check definitions

struct GateCheck: Sendable {
    let id: String
    let suite: String
    let requiredModels: [String]
    let comparesBaseline: Bool
    let successDetail: String
    let run: @Sendable (GateRunner) async throws -> GateObservation

    init(
        id: String,
        suite: String,
        requiredModels: [String],
        comparesBaseline: Bool = true,
        successDetail: String = "true generation completed and decoded",
        run: @escaping @Sendable (GateRunner) async throws -> GateObservation
    ) {
        self.id = id
        self.suite = suite
        self.requiredModels = requiredModels
        self.comparesBaseline = comparesBaseline
        self.successDetail = successDetail
        self.run = run
    }
}

enum GateChecks {
    /// A fixed paragraph repeated to push prompts past ~1200 tokens — the
    /// regime where the stale-metallib SDPA corruption lived undetected.
    static let longContextFiller = String(
        repeating: "The runtime schedules quantized matmuls, rotary embeddings, and "
            + "attention over unified memory while the sampler stays on the GPU. ",
        count: 90
    )

    static let all: [GateCheck] =
        textChecks + speechChecks + visionChecks + imageChecks + embedChecks + videoChecks

    private static let textModels: [(label: String, id: String)] = [
        ("gemma", "text-chat-gemma4-12b-4bit"),
        ("ornith", "text-agent-ornith-35b-mlx"),
        ("lfm2", "text-chat-lfm25-a1b-8bit"),
    ]

    private static var textChecks: [GateCheck] {
        textModels.flatMap { model in
            [
                GateCheck(id: "text-\(model.label)-short", suite: "text", requiredModels: [model.id]) { runner in
                    try await runner.chatCheck(
                        model: model.id,
                        prompt: "List the first four prime numbers, comma separated, nothing else.",
                        maxTokens: 160
                    )
                },
                GateCheck(id: "text-\(model.label)-long", suite: "text", requiredModels: [model.id]) { runner in
                    try await runner.chatCheck(
                        model: model.id,
                        prompt: "Summarize the following note in exactly one sentence. NOTE: "
                            + longContextFiller,
                        maxTokens: 160
                    )
                },
            ]
        }
    }

    private static var speechChecks: [GateCheck] {
        let sentence = "The quality gate proves every voice and every model still speaks correctly."
        return [
            GateCheck(id: "tts-wav", suite: "speech", requiredModels: ["speech-tts-qwen3-nano"]) { runner in
                let wav = runner.workDirectory.appendingPathComponent("gate-tts.wav")
                let run = try await runner.exec(
                    ["speech", "synthesize", sentence, "-o", wav.path, "--temperature", "0", "-q"],
                    timeout: 600
                )
                let bytes = try Data(contentsOf: wav)
                return GateObservation(
                    hash: GateRunner.sha256(bytes),
                    secondRunHash: nil,
                    wallSeconds: run.wallSeconds,
                    decodeTps: nil,
                    semanticFailure: bytes.count > 10_000 ? nil : "WAV suspiciously small (\(bytes.count) bytes)"
                )
            },
            GateCheck(
                id: "stt-roundtrip", suite: "speech",
                requiredModels: ["speech-tts-qwen3-nano", "speech-asr-parakeet"]
            ) { runner in
                let wav = runner.workDirectory.appendingPathComponent("gate-tts.wav")
                if !FileManager.default.fileExists(atPath: wav.path) {
                    _ = try await runner.exec(
                        ["speech", "synthesize", sentence, "-o", wav.path, "--temperature", "0", "-q"],
                        timeout: 600
                    )
                }
                let first = try await runner.exec(
                    ["speech", "transcribe", wav.path, "-m", "speech-asr-parakeet"], timeout: 600
                )
                let second = try await runner.exec(
                    ["speech", "transcribe", wav.path, "-m", "speech-asr-parakeet"], timeout: 600
                )
                let transcript = GateRunner.normalizeWords(first.stdout)
                let overlap = GateRunner.wordOverlap(source: sentence, candidate: first.stdout)
                return GateObservation(
                    hash: GateRunner.sha256(Data(transcript.utf8)),
                    secondRunHash: GateRunner.sha256(Data(GateRunner.normalizeWords(second.stdout).utf8)),
                    wallSeconds: first.wallSeconds,
                    decodeTps: nil,
                    semanticFailure: overlap >= 0.85
                        ? nil
                        : String(format: "STT roundtrip lost the sentence (%.0f%% word overlap): %@", overlap * 100, transcript.prefix(120) as CVarArg)
                )
            },
        ]
    }

    private static var visionChecks: [GateCheck] {
        [
            GateCheck(id: "ocr-page", suite: "vision", requiredModels: ["vision-ocr-lighton"]) { runner in
                #if canImport(CoreGraphics)
                let page = runner.workDirectory.appendingPathComponent("gate-ocr.png")
                let sourceText = GateTextPageRenderer.render(to: page)
                let first = try await runner.exec(
                    ["vision", "ocr", page.path, "-m", "vision-ocr-lighton"], timeout: 600
                )
                let second = try await runner.exec(
                    ["vision", "ocr", page.path, "-m", "vision-ocr-lighton"], timeout: 600
                )
                let overlap = GateRunner.wordOverlap(source: sourceText, candidate: first.stdout)
                return GateObservation(
                    hash: GateRunner.sha256(Data(GateRunner.normalizeWords(first.stdout).utf8)),
                    secondRunHash: GateRunner.sha256(Data(GateRunner.normalizeWords(second.stdout).utf8)),
                    wallSeconds: first.wallSeconds,
                    decodeTps: nil,
                    semanticFailure: overlap >= 0.9
                        ? nil
                        : String(format: "OCR lost the page (%.0f%% word overlap)", overlap * 100)
                )
                #else
                throw GateError.unsupportedPlatform("OCR page rendering requires CoreGraphics")
                #endif
            }
        ]
    }

    private static var imageChecks: [GateCheck] {
        [
            GateCheck(id: "image-klein-seed7", suite: "image", requiredModels: ["image-klein-nano"]) { runner in
                let png = runner.workDirectory.appendingPathComponent("gate-klein.png")
                let run = try await runner.exec(
                    [
                        "image", "generate", "--model", "image-klein-nano",
                        "--prompt", "a lighthouse at dusk, oil painting",
                        "--seed", "7", "-o", png.path,
                    ],
                    timeout: 900
                )
                let bytes = try Data(contentsOf: png)
                return GateObservation(
                    hash: GateRunner.sha256(bytes),
                    secondRunHash: nil,
                    wallSeconds: run.wallSeconds,
                    decodeTps: nil,
                    semanticFailure: bytes.count > 50_000 ? nil : "PNG suspiciously small (\(bytes.count) bytes)"
                )
            }
        ]
    }

    private static var embedChecks: [GateCheck] {
        [
            GateCheck(id: "embeddings", suite: "embed", requiredModels: ["text-embed-qwen3-0.6b"]) { runner in
                let out1 = runner.workDirectory.appendingPathComponent("gate-embed-1.json")
                let out2 = runner.workDirectory.appendingPathComponent("gate-embed-2.json")
                let texts = ["quality gates protect platforms", "unified memory bandwidth", "gate"]
                let run = try await runner.exec(
                    ["text", "embed"] + texts + ["-o", out1.path], timeout: 300
                )
                _ = try await runner.exec(
                    ["text", "embed"] + texts + ["-o", out2.path], timeout: 300
                )
                return GateObservation(
                    hash: try GateRunner.embeddingVectorHash(try Data(contentsOf: out1)),
                    secondRunHash: try GateRunner.embeddingVectorHash(try Data(contentsOf: out2)),
                    wallSeconds: run.wallSeconds,
                    decodeTps: nil,
                    semanticFailure: nil
                )
            }
        ]
    }

    private static var videoChecks: [GateCheck] {
        [
            GateCheck(
                id: "video-ltx23-draft",
                suite: "video",
                requiredModels: ["video-ltx23-av-mlx"],
                comparesBaseline: false
            ) { runner in
                try await runner.videoCheck(
                    id: "ltx23-draft",
                    model: "video-ltx23-av-mlx",
                    arguments: [
                        "--quality", "draft",
                        "--output-mode", "video-only",
                    ],
                    requireAudio: false
                )
            },
            GateCheck(
                id: "video-ltx23-full-av",
                suite: "video",
                requiredModels: ["video-ltx23-full-mlx"],
                comparesBaseline: false
            ) { runner in
                try await runner.videoCheck(
                    id: "ltx23-full-av",
                    model: "video-ltx23-full-mlx",
                    arguments: [
                        "--quality", "final",
                        "--output-mode", "audio-video",
                        "--a2v-steps", "4",
                    ],
                    requireAudio: true
                )
            },
            GateCheck(
                id: "video-ltx23-a2vid",
                suite: "video",
                requiredModels: ["video-ltx23-a2vid-mlx"],
                comparesBaseline: false
            ) { runner in
                let audioURL = runner.workDirectory.appendingPathComponent("gate-a2vid-source.wav")
                try GateRunner.writeSineWaveFixture(to: audioURL)
                return try await runner.videoCheck(
                    id: "ltx23-a2vid",
                    model: "video-ltx23-a2vid-mlx",
                    arguments: [
                        "--audio", audioURL.path,
                        "--a2v-steps", "1",
                    ],
                    requireAudio: true
                )
            },
        ]
    }
}

// MARK: - Runner

enum GateError: Error, LocalizedError {
    case commandFailed(String, exitCode: Int32, stderr: String)
    case timedOut(String)
    case unsupportedPlatform(String)
    case invalidArtifact(String)

    var errorDescription: String? {
        switch self {
        case .commandFailed(let command, let code, let stderr):
            return "`\(command)` exited \(code): \(stderr.suffix(300))"
        case .timedOut(let command):
            return "`\(command)` timed out"
        case .unsupportedPlatform(let reason):
            return reason
        case .invalidArtifact(let reason):
            return reason
        }
    }
}

private final class GateProcessBox: @unchecked Sendable {
    let process: Process
    init(_ process: Process) { self.process = process }
}

private final class GateFileHandleBox: @unchecked Sendable {
    let fileHandle: FileHandle
    init(_ fileHandle: FileHandle) { self.fileHandle = fileHandle }
}

struct GateRunner: Sendable {
    struct ExecResult {
        let stdout: String
        let stderr: String
        let wallSeconds: Double
    }

    let executableURL: URL
    let workDirectory: URL

    /// Runs a real `mere.run` subcommand end-to-end in a subprocess.
    func exec(_ arguments: [String], timeout: TimeInterval) async throws -> ExecResult {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        let terminationEvents = AsyncStream<Void> { continuation in
            process.terminationHandler = { _ in
                continuation.yield()
                continuation.finish()
            }
        }

        let start = Date()
        try process.run()
        try? stdout.fileHandleForWriting.close()
        try? stderr.fileHandleForWriting.close()

        let box = GateProcessBox(process)
        let stdoutBox = GateFileHandleBox(stdout.fileHandleForReading)
        let stderrBox = GateFileHandleBox(stderr.fileHandleForReading)
        let stdoutTask = Task.detached {
            stdoutBox.fileHandle.readDataToEndOfFile()
        }
        let stderrTask = Task.detached {
            stderrBox.fileHandle.readDataToEndOfFile()
        }
        let watchdog = Task.detached {
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            if box.process.isRunning {
                box.process.terminate()
            }
        }
        for await _ in terminationEvents {
            break
        }
        watchdog.cancel()
        let wall = Date().timeIntervalSince(start)
        let timedOut = wall >= timeout
        let outData = await stdoutTask.value
        let errData = await stderrTask.value
        let outText = String(decoding: outData, as: UTF8.self)
        let errText = String(decoding: errData, as: UTF8.self)

        if timedOut {
            throw GateError.timedOut(arguments.joined(separator: " "))
        }
        guard process.terminationStatus == 0 else {
            throw GateError.commandFailed(
                arguments.prefix(3).joined(separator: " "),
                exitCode: process.terminationStatus,
                stderr: errText
            )
        }
        return ExecResult(stdout: outText, stderr: errText, wallSeconds: wall)
    }

    /// Temperature-0 chat run, executed twice for in-run determinism, with
    /// decode throughput parsed from --stats when the engine reports it.
    func chatCheck(model: String, prompt: String, maxTokens: Int) async throws -> GateObservation {
        func once() async throws -> (hash: String, tps: Double?, wall: Double, response: String) {
            let run = try await exec(
                [
                    "text", "chat", "--model", model, "--prompt", prompt,
                    "--temperature", "0", "--max-tokens", String(maxTokens),
                    "--thinking", "--stats",
                ],
                timeout: 900
            )
            let response = Self.stripStatsAndProgress(run.stdout)
            let tps = Self.parseDecodeTps(run.stdout + run.stderr)
            return (GateRunner.sha256(Data(response.utf8)), tps, run.wallSeconds, response)
        }
        let first = try await once()
        let second = try await once()
        return GateObservation(
            hash: first.hash,
            secondRunHash: second.hash,
            wallSeconds: second.wall,
            decodeTps: second.tps,
            semanticFailure: first.response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "empty response"
                : nil
        )
    }

    func videoCheck(
        id: String,
        model: String,
        arguments: [String],
        requireAudio: Bool
    ) async throws -> GateObservation {
        let outputURL = workDirectory.appendingPathComponent("gate-\(id).mp4")
        let run = try await exec(
            [
                "video", "generate", "A red cube rotates slowly on a neutral background.",
                "--model", model,
                "--width", "512",
                "--height", "320",
                "--num-frames", "9",
                "--fps", "24",
                "--seed", "7",
                "--output", outputURL.path,
                "--quiet",
            ] + arguments,
            timeout: 1_800
        )
        let bytes = try Data(contentsOf: outputURL)
        let semanticFailure = try validateVideoArtifact(outputURL, requireAudio: requireAudio)
        return GateObservation(
            hash: Self.sha256(bytes),
            secondRunHash: nil,
            wallSeconds: run.wallSeconds,
            decodeTps: nil,
            semanticFailure: semanticFailure
        )
    }

    func validateVideoArtifact(_ url: URL, requireAudio: Bool) throws -> String? {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let byteCount = (attributes[.size] as? NSNumber)?.intValue ?? 0
        guard byteCount > 1_024 else {
            return "MP4 suspiciously small (\(byteCount) bytes)"
        }

        let framesURL = workDirectory.appendingPathComponent(
            "decoded-\(url.deletingPathExtension().lastPathComponent)",
            isDirectory: true
        )
        let sequence = try MediaVideoIO.extractFrames(from: url, into: framesURL, endFrame: 1)
        guard !sequence.frameURLs.isEmpty,
              sequence.frameWidth > 0,
              sequence.frameHeight > 0 else {
            return "MP4 did not decode a video frame"
        }
        guard requireAudio else { return nil }
        guard MediaVideoIO.hasAudioTrack(url) else {
            return "MP4 has no audio track"
        }

        let audio = try MediaAudioIO.decodeSegment(
            url,
            startTime: 0,
            duration: 0.25,
            targetSampleRate: 16_000,
            channels: 2
        )
        guard !audio.samples.isEmpty else {
            return "MP4 audio track decoded no samples"
        }
        let meanSquare = audio.samples.reduce(0.0) { partial, sample in
            partial + Double(sample) * Double(sample)
        } / Double(audio.samples.count)
        return meanSquare.squareRoot() > 0.000_01 ? nil : "MP4 audio track is silent"
    }

    static func writeSineWaveFixture(to url: URL) throws {
        let sampleRate = 24_000
        let channels = 2
        let frameCount = sampleRate
        let samples = (0..<frameCount).flatMap { frame -> [Float] in
            let value = Float(sin(2 * Double.pi * 440 * Double(frame) / Double(sampleRate)) * 0.2)
            return [value, value]
        }
        try MediaAudioIO.writeFloatWAV(
            samples: samples,
            sampleRate: sampleRate,
            channels: channels,
            to: url
        )
    }

    static func modelInstalled(_ id: String) -> Bool {
        let root = ProcessInfo.processInfo.environment["MERERUN_MODELS_DIR"]
            .map { URL(fileURLWithPath: $0) }
            ?? GateBaselineStore.applicationSupport.appendingPathComponent("models", isDirectory: true)
        return modelInstalled(id, root: root)
    }

    static func modelInstalled(_ id: String, root: URL) -> Bool {
        let directURL = root.appendingPathComponent(id, isDirectory: true)
        if FileManager.default.fileExists(atPath: directURL.path) {
            return true
        }
        guard let spec = ManagedModelCatalog.spec(for: id) else {
            return false
        }
        return spec.resolutionFallbackIDs.contains { fallbackID in
            FileManager.default.fileExists(
                atPath: root.appendingPathComponent(fallbackID, isDirectory: true).path
            )
        }
    }

    static func hardwareModel() -> String {
        #if canImport(Darwin)
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var value = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &value, &size, nil, 0)
        let bytes = value.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
        #elseif os(Linux)
        return "Linux"
        #else
        return "Unknown"
        #endif
    }

    /// Hashes only the embedding vectors from a /v1/embeddings-shaped JSON
    /// document — response metadata (ids, created timestamps) legitimately
    /// varies per call and must not participate in determinism checks.
    static func embeddingVectorHash(_ data: Data) throws -> String {
        struct EmbedFile: Decodable {
            struct Item: Decodable { let embedding: [Double] }
            let data: [Item]
        }
        let decoded = try JSONDecoder().decode(EmbedFile.self, from: data)
        var canonical = ""
        for item in decoded.data {
            for value in item.embedding {
                canonical += String(format: "%.8f,", value)
            }
            canonical += ";"
        }
        return sha256(Data(canonical.utf8))
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func normalizeWords(_ text: String) -> String {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    static func wordOverlap(source: String, candidate: String) -> Double {
        let sourceWords = normalizeWords(source).split(separator: " ").map(String.init)
        guard !sourceWords.isEmpty else { return 1 }
        let candidateWords = Set(normalizeWords(candidate).split(separator: " ").map(String.init))
        let hit = sourceWords.filter(candidateWords.contains).count
        return Double(hit) / Double(sourceWords.count)
    }

    private static func stripStatsAndProgress(_ stdout: String) -> String {
        stdout
            .split(whereSeparator: \.isNewline)
            .filter { line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("[") && trimmed.contains("]") { return false }
                if trimmed.hasPrefix("time=") && trimmed.contains("decode_tps=") { return false }
                return true
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func parseDecodeTps(_ text: String) -> Double? {
        guard let range = text.range(of: "decode_tps=") else { return nil }
        let tail = text[range.upperBound...].prefix(16)
        let number = tail.prefix { $0.isNumber || $0 == "." }
        return Double(number)
    }
}

// MARK: - Deterministic OCR page

#if canImport(CoreGraphics)
enum GateTextPageRenderer {
    static let lines: [String] = [
        "Performance work rewards patience and measurement in equal parts.",
        "Every token kept on the accelerator is a round trip the processor",
        "never waits for, and every mask built once per generation replaces",
        "a thousand uploads. The buffer cache grows to the high-water mark",
        "and never shrinks, so a cap keeps the footprint honest. Blank",
        "frames scan host-side from a single readback, and decoder state",
        "only changes when a token is emitted downstream of the joint.",
    ]

    /// Renders the fixed paragraph to a PNG and returns the source text used,
    /// so the OCR check can score word overlap against it.
    @discardableResult
    static func render(to url: URL) -> String {
        let width = 1100
        let height = 460
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return lines.joined(separator: " ")
        }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        let font = CTFontCreateWithName("Helvetica" as CFString, 24, nil)
        for (index, line) in lines.enumerated() {
            let attributed = NSAttributedString(string: line, attributes: [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(red: 0, green: 0, blue: 0, alpha: 1),
            ])
            let ctLine = CTLineCreateWithAttributedString(attributed)
            context.textPosition = CGPoint(x: 40, y: height - 60 - index * 52)
            CTLineDraw(ctLine, context)
        }

        guard let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil
              ) else {
            return lines.joined(separator: " ")
        }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
        return lines.joined(separator: " ")
    }
}
#endif
