import ArgumentParser
import Foundation
import MereRunCore

struct VisionFace: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "face",
        abstract: "Detect faces, create identity embeddings, and compare faces locally.",
        discussion: """
        Uses the managed Buffalo-L detector and ArcFace recognizer. Raw inference belongs here;
        library indexing, search, clustering, and review workflows live in mere-run-plugins.
        """,
        subcommands: [
            VisionFaceDetect.self,
            VisionFaceEmbed.self,
            VisionFaceCompare.self,
            VisionFaceBatch.self,
        ]
    )
}

struct FaceModelOptions: ParsableArguments {
    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Buffalo-L model root.")
    var model: String?

    @Option(name: [.customLong("score-threshold")], help: "Minimum face detector score in [0, 1].")
    var scoreThreshold = 0.65

    @Option(name: [.customLong("execution-provider")], help: "Execution provider: auto, coreml, or cpu.")
    var executionProvider = "auto"

    func validate() throws {
        guard (0...1).contains(scoreThreshold) else {
            throw ValidationError("--score-threshold must be between 0 and 1.")
        }
        guard FaceExecutionProvider(rawValue: executionProvider.lowercased()) != nil else {
            throw ValidationError("--execution-provider must be auto, coreml, or cpu.")
        }
    }

    var resolvedExecutionProvider: FaceExecutionProvider {
        FaceExecutionProvider(rawValue: executionProvider.lowercased()) ?? .auto
    }
}

struct VisionFaceDetect: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "detect",
        abstract: "Detect faces and five-point landmarks in an image."
    )

    @Argument(help: "Image file path.")
    var image: String

    @OptionGroup var modelOptions: FaceModelOptions

    @Option(name: [.customLong("max-faces")], help: "Maximum detections to return after NMS.")
    var maxFaces: Int?

    @Flag(name: [.customLong("include-embeddings")], help: "Also compute normalized 512-dimensional ArcFace embeddings.")
    var includeEmbeddings = false

    @Option(name: [.customLong("json-output")], help: "Optional JSON output file.")
    var jsonOutput: String?

    @Flag(name: [.long], help: "Print JSON on stdout.")
    var json = false

    func validate() throws {
        try modelOptions.validate()
        if let maxFaces, maxFaces <= 0 {
            throw ValidationError("--max-faces must be greater than zero.")
        }
    }

    func run() throws {
        let imageURL = try FaceCLI.resolveImage(image)
        let modelRoot = try FaceCLI.resolveModelRoot(modelOptions.model)
        let analyzer = try FaceAnalyzer(
            modelRootURL: modelRoot,
            includeRecognizer: includeEmbeddings,
            executionProvider: modelOptions.resolvedExecutionProvider
        )
        let result = try analyzer.analyze(
            imageURL: imageURL,
            options: .init(
                scoreThreshold: Float(modelOptions.scoreThreshold),
                maxFaces: maxFaces,
                includeEmbeddings: includeEmbeddings
            )
        )
        try FaceCLI.emit(result, jsonOutput: jsonOutput, printJSON: json) {
            print("Faces: \(result.faces.count)")
            if let jsonOutput { print("JSON: \(URL(fileURLWithPath: jsonOutput).standardizedFileURL.path)") }
        }
    }
}

struct VisionFaceEmbed: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Create a normalized ArcFace embedding for one face in an image."
    )

    @Argument(help: "Image file path.")
    var image: String

    @OptionGroup var modelOptions: FaceModelOptions

    @Option(name: [.customLong("face-index")], help: "Zero-based detected face index. Defaults to the largest face.")
    var faceIndex: Int?

    @Option(name: [.customLong("json-output")], help: "Optional JSON output file.")
    var jsonOutput: String?

    @Flag(name: [.long], help: "Print JSON on stdout.")
    var json = false

    func validate() throws {
        try modelOptions.validate()
        if let faceIndex, faceIndex < 0 {
            throw ValidationError("--face-index must be zero or greater.")
        }
    }

    func run() throws {
        let imageURL = try FaceCLI.resolveImage(image)
        let modelRoot = try FaceCLI.resolveModelRoot(modelOptions.model)
        let analyzer = try FaceAnalyzer(
            modelRootURL: modelRoot,
            executionProvider: modelOptions.resolvedExecutionProvider
        )
        let face = try analyzer.embedding(
            imageURL: imageURL,
            scoreThreshold: Float(modelOptions.scoreThreshold),
            selection: faceIndex
        )
        let result = FaceEmbeddingOutput(
            image: imageURL.path,
            modelID: FaceAnalysisResources.modelID,
            face: face
        )
        try FaceCLI.emit(result, jsonOutput: jsonOutput, printJSON: json) {
            print("Face: \(face.index)")
            print("Dimensions: \(face.embedding?.count ?? 0)")
            if let jsonOutput { print("JSON: \(URL(fileURLWithPath: jsonOutput).standardizedFileURL.path)") }
        }
    }
}

struct VisionFaceCompare: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "compare",
        abstract: "Compare one face from each of two images with cosine similarity."
    )

    @Argument(help: "Reference image file path.")
    var reference: String

    @Argument(help: "Candidate image file path.")
    var candidate: String

    @OptionGroup var modelOptions: FaceModelOptions

    @Option(name: [.customLong("reference-face-index")], help: "Reference face index. Defaults to largest face.")
    var referenceFaceIndex: Int?

    @Option(name: [.customLong("candidate-face-index")], help: "Candidate face index. Defaults to largest face.")
    var candidateFaceIndex: Int?

    @Option(name: [.customLong("json-output")], help: "Optional JSON output file.")
    var jsonOutput: String?

    @Flag(name: [.long], help: "Print JSON on stdout.")
    var json = false

    func validate() throws {
        try modelOptions.validate()
        if let referenceFaceIndex, referenceFaceIndex < 0 {
            throw ValidationError("--reference-face-index must be zero or greater.")
        }
        if let candidateFaceIndex, candidateFaceIndex < 0 {
            throw ValidationError("--candidate-face-index must be zero or greater.")
        }
    }

    func run() throws {
        let referenceURL = try FaceCLI.resolveImage(reference)
        let candidateURL = try FaceCLI.resolveImage(candidate)
        let analyzer = try FaceAnalyzer(
            modelRootURL: FaceCLI.resolveModelRoot(modelOptions.model),
            executionProvider: modelOptions.resolvedExecutionProvider
        )
        let referenceFace = try analyzer.embedding(
            imageURL: referenceURL,
            scoreThreshold: Float(modelOptions.scoreThreshold),
            selection: referenceFaceIndex
        )
        let candidateFace = try analyzer.embedding(
            imageURL: candidateURL,
            scoreThreshold: Float(modelOptions.scoreThreshold),
            selection: candidateFaceIndex
        )
        guard let lhs = referenceFace.embedding,
              let rhs = candidateFace.embedding,
              let similarity = FaceAnalysisMath.cosineSimilarity(lhs, rhs) else {
            throw FaceAnalysisError.invalidModelOutput("could not compare face embeddings")
        }
        let result = FaceComparisonOutput(
            modelID: FaceAnalysisResources.modelID,
            referenceImage: referenceURL.path,
            referenceFaceIndex: referenceFace.index,
            candidateImage: candidateURL.path,
            candidateFaceIndex: candidateFace.index,
            cosineSimilarity: similarity
        )
        try FaceCLI.emit(result, jsonOutput: jsonOutput, printJSON: json) {
            print(String(format: "%.6f", similarity))
        }
    }
}

struct VisionFaceBatch: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "batch",
        abstract: "Analyze many images with one warm detector/recognizer session and emit JSONL."
    )

    @Argument(help: "One or more image file paths.")
    var images: [String] = []

    @Option(name: [.customLong("input-list")], help: "UTF-8 file containing one image path per line.")
    var inputList: String?

    @OptionGroup var modelOptions: FaceModelOptions

    @Option(name: [.customLong("max-faces")], help: "Maximum detections per image after NMS.")
    var maxFaces: Int?

    @Flag(name: [.customLong("include-embeddings")], help: "Include normalized 512-dimensional embeddings.")
    var includeEmbeddings = false

    @Option(name: [.customLong("jsonl-output")], help: "Optional JSONL output file. Defaults to stdout.")
    var jsonlOutput: String?

    @Flag(name: [.customLong("fail-fast")], help: "Stop at the first unreadable or invalid image.")
    var failFast = false

    func validate() throws {
        try modelOptions.validate()
        guard !images.isEmpty || inputList != nil else {
            throw ValidationError("Provide image paths or --input-list PATH.")
        }
        if let maxFaces, maxFaces <= 0 {
            throw ValidationError("--max-faces must be greater than zero.")
        }
    }

    func run() throws {
        let paths = try resolvePaths()
        guard !paths.isEmpty else {
            throw ValidationError("No non-empty image paths were supplied.")
        }
        let analyzer = try FaceAnalyzer(
            modelRootURL: FaceCLI.resolveModelRoot(modelOptions.model),
            includeRecognizer: includeEmbeddings,
            executionProvider: modelOptions.resolvedExecutionProvider
        )
        let writer = try FaceJSONLWriter(outputPath: jsonlOutput)
        for path in paths {
            let url = URL(fileURLWithPath: path).standardizedFileURL
            do {
                let result = try analyzer.analyze(
                    imageURL: url,
                    options: .init(
                        scoreThreshold: Float(modelOptions.scoreThreshold),
                        maxFaces: maxFaces,
                        includeEmbeddings: includeEmbeddings
                    )
                )
                try writer.write(FaceBatchOutput(ok: true, image: url.path, result: result, error: nil))
            } catch {
                try writer.write(FaceBatchOutput(
                    ok: false,
                    image: url.path,
                    result: nil,
                    error: error.localizedDescription
                ))
                if failFast { throw error }
            }
        }
    }

    private func resolvePaths() throws -> [String] {
        var paths = images
        if let inputList {
            let listURL = URL(fileURLWithPath: inputList).standardizedFileURL
            let contents = try String(contentsOf: listURL, encoding: .utf8)
            paths.append(contentsOf: contents.split(whereSeparator: { $0.isNewline }).map(String.init))
        }
        return paths.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }
}

private struct FaceEmbeddingOutput: Codable {
    let image: String
    let modelID: String
    let face: FaceRecord
}

private struct FaceComparisonOutput: Codable {
    let modelID: String
    let referenceImage: String
    let referenceFaceIndex: Int
    let candidateImage: String
    let candidateFaceIndex: Int
    let cosineSimilarity: Double
}

private struct FaceBatchOutput: Codable {
    let ok: Bool
    let image: String
    let result: FaceAnalysisResult?
    let error: String?
}

private final class FaceJSONLWriter {
    private let encoder = JSONEncoder()
    private let handle: FileHandle?

    init(outputPath: String?) throws {
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let outputPath else {
            handle = nil
            return
        }
        let url = URL(fileURLWithPath: outputPath).standardizedFileURL
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let outputHandle = try FileHandle(forWritingTo: url)
        try outputHandle.truncate(atOffset: 0)
        handle = outputHandle
    }

    deinit { try? handle?.close() }

    func write<T: Encodable>(_ value: T) throws {
        var data = try encoder.encode(value)
        data.append(0x0A)
        if let handle {
            try handle.write(contentsOf: data)
            try handle.synchronize()
        } else {
            FileHandle.standardOutput.write(data)
        }
    }
}

private enum FaceCLI {
    static func resolveImage(_ path: String) throws -> URL {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("Image not found: \(url.path)")
        }
        return url
    }

    static func resolveModelRoot(_ rawModel: String?) throws -> URL {
        if let rawModel, !rawModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let path = URL(fileURLWithPath: rawModel).standardizedFileURL
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(atPath: path.path, isDirectory: &isDirectory) {
                guard isDirectory.boolValue else {
                    throw ValidationError("Model path must be a directory: \(path.path)")
                }
                return path
            }
            guard rawModel == FaceAnalysisResources.modelID else {
                throw ValidationError(
                    "Unknown face model \(rawModel). Pass a local model root or \(FaceAnalysisResources.modelID)."
                )
            }
        }
        do {
            return try ModelResolver().resolve(.visionFaceBuffaloL).rootURL
        } catch {
            throw ValidationError(
                "Model \(FaceAnalysisResources.modelID) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: FaceAnalysisResources.modelID))` or pass --model PATH."
            )
        }
    }

    static func emit<T: Encodable>(
        _ value: T,
        jsonOutput: String?,
        printJSON: Bool,
        humanOutput: () -> Void
    ) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(value)
        if let jsonOutput {
            let url = URL(fileURLWithPath: jsonOutput).standardizedFileURL
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: url, options: .atomic)
        }
        if printJSON {
            print(String(decoding: data, as: UTF8.self))
        } else {
            humanOutput()
        }
    }
}
