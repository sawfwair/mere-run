import Foundation
import MediaIO

public final class FaceAnalyzer {
    public struct Options: Hashable, Sendable {
        public var scoreThreshold: Float
        public var nmsThreshold: Double
        public var maxFaces: Int?
        public var includeEmbeddings: Bool

        public init(
            scoreThreshold: Float = 0.65,
            nmsThreshold: Double = 0.4,
            maxFaces: Int? = nil,
            includeEmbeddings: Bool = false
        ) {
            self.scoreThreshold = scoreThreshold
            self.nmsThreshold = nmsThreshold
            self.maxFaces = maxFaces
            self.includeEmbeddings = includeEmbeddings
        }
    }

    private let detector: FaceDetector
    private let embedder: FaceEmbedder?

    public init(
        modelRootURL: URL,
        includeRecognizer: Bool = true,
        executionProvider: FaceExecutionProvider = .auto,
        fileManager: FileManager = .default
    ) throws {
        let resources = FaceAnalysisResources(rootURL: modelRootURL)
        let missing = resources.validate(fileManager: fileManager)
        guard missing.isEmpty else { throw FaceAnalysisError.missingModelFiles(missing) }
        detector = try FaceDetector(modelURL: resources.detectorURL, executionProvider: executionProvider)
        embedder = includeRecognizer
            ? try FaceEmbedder(modelURL: resources.recognizerURL, executionProvider: executionProvider)
            : nil
    }

    public func analyze(imageURL: URL, options: Options = Options()) throws -> FaceAnalysisResult {
        let started = ContinuousClock.now
        let image = try MediaImageIO.decode(imageURL)
        let detections = try detector.detect(
            image: image,
            scoreThreshold: options.scoreThreshold,
            nmsThreshold: options.nmsThreshold,
            maxFaces: options.maxFaces
        )
        let faces = try detections.enumerated().map { index, detection in
            FaceRecord(
                index: index,
                detection: detection,
                embedding: options.includeEmbeddings ? try requireEmbedder().embed(
                    image: image,
                    landmarks: detection.landmarks
                ) : nil
            )
        }
        let duration = started.duration(to: .now)
        return FaceAnalysisResult(
            image: imageURL.standardizedFileURL.path,
            width: image.width,
            height: image.height,
            modelID: FaceAnalysisResources.modelID,
            faces: faces,
            elapsedMilliseconds: duration.seconds * 1_000
        )
    }

    public func embedding(
        imageURL: URL,
        scoreThreshold: Float = 0.65,
        selection: Int? = nil
    ) throws -> FaceRecord {
        let result = try analyze(
            imageURL: imageURL,
            options: Options(
                scoreThreshold: scoreThreshold,
                includeEmbeddings: true
            )
        )
        guard !result.faces.isEmpty else { throw FaceAnalysisError.noFaceDetected(imageURL) }
        if let selection {
            guard result.faces.indices.contains(selection) else {
                throw FaceAnalysisError.invalidModelOutput(
                    "face index \(selection) is outside 0..<\(result.faces.count)"
                )
            }
            return result.faces[selection]
        }
        return result.faces.max { $0.detection.boundingBox.area < $1.detection.boundingBox.area }!
    }

    private func requireEmbedder() throws -> FaceEmbedder {
        guard let embedder else {
            throw FaceAnalysisError.runtimeFailure("recognizer session was not loaded")
        }
        return embedder
    }
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + (Double(components.attoseconds) / 1e18)
    }
}
