import Foundation

public enum FaceExecutionProvider: String, Codable, CaseIterable, Hashable, Sendable {
    case auto
    case coreML = "coreml"
    case cpu
}

public struct FacePoint: Codable, Hashable, Sendable {
    public let x: Double
    public let y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

public struct FaceBoundingBox: Codable, Hashable, Sendable {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    public var area: Double { max(0, width) * max(0, height) }
}

public struct FaceDetection: Codable, Hashable, Sendable {
    public let score: Double
    public let boundingBox: FaceBoundingBox
    public let landmarks: [FacePoint]

    public init(score: Double, boundingBox: FaceBoundingBox, landmarks: [FacePoint]) {
        self.score = score
        self.boundingBox = boundingBox
        self.landmarks = landmarks
    }
}

public struct FaceRecord: Codable, Hashable, Sendable {
    public let index: Int
    public let detection: FaceDetection
    public let embedding: [Float]?

    public init(index: Int, detection: FaceDetection, embedding: [Float]? = nil) {
        self.index = index
        self.detection = detection
        self.embedding = embedding
    }
}

public struct FaceAnalysisResult: Codable, Hashable, Sendable {
    public let image: String
    public let width: Int
    public let height: Int
    public let modelID: String
    public let faces: [FaceRecord]
    public let elapsedMilliseconds: Double

    public init(
        image: String,
        width: Int,
        height: Int,
        modelID: String,
        faces: [FaceRecord],
        elapsedMilliseconds: Double
    ) {
        self.image = image
        self.width = width
        self.height = height
        self.modelID = modelID
        self.faces = faces
        self.elapsedMilliseconds = elapsedMilliseconds
    }
}

public enum FaceAnalysisMath {
    public static func l2Normalized(_ values: [Float]) -> [Float] {
        let squaredNorm = values.reduce(Float.zero) { $0 + ($1 * $1) }
        let norm = sqrt(squaredNorm)
        guard norm > 0 else { return values }
        return values.map { $0 / norm }
    }

    public static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Double? {
        guard !lhs.isEmpty, lhs.count == rhs.count else { return nil }
        var dot = Float.zero
        var lhsNorm = Float.zero
        var rhsNorm = Float.zero
        for index in lhs.indices {
            dot += lhs[index] * rhs[index]
            lhsNorm += lhs[index] * lhs[index]
            rhsNorm += rhs[index] * rhs[index]
        }
        let denominator = sqrt(lhsNorm) * sqrt(rhsNorm)
        guard denominator > 0 else { return nil }
        return Double(dot / denominator)
    }
}

public enum FaceAnalysisError: LocalizedError, Sendable {
    case unavailableRuntime(String)
    case missingModelFiles([URL])
    case invalidModelOutput(String)
    case noFaceDetected(URL)
    case invalidLandmarks
    case runtimeFailure(String)

    public var errorDescription: String? {
        switch self {
        case .unavailableRuntime(let details):
            return details
        case .missingModelFiles(let urls):
            return "Face model is incomplete. Missing or invalid: \(urls.map(\.path).joined(separator: ", "))"
        case .invalidModelOutput(let details):
            return "Invalid face model output: \(details)"
        case .noFaceDetected(let url):
            return "No face detected in \(url.path)."
        case .invalidLandmarks:
            return "Expected five valid facial landmarks for alignment."
        case .runtimeFailure(let details):
            return "Face runtime failed: \(details)"
        }
    }
}
