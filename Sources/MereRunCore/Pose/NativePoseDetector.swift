import Foundation

#if canImport(CoreGraphics) && canImport(ImageIO) && canImport(Vision)
import CoreGraphics
import ImageIO
import Vision
#endif

public enum NativePoseDetectorError: Error, Equatable, LocalizedError, Sendable {
    case imageNotFound(String)
    case imageDecodeFailed(String)
    case unsupportedPlatform

    public var errorDescription: String? {
        switch self {
        case .imageNotFound(let path):
            "Image not found: \(path)"
        case .imageDecodeFailed(let path):
            "Could not decode image: \(path)"
        case .unsupportedPlatform:
            "Native pose detection requires an Apple platform with the Vision framework."
        }
    }
}

public struct NativePosePoint: Codable, Equatable, Sendable {
    public let name: String
    public let x: Double
    public let y: Double
    public let confidence: Double

    public init(name: String, x: Double, y: Double, confidence: Double) {
        self.name = name
        self.x = x
        self.y = y
        self.confidence = confidence
    }
}

public struct NativePoseSubject: Codable, Equatable, Sendable {
    public enum Kind: String, Codable, Equatable, Sendable {
        case body
        case hand
        case face
    }

    public let kind: Kind
    public let index: Int
    public let confidence: Double
    public let points: [NativePosePoint]

    public init(kind: Kind, index: Int, confidence: Double, points: [NativePosePoint]) {
        self.kind = kind
        self.index = index
        self.confidence = confidence
        self.points = points
    }
}

public struct NativePoseResult: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let inputPath: String
    public let imageWidth: Int
    public let imageHeight: Int
    public let coordinateSpace: String
    public let subjects: [NativePoseSubject]

    public init(
        schemaVersion: Int = 1,
        inputPath: String,
        imageWidth: Int,
        imageHeight: Int,
        coordinateSpace: String = "normalized-bottom-left",
        subjects: [NativePoseSubject]
    ) {
        self.schemaVersion = schemaVersion
        self.inputPath = inputPath
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.coordinateSpace = coordinateSpace
        self.subjects = subjects
    }
}

public struct NativePoseRequest: Equatable, Sendable {
    public let includeBody: Bool
    public let includeHands: Bool
    public let includeFace: Bool
    public let maximumHandCount: Int
    public let minimumConfidence: Float

    public init(
        includeBody: Bool = true,
        includeHands: Bool = true,
        includeFace: Bool = true,
        maximumHandCount: Int = 2,
        minimumConfidence: Float = 0.1
    ) {
        self.includeBody = includeBody
        self.includeHands = includeHands
        self.includeFace = includeFace
        self.maximumHandCount = maximumHandCount
        self.minimumConfidence = minimumConfidence
    }
}

public struct NativePoseDetector: Sendable {
    public init() {}

    public func detect(imageURL: URL, request: NativePoseRequest = NativePoseRequest()) throws -> NativePoseResult {
        let standardizedURL = imageURL.standardizedFileURL
        guard FileManager.default.fileExists(atPath: standardizedURL.path) else {
            throw NativePoseDetectorError.imageNotFound(standardizedURL.path)
        }

        #if canImport(CoreGraphics) && canImport(ImageIO) && canImport(Vision)
        guard
            let source = CGImageSourceCreateWithURL(standardizedURL as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw NativePoseDetectorError.imageDecodeFailed(standardizedURL.path)
        }

        var visionRequests: [VNRequest] = []
        let bodyRequest = request.includeBody ? VNDetectHumanBodyPoseRequest() : nil
        let handRequest = request.includeHands ? VNDetectHumanHandPoseRequest() : nil
        let faceRequest = request.includeFace ? VNDetectFaceLandmarksRequest() : nil
        if let bodyRequest { visionRequests.append(bodyRequest) }
        if let handRequest {
            handRequest.maximumHandCount = max(1, request.maximumHandCount)
            visionRequests.append(handRequest)
        }
        if let faceRequest { visionRequests.append(faceRequest) }

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform(visionRequests)

        var subjects: [NativePoseSubject] = []
        if let observations = bodyRequest?.results {
            for (index, observation) in observations.enumerated() {
                let recognized = try observation.recognizedPoints(.all)
                let points = recognized.compactMap { name, point -> NativePosePoint? in
                    guard point.confidence >= request.minimumConfidence else { return nil }
                    return NativePosePoint(
                        name: name.rawValue.rawValue,
                        x: point.location.x,
                        y: point.location.y,
                        confidence: Double(point.confidence)
                    )
                }.sorted { $0.name < $1.name }
                subjects.append(NativePoseSubject(
                    kind: .body,
                    index: index,
                    confidence: Double(observation.confidence),
                    points: points
                ))
            }
        }
        if let observations = handRequest?.results {
            for (index, observation) in observations.enumerated() {
                let recognized = try observation.recognizedPoints(.all)
                let points = recognized.compactMap { name, point -> NativePosePoint? in
                    guard point.confidence >= request.minimumConfidence else { return nil }
                    return NativePosePoint(
                        name: Self.handPointName(name),
                        x: point.location.x,
                        y: point.location.y,
                        confidence: Double(point.confidence)
                    )
                }.sorted { $0.name < $1.name }
                subjects.append(NativePoseSubject(
                    kind: .hand,
                    index: index,
                    confidence: Double(observation.confidence),
                    points: points
                ))
            }
        }
        if let observations = faceRequest?.results {
            for (index, observation) in observations.enumerated() {
                let points = Self.facePoints(observation: observation, minimumConfidence: request.minimumConfidence)
                subjects.append(NativePoseSubject(
                    kind: .face,
                    index: index,
                    confidence: Double(observation.confidence),
                    points: points
                ))
            }
        }

        return NativePoseResult(
            inputPath: standardizedURL.path,
            imageWidth: image.width,
            imageHeight: image.height,
            subjects: subjects
        )
        #else
        throw NativePoseDetectorError.unsupportedPlatform
        #endif
    }

    #if canImport(CoreGraphics) && canImport(ImageIO) && canImport(Vision)
    private static func facePoints(
        observation: VNFaceObservation,
        minimumConfidence: Float
    ) -> [NativePosePoint] {
        guard observation.confidence >= minimumConfidence, let landmarks = observation.landmarks else { return [] }
        let regions: [(String, VNFaceLandmarkRegion2D?)] = [
            ("faceContour", landmarks.faceContour),
            ("leftEye", landmarks.leftEye),
            ("rightEye", landmarks.rightEye),
            ("leftEyebrow", landmarks.leftEyebrow),
            ("rightEyebrow", landmarks.rightEyebrow),
            ("nose", landmarks.nose),
            ("noseCrest", landmarks.noseCrest),
            ("outerLips", landmarks.outerLips),
            ("innerLips", landmarks.innerLips),
            ("medianLine", landmarks.medianLine),
            ("leftPupil", landmarks.leftPupil),
            ("rightPupil", landmarks.rightPupil),
        ]
        return regions.flatMap { name, region -> [NativePosePoint] in
            guard let region else { return [] }
            return region.normalizedPoints.enumerated().map { index, point in
                NativePosePoint(
                    name: "\(name).\(index)",
                    x: observation.boundingBox.origin.x + point.x * observation.boundingBox.width,
                    y: observation.boundingBox.origin.y + point.y * observation.boundingBox.height,
                    confidence: Double(observation.confidence)
                )
            }
        }
    }

    private static func handPointName(_ name: VNHumanHandPoseObservation.JointName) -> String {
        switch name {
        case .wrist: "wrist"
        case .thumbCMC: "thumbCMC"
        case .thumbMP: "thumbMP"
        case .thumbIP: "thumbIP"
        case .thumbTip: "thumbTip"
        case .indexMCP: "indexMCP"
        case .indexPIP: "indexPIP"
        case .indexDIP: "indexDIP"
        case .indexTip: "indexTip"
        case .middleMCP: "middleMCP"
        case .middlePIP: "middlePIP"
        case .middleDIP: "middleDIP"
        case .middleTip: "middleTip"
        case .ringMCP: "ringMCP"
        case .ringPIP: "ringPIP"
        case .ringDIP: "ringDIP"
        case .ringTip: "ringTip"
        case .littleMCP: "littleMCP"
        case .littlePIP: "littlePIP"
        case .littleDIP: "littleDIP"
        case .littleTip: "littleTip"
        default: name.rawValue.rawValue
        }
    }
    #endif
}
