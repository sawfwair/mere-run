import Foundation

public enum SAM31PromptKind: String, Codable, CaseIterable, Hashable, Sendable {
    case text
    case box
    case point
}

public struct SAM31PromptPoint: Codable, Hashable, Sendable {
    public let x: Float
    public let y: Float
    public let isPositive: Bool
    public let label: String?

    public init(x: Float, y: Float, isPositive: Bool, label: String? = nil) {
        self.x = x
        self.y = y
        self.isPositive = isPositive
        self.label = label
    }
}

public struct SAM31PromptBox: Codable, Hashable, Sendable {
    public let x1: Float
    public let y1: Float
    public let x2: Float
    public let y2: Float
    public let label: String?

    public init(x1: Float, y1: Float, x2: Float, y2: Float, label: String? = nil) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label
    }

    public var segmentationBox: SAM31SegmentationBox {
        SAM31SegmentationBox(x1: x1, y1: y1, x2: x2, y2: y2)
    }
}

public struct SAM31PromptObject: Codable, Hashable, Sendable {
    public let objectID: String
    public let label: String
    public let promptKind: SAM31PromptKind
    public let textPrompt: String?
    public let boxPrompt: SAM31PromptBox?
    public let pointPrompts: [SAM31PromptPoint]

    public init(
        objectID: String,
        label: String,
        promptKind: SAM31PromptKind,
        textPrompt: String? = nil,
        boxPrompt: SAM31PromptBox? = nil,
        pointPrompts: [SAM31PromptPoint] = []
    ) {
        self.objectID = objectID
        self.label = label
        self.promptKind = promptKind
        self.textPrompt = textPrompt
        self.boxPrompt = boxPrompt
        self.pointPrompts = pointPrompts
    }
}

public struct SAM31PromptSet: Codable, Hashable, Sendable {
    public enum ValidationError: LocalizedError, Sendable {
        case tooManyObjects(Int, maxSupported: Int)
        case invalidBox(SAM31PromptBox)

        public var errorDescription: String? {
            switch self {
            case .tooManyObjects(let count, let maxSupported):
                return "SAM 3.1 supports up to \(maxSupported) prompted objects per run. Received \(count)."
            case .invalidBox(let box):
                return "Invalid box prompt: (\(box.x1), \(box.y1), \(box.x2), \(box.y2))."
            }
        }
    }

    public var textPrompts: [String]
    public var boxPrompts: [SAM31PromptBox]
    public var pointPrompts: [SAM31PromptPoint]

    public init(
        textPrompts: [String] = [],
        boxPrompts: [SAM31PromptBox] = [],
        pointPrompts: [SAM31PromptPoint] = []
    ) {
        self.textPrompts = textPrompts
        self.boxPrompts = boxPrompts
        self.pointPrompts = pointPrompts
    }

    public var isEmpty: Bool {
        textPrompts.isEmpty && boxPrompts.isEmpty && pointPrompts.isEmpty
    }

    public func normalized(maxObjects: Int = 16) throws -> [SAM31PromptObject] {
        var objects: [SAM31PromptObject] = []
        objects.reserveCapacity(textPrompts.count + boxPrompts.count + pointPrompts.count)

        for prompt in textPrompts {
            let label = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !label.isEmpty else { continue }
            objects.append(
                SAM31PromptObject(
                    objectID: "",
                    label: label,
                    promptKind: .text,
                    textPrompt: label
                )
            )
        }

        for box in boxPrompts {
            guard box.x2 > box.x1, box.y2 > box.y1 else {
                throw ValidationError.invalidBox(box)
            }
            let label = normalizedLabel(box.label, fallback: "object")
            objects.append(
                SAM31PromptObject(
                    objectID: "",
                    label: label,
                    promptKind: .box,
                    boxPrompt: box
                )
            )
        }

        let groupedPoints = Dictionary(grouping: pointPrompts) { point in
            let label = point.label?.trimmingCharacters(in: .whitespacesAndNewlines)
            return (label?.isEmpty == false) ? label! : UUID().uuidString
        }
        let sortedKeys = groupedPoints.keys.sorted()
        for key in sortedKeys {
            guard let points = groupedPoints[key], !points.isEmpty else { continue }
            let label = points[0].label.flatMap { normalizedLabel($0, fallback: nil) } ?? "point-object"
            objects.append(
                SAM31PromptObject(
                    objectID: "",
                    label: label,
                    promptKind: .point,
                    pointPrompts: points
                )
            )
        }

        guard objects.count <= maxObjects else {
            throw ValidationError.tooManyObjects(objects.count, maxSupported: maxObjects)
        }

        var countsByBaseID: [String: Int] = [:]
        return objects.map { object in
            let base = slugify(object.label.isEmpty ? "object" : object.label)
            let count = countsByBaseID[base, default: 0] + 1
            countsByBaseID[base] = count
            let objectID = count == 1 ? base : "\(base)-\(count)"
            return SAM31PromptObject(
                objectID: objectID,
                label: object.label,
                promptKind: object.promptKind,
                textPrompt: object.textPrompt,
                boxPrompt: object.boxPrompt,
                pointPrompts: object.pointPrompts
            )
        }
    }

    private func normalizedLabel(_ raw: String?, fallback: String?) -> String {
        let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !trimmed.isEmpty {
            return trimmed
        }
        return fallback ?? "object"
    }

    private func slugify(_ value: String) -> String {
        let lowered = value.lowercased()
        let parts = lowered.split { !$0.isLetter && !$0.isNumber }
        let joined = parts.joined(separator: "-")
        return joined.isEmpty ? "object" : joined
    }
}

public struct SAM31MaskArtifact: Codable, Hashable, Sendable {
    public let objectID: String
    public let candidateIndex: Int?
    public let path: String
    public let maskAreaPixels: Int

    public init(objectID: String, candidateIndex: Int? = nil, path: String, maskAreaPixels: Int) {
        self.objectID = objectID
        self.candidateIndex = candidateIndex
        self.path = path
        self.maskAreaPixels = maskAreaPixels
    }
}

public struct SAM31TrackedObject: Codable, Hashable, Sendable {
    public let objectID: String
    public let label: String
    public let promptKind: SAM31PromptKind
    public let seedFrameIndex: Int
    public let textPrompt: String?
    public let seedBox: SAM31SegmentationBox?
    public let seedPoints: [SAM31PromptPoint]

    public init(
        objectID: String,
        label: String,
        promptKind: SAM31PromptKind,
        seedFrameIndex: Int,
        textPrompt: String? = nil,
        seedBox: SAM31SegmentationBox? = nil,
        seedPoints: [SAM31PromptPoint] = []
    ) {
        self.objectID = objectID
        self.label = label
        self.promptKind = promptKind
        self.seedFrameIndex = seedFrameIndex
        self.textPrompt = textPrompt
        self.seedBox = seedBox
        self.seedPoints = seedPoints
    }
}

public struct SAM31TrackingObjectResult: Codable, Hashable, Sendable {
    public let objectID: String
    public let label: String
    public let score: Float
    public let visible: Bool
    public let box: SAM31SegmentationBox
    public let maskAreaPixels: Int
    public let maskPath: String?

    public init(
        objectID: String,
        label: String,
        score: Float,
        visible: Bool,
        box: SAM31SegmentationBox,
        maskAreaPixels: Int,
        maskPath: String? = nil
    ) {
        self.objectID = objectID
        self.label = label
        self.score = score
        self.visible = visible
        self.box = box
        self.maskAreaPixels = maskAreaPixels
        self.maskPath = maskPath
    }
}

public struct SAM31TrackingFrameResult: Codable, Hashable, Sendable {
    public let frameIndex: Int
    public let timestampSeconds: Double
    public let detections: [SAM31TrackingObjectResult]

    public init(frameIndex: Int, timestampSeconds: Double, detections: [SAM31TrackingObjectResult]) {
        self.frameIndex = frameIndex
        self.timestampSeconds = timestampSeconds
        self.detections = detections
    }
}

public struct SAM31TrackingRun: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let modelID: String
    public let inputVideoPath: String
    public let annotatedVideoPath: String
    public let jsonOutputPath: String?
    public let fps: Double
    public let frameWidth: Int
    public let frameHeight: Int
    public let initFrameIndex: Int
    public let droppedFrameCount: Int
    public let objects: [SAM31TrackedObject]
    public let frames: [SAM31TrackingFrameResult]

    public init(
        schemaVersion: Int = 1,
        modelID: String,
        inputVideoPath: String,
        annotatedVideoPath: String,
        jsonOutputPath: String? = nil,
        fps: Double,
        frameWidth: Int,
        frameHeight: Int,
        initFrameIndex: Int,
        droppedFrameCount: Int = 0,
        objects: [SAM31TrackedObject],
        frames: [SAM31TrackingFrameResult]
    ) {
        self.schemaVersion = schemaVersion
        self.modelID = modelID
        self.inputVideoPath = inputVideoPath
        self.annotatedVideoPath = annotatedVideoPath
        self.jsonOutputPath = jsonOutputPath
        self.fps = fps
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.initFrameIndex = initFrameIndex
        self.droppedFrameCount = droppedFrameCount
        self.objects = objects
        self.frames = frames
    }
}

public final class SAM31TrackingSession: @unchecked Sendable {
    public let modelID: String
    public let objects: [SAM31TrackedObject]

    public init(modelID: String, objects: [SAM31TrackedObject]) {
        self.modelID = modelID
        self.objects = objects
    }
}
