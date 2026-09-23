import Foundation

public enum SAM31PromptKind: String, Codable, CaseIterable, Hashable, Sendable {
    case text
    case box
    case point
    case mask
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

public struct SAM31PromptMask: Codable, Hashable, Sendable {
    public let path: String
    public let label: String?

    public init(path: String, label: String? = nil) {
        self.path = path
        self.label = label
    }
}

public struct SAM31PromptObject: Codable, Hashable, Sendable {
    public let objectID: String
    public let label: String
    public let promptKind: SAM31PromptKind
    public let textPrompt: String?
    public let boxPrompt: SAM31PromptBox?
    public let pointPrompts: [SAM31PromptPoint]
    public let maskPrompt: SAM31PromptMask?

    public init(
        objectID: String,
        label: String,
        promptKind: SAM31PromptKind,
        textPrompt: String? = nil,
        boxPrompt: SAM31PromptBox? = nil,
        pointPrompts: [SAM31PromptPoint] = [],
        maskPrompt: SAM31PromptMask? = nil
    ) {
        self.objectID = objectID
        self.label = label
        self.promptKind = promptKind
        self.textPrompt = textPrompt
        self.boxPrompt = boxPrompt
        self.pointPrompts = pointPrompts
        self.maskPrompt = maskPrompt
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
    public var maskPrompts: [SAM31PromptMask]
    public var objectPrompts: [SAM31PromptObject]

    public init(
        textPrompts: [String] = [],
        boxPrompts: [SAM31PromptBox] = [],
        pointPrompts: [SAM31PromptPoint] = [],
        maskPrompts: [SAM31PromptMask] = [],
        objectPrompts: [SAM31PromptObject] = []
    ) {
        self.textPrompts = textPrompts
        self.boxPrompts = boxPrompts
        self.pointPrompts = pointPrompts
        self.maskPrompts = maskPrompts
        self.objectPrompts = objectPrompts
    }

    enum CodingKeys: String, CodingKey {
        case textPrompts
        case boxPrompts
        case pointPrompts
        case maskPrompts
        case objectPrompts
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        textPrompts = try container.decodeIfPresent([String].self, forKey: .textPrompts) ?? []
        boxPrompts = try container.decodeIfPresent([SAM31PromptBox].self, forKey: .boxPrompts) ?? []
        pointPrompts = try container.decodeIfPresent([SAM31PromptPoint].self, forKey: .pointPrompts) ?? []
        maskPrompts = try container.decodeIfPresent([SAM31PromptMask].self, forKey: .maskPrompts) ?? []
        objectPrompts = try container.decodeIfPresent([SAM31PromptObject].self, forKey: .objectPrompts) ?? []
    }

    public var isEmpty: Bool {
        textPrompts.isEmpty
            && boxPrompts.isEmpty
            && pointPrompts.isEmpty
            && maskPrompts.isEmpty
            && objectPrompts.isEmpty
    }

    /// The prompted objects, in a fixed order: explicit objects, text prompts, boxes, point-only
    /// groups, masks.
    ///
    /// Boxes and points combine by label, since the interactive decoder takes a box and points for
    /// one object. A labeled point refines the first box with the same label, or, with no such
    /// box, joins the other points of that label as one object. Unlabeled points form one object
    /// together; when exactly one box is unlabeled they refine it instead. Every box is its own
    /// object. Point-only groups keep the order their label first appears in.
    public func normalized(maxObjects: Int = 16) throws -> [SAM31PromptObject] {
        var objects = objectPrompts
        objects.reserveCapacity(
            objectPrompts.count + textPrompts.count + boxPrompts.count + pointPrompts.count + maskPrompts.count
        )

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
        }
        let boxLabels = boxPrompts.map { trimmedLabel($0.label) }
        let unlabeledBoxIndices = boxLabels.indices.filter { boxLabels[$0] == nil }
        var pointsByBoxIndex: [Int: [SAM31PromptPoint]] = [:]
        var pointGroups: [(label: String?, points: [SAM31PromptPoint])] = []
        for point in pointPrompts {
            let label = trimmedLabel(point.label)
            let boxIndex: Int?
            if let label {
                boxIndex = boxLabels.firstIndex(of: label)
            } else {
                boxIndex = unlabeledBoxIndices.count == 1 ? unlabeledBoxIndices[0] : nil
            }
            if let boxIndex {
                pointsByBoxIndex[boxIndex, default: []].append(point)
            } else if let groupIndex = pointGroups.firstIndex(where: { $0.label == label }) {
                pointGroups[groupIndex].points.append(point)
            } else {
                pointGroups.append((label, [point]))
            }
        }

        for (index, box) in boxPrompts.enumerated() {
            objects.append(
                SAM31PromptObject(
                    objectID: "",
                    label: boxLabels[index] ?? "object",
                    promptKind: .box,
                    boxPrompt: box,
                    pointPrompts: pointsByBoxIndex[index] ?? []
                )
            )
        }

        for group in pointGroups {
            objects.append(
                SAM31PromptObject(
                    objectID: "",
                    label: group.label ?? "point-object",
                    promptKind: .point,
                    pointPrompts: group.points
                )
            )
        }

        for mask in maskPrompts {
            let label = trimmedLabel(mask.label) ?? "mask-object"
            objects.append(
                SAM31PromptObject(
                    objectID: "",
                    label: label,
                    promptKind: .mask,
                    maskPrompt: mask
                )
            )
        }

        guard objects.count <= maxObjects else {
            throw ValidationError.tooManyObjects(objects.count, maxSupported: maxObjects)
        }

        var countsByBaseID: [String: Int] = [:]
        return objects.map { object in
            let explicitID = slugify(object.objectID)
            if !object.objectID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return SAM31PromptObject(
                    objectID: explicitID,
                    label: object.label,
                    promptKind: object.promptKind,
                    textPrompt: object.textPrompt,
                    boxPrompt: object.boxPrompt,
                    pointPrompts: object.pointPrompts,
                    maskPrompt: object.maskPrompt
                )
            }
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
                pointPrompts: object.pointPrompts,
                maskPrompt: object.maskPrompt
            )
        }
    }

    /// The label with its surrounding whitespace removed, or nil when nothing is left.
    private func trimmedLabel(_ raw: String?) -> String? {
        let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
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
