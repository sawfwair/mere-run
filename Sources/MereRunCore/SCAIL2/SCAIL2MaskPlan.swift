import Foundation
import MediaIO

public enum SCAIL2SubjectColor: String, Codable, CaseIterable, Hashable, Sendable {
    case blue
    case red
    case green
    case magenta
    case cyan
    case yellow

    public static let assignmentOrder: [Self] = [.blue, .red, .green, .magenta, .cyan, .yellow]

    public var rgba8: (UInt8, UInt8, UInt8, UInt8) {
        switch self {
        case .blue: (0, 0, 255, 255)
        case .red: (255, 0, 0, 255)
        case .green: (0, 255, 0, 255)
        case .magenta: (255, 0, 255, 255)
        case .cyan: (0, 255, 255, 255)
        case .yellow: (255, 255, 0, 255)
        }
    }
}

public struct SCAIL2MaskPoint: Codable, Hashable, Sendable {
    public let x: Float
    public let y: Float

    public init(x: Float, y: Float) {
        self.x = x
        self.y = y
    }
}

public struct SCAIL2MaskBox: Codable, Hashable, Sendable {
    public let x1: Float
    public let y1: Float
    public let x2: Float
    public let y2: Float

    public init(x1: Float, y1: Float, x2: Float, y2: Float) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    }
}

public struct SCAIL2MaskSelector: Codable, Hashable, Sendable {
    public let text: String?
    public let box: SCAIL2MaskBox?
    public let positivePoints: [SCAIL2MaskPoint]
    public let negativePoints: [SCAIL2MaskPoint]

    enum CodingKeys: String, CodingKey {
        case text
        case box
        case positivePoints = "positive_points"
        case negativePoints = "negative_points"
    }

    public init(
        text: String? = nil,
        box: SCAIL2MaskBox? = nil,
        positivePoints: [SCAIL2MaskPoint] = [],
        negativePoints: [SCAIL2MaskPoint] = []
    ) {
        self.text = text
        self.box = box
        self.positivePoints = positivePoints
        self.negativePoints = negativePoints
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        text = try container.decodeIfPresent(String.self, forKey: .text)
        box = try container.decodeIfPresent(SCAIL2MaskBox.self, forKey: .box)
        positivePoints = try container.decodeIfPresent([SCAIL2MaskPoint].self, forKey: .positivePoints) ?? []
        negativePoints = try container.decodeIfPresent([SCAIL2MaskPoint].self, forKey: .negativePoints) ?? []
    }

    public var isEmpty: Bool {
        text?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false
            && box == nil
            && positivePoints.isEmpty
            && negativePoints.isEmpty
    }

    func promptObject(subjectID: String, maskURL: URL? = nil) -> SAM31PromptObject {
        let points = positivePoints.map {
            SAM31PromptPoint(x: $0.x, y: $0.y, isPositive: true, label: subjectID)
        } + negativePoints.map {
            SAM31PromptPoint(x: $0.x, y: $0.y, isPositive: false, label: subjectID)
        }
        let samBox = box.map {
            SAM31PromptBox(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2, label: subjectID)
        }
        let trimmedText = text?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let maskURL {
            return SAM31PromptObject(
                objectID: subjectID,
                label: subjectID,
                promptKind: .mask,
                boxPrompt: samBox,
                pointPrompts: points,
                maskPrompt: SAM31PromptMask(path: maskURL.path, label: subjectID)
            )
        }
        if samBox != nil || !points.isEmpty {
            return SAM31PromptObject(
                objectID: subjectID,
                label: subjectID,
                promptKind: samBox == nil ? .point : .box,
                boxPrompt: samBox,
                pointPrompts: points
            )
        }
        return SAM31PromptObject(
            objectID: subjectID,
            label: subjectID,
            promptKind: .text,
            textPrompt: trimmedText
        )
    }
}

public struct SCAIL2MaskSubject: Codable, Hashable, Sendable {
    public let id: String
    public let color: SCAIL2SubjectColor
    public let referenceImage: String
    public let referenceSelector: SCAIL2MaskSelector
    public let drivingSelector: SCAIL2MaskSelector

    enum CodingKeys: String, CodingKey {
        case id
        case color
        case referenceImage = "reference_image"
        case referenceSelector = "reference_selector"
        case drivingSelector = "driving_selector"
    }

    public init(
        id: String,
        color: SCAIL2SubjectColor,
        referenceImage: String,
        referenceSelector: SCAIL2MaskSelector,
        drivingSelector: SCAIL2MaskSelector
    ) {
        self.id = id
        self.color = color
        self.referenceImage = referenceImage
        self.referenceSelector = referenceSelector
        self.drivingSelector = drivingSelector
    }
}

public struct SCAIL2MaskCorrection: Codable, Hashable, Sendable {
    public let subjectID: String
    public let frameIndex: Int
    public let box: SCAIL2MaskBox?
    public let positivePoints: [SCAIL2MaskPoint]
    public let negativePoints: [SCAIL2MaskPoint]
    public let paintedBinaryCorrectionPNG: String?

    enum CodingKeys: String, CodingKey {
        case subjectID = "subject_id"
        case frameIndex = "frame_index"
        case box
        case positivePoints = "positive_points"
        case negativePoints = "negative_points"
        case paintedBinaryCorrectionPNG = "painted_binary_correction_png"
    }

    public init(
        subjectID: String,
        frameIndex: Int,
        box: SCAIL2MaskBox? = nil,
        positivePoints: [SCAIL2MaskPoint] = [],
        negativePoints: [SCAIL2MaskPoint] = [],
        paintedBinaryCorrectionPNG: String? = nil
    ) {
        self.subjectID = subjectID
        self.frameIndex = frameIndex
        self.box = box
        self.positivePoints = positivePoints
        self.negativePoints = negativePoints
        self.paintedBinaryCorrectionPNG = paintedBinaryCorrectionPNG
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        subjectID = try container.decode(String.self, forKey: .subjectID)
        frameIndex = try container.decode(Int.self, forKey: .frameIndex)
        box = try container.decodeIfPresent(SCAIL2MaskBox.self, forKey: .box)
        positivePoints = try container.decodeIfPresent([SCAIL2MaskPoint].self, forKey: .positivePoints) ?? []
        negativePoints = try container.decodeIfPresent([SCAIL2MaskPoint].self, forKey: .negativePoints) ?? []
        paintedBinaryCorrectionPNG = try container.decodeIfPresent(
            String.self,
            forKey: .paintedBinaryCorrectionPNG
        )
    }

    var selector: SCAIL2MaskSelector {
        SCAIL2MaskSelector(
            box: box,
            positivePoints: positivePoints,
            negativePoints: negativePoints
        )
    }
}

public struct SCAIL2MaskPlan: Codable, Hashable, Sendable {
    public enum ValidationError: LocalizedError, Equatable, Sendable {
        case unsupportedSchemaVersion(Int)
        case invalidRange
        case invalidResolution(width: Int, height: Int)
        case invalidFPS(Double)
        case invalidSubjectCount(Int)
        case invalidSubjectID(String)
        case duplicateSubjectID(String)
        case duplicateColor(SCAIL2SubjectColor)
        case missingSelector(subjectID: String, kind: String)
        case invalidBox(subjectID: String, kind: String)
        case invalidCorrection(subjectID: String, frameIndex: Int)

        public var errorDescription: String? {
            switch self {
            case .unsupportedSchemaVersion(let version):
                "Unsupported SCAIL-2 mask plan schema_version \(version); expected 1."
            case .invalidRange:
                "The driving in/out range must be finite, nonnegative, and end after start."
            case .invalidResolution(let width, let height):
                "Mask output dimensions must be positive and divisible by 32; received \(width)x\(height)."
            case .invalidFPS(let fps):
                "Mask output FPS must be finite and positive; received \(fps)."
            case .invalidSubjectCount(let count):
                "A SCAIL-2 mask plan requires one to six subjects; received \(count)."
            case .invalidSubjectID(let id):
                "Subject IDs may contain only letters, numbers, hyphens, and underscores; received: \(id)."
            case .duplicateSubjectID(let id):
                "Subject IDs must be unique; duplicate: \(id)."
            case .duplicateColor(let color):
                "Subject palette colors must be unique; duplicate: \(color.rawValue)."
            case .missingSelector(let id, let kind):
                "Subject \(id) requires a non-empty \(kind) selector."
            case .invalidBox(let id, let kind):
                "Subject \(id) has an invalid \(kind) selector box."
            case .invalidCorrection(let id, let frame):
                "Correction for subject \(id) at frame \(frame) is invalid."
            }
        }
    }

    public let schemaVersion: Int
    public let mode: SCAIL2Mode
    public let drivingVideo: String
    public let inSeconds: Double?
    public let outSeconds: Double?
    public let width: Int
    public let height: Int
    public let fps: Double
    public let subjects: [SCAIL2MaskSubject]
    public let corrections: [SCAIL2MaskCorrection]
    public let threshold: Float
    public let resolution: Int
    public let seedFrameSearchLimit: Int
    public let paletteTolerance: UInt8

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case mode
        case drivingVideo = "driving_video"
        case inSeconds = "in_seconds"
        case outSeconds = "out_seconds"
        case width
        case height
        case fps
        case subjects
        case corrections
        case threshold
        case resolution
        case seedFrameSearchLimit = "seed_frame_search_limit"
        case paletteTolerance = "palette_tolerance"
    }

    public init(
        schemaVersion: Int = 1,
        mode: SCAIL2Mode = .animation,
        drivingVideo: String,
        inSeconds: Double? = nil,
        outSeconds: Double? = nil,
        width: Int,
        height: Int,
        fps: Double,
        subjects: [SCAIL2MaskSubject],
        corrections: [SCAIL2MaskCorrection] = [],
        threshold: Float = 0.05,
        resolution: Int = 1008,
        seedFrameSearchLimit: Int = 48,
        paletteTolerance: UInt8 = SCAIL2Palette.codecTolerance
    ) {
        self.schemaVersion = schemaVersion
        self.mode = mode
        self.drivingVideo = drivingVideo
        self.inSeconds = inSeconds
        self.outSeconds = outSeconds
        self.width = width
        self.height = height
        self.fps = fps
        self.subjects = subjects
        self.corrections = corrections
        self.threshold = threshold
        self.resolution = resolution
        self.seedFrameSearchLimit = seedFrameSearchLimit
        self.paletteTolerance = paletteTolerance
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        mode = try container.decodeIfPresent(SCAIL2Mode.self, forKey: .mode) ?? .animation
        drivingVideo = try container.decode(String.self, forKey: .drivingVideo)
        inSeconds = try container.decodeIfPresent(Double.self, forKey: .inSeconds)
        outSeconds = try container.decodeIfPresent(Double.self, forKey: .outSeconds)
        width = try container.decode(Int.self, forKey: .width)
        height = try container.decode(Int.self, forKey: .height)
        fps = try container.decode(Double.self, forKey: .fps)
        subjects = try container.decode([SCAIL2MaskSubject].self, forKey: .subjects)
        corrections = try container.decodeIfPresent([SCAIL2MaskCorrection].self, forKey: .corrections) ?? []
        threshold = try container.decodeIfPresent(Float.self, forKey: .threshold) ?? 0.05
        resolution = try container.decodeIfPresent(Int.self, forKey: .resolution) ?? 1008
        seedFrameSearchLimit = try container.decodeIfPresent(Int.self, forKey: .seedFrameSearchLimit) ?? 48
        paletteTolerance = try container.decodeIfPresent(UInt8.self, forKey: .paletteTolerance)
            ?? SCAIL2Palette.codecTolerance
    }

    public func validate() throws {
        guard schemaVersion == 1 else { throw ValidationError.unsupportedSchemaVersion(schemaVersion) }
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            throw ValidationError.invalidResolution(width: width, height: height)
        }
        guard fps.isFinite, fps > 0 else { throw ValidationError.invalidFPS(fps) }
        if let inSeconds, (!inSeconds.isFinite || inSeconds < 0) {
            throw ValidationError.invalidRange
        }
        if let outSeconds,
           (!outSeconds.isFinite || outSeconds <= 0 || outSeconds <= (inSeconds ?? 0)) {
            throw ValidationError.invalidRange
        }
        guard (1...6).contains(subjects.count) else {
            throw ValidationError.invalidSubjectCount(subjects.count)
        }
        var subjectIDs = Set<String>()
        var colors = Set<SCAIL2SubjectColor>()
        for subject in subjects {
            let id = subject.id.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !id.isEmpty,
                  id.unicodeScalars.allSatisfy({
                      CharacterSet.alphanumerics.contains($0) || $0 == "-" || $0 == "_"
                  }) else {
                throw ValidationError.invalidSubjectID(subject.id)
            }
            guard subjectIDs.insert(id).inserted else {
                throw ValidationError.duplicateSubjectID(subject.id)
            }
            guard colors.insert(subject.color).inserted else {
                throw ValidationError.duplicateColor(subject.color)
            }
            try Self.validate(subject.referenceSelector, subjectID: id, kind: "reference")
            try Self.validate(subject.drivingSelector, subjectID: id, kind: "driving")
        }
        for correction in corrections {
            guard subjectIDs.contains(correction.subjectID),
                  correction.frameIndex >= 0,
                  correction.paintedBinaryCorrectionPNG != nil || !correction.selector.isEmpty else {
                throw ValidationError.invalidCorrection(
                    subjectID: correction.subjectID,
                    frameIndex: correction.frameIndex
                )
            }
            try Self.validate(correction.selector, subjectID: correction.subjectID, kind: "correction", allowEmpty: true)
        }
        guard threshold >= 0, threshold <= 1, resolution > 0, seedFrameSearchLimit >= 0 else {
            throw ValidationError.invalidRange
        }
    }

    public func resolvingPaths(relativeTo planURL: URL) -> Self {
        let baseURL = planURL.deletingLastPathComponent()
        func resolve(_ path: String) -> String {
            let url = path.hasPrefix("/")
                ? URL(fileURLWithPath: path)
                : baseURL.appendingPathComponent(path)
            return url
                .standardizedFileURL.path
        }
        return Self(
            schemaVersion: schemaVersion,
            mode: mode,
            drivingVideo: resolve(drivingVideo),
            inSeconds: inSeconds,
            outSeconds: outSeconds,
            width: width,
            height: height,
            fps: fps,
            subjects: subjects.map {
                SCAIL2MaskSubject(
                    id: $0.id,
                    color: $0.color,
                    referenceImage: resolve($0.referenceImage),
                    referenceSelector: $0.referenceSelector,
                    drivingSelector: $0.drivingSelector
                )
            },
            corrections: corrections.map {
                SCAIL2MaskCorrection(
                    subjectID: $0.subjectID,
                    frameIndex: $0.frameIndex,
                    box: $0.box,
                    positivePoints: $0.positivePoints,
                    negativePoints: $0.negativePoints,
                    paintedBinaryCorrectionPNG: $0.paintedBinaryCorrectionPNG.map(resolve)
                )
            },
            threshold: threshold,
            resolution: resolution,
            seedFrameSearchLimit: seedFrameSearchLimit,
            paletteTolerance: paletteTolerance
        )
    }

    private static func validate(
        _ selector: SCAIL2MaskSelector,
        subjectID: String,
        kind: String,
        allowEmpty: Bool = false
    ) throws {
        if selector.isEmpty {
            if !allowEmpty {
                throw ValidationError.missingSelector(subjectID: subjectID, kind: kind)
            }
            return
        }
        if let box = selector.box, box.x2 <= box.x1 || box.y2 <= box.y1 {
            throw ValidationError.invalidBox(subjectID: subjectID, kind: kind)
        }
    }
}
