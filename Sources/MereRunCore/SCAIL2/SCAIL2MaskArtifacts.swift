import Foundation

public struct SCAIL2MaskArtifact: Codable, Hashable, Sendable {
    public let kind: String
    public let path: String
    public let sha256: String
    public let byteCount: Int64

    enum CodingKeys: String, CodingKey {
        case kind
        case path
        case sha256
        case byteCount = "byte_count"
    }

    public init(kind: String, path: String, sha256: String, byteCount: Int64) {
        self.kind = kind
        self.path = path
        self.sha256 = sha256
        self.byteCount = byteCount
    }
}

public struct SCAIL2MaskSubjectManifest: Codable, Hashable, Sendable {
    public let id: String
    public let color: SCAIL2SubjectColor
    public let referenceImagePath: String
    public let preparedReferenceImagePath: String
    public let referenceMaskPath: String
    public let seedFrameIndex: Int?
    public let gapRanges: [ClosedRange<Int>]

    enum CodingKeys: String, CodingKey {
        case id
        case color
        case referenceImagePath = "reference_image_path"
        case preparedReferenceImagePath = "prepared_reference_image_path"
        case referenceMaskPath = "reference_mask_path"
        case seedFrameIndex = "seed_frame_index"
        case gapRanges = "gap_ranges"
    }

    public init(
        id: String,
        color: SCAIL2SubjectColor,
        referenceImagePath: String,
        preparedReferenceImagePath: String,
        referenceMaskPath: String,
        seedFrameIndex: Int?,
        gapRanges: [ClosedRange<Int>]
    ) {
        self.id = id
        self.color = color
        self.referenceImagePath = referenceImagePath
        self.preparedReferenceImagePath = preparedReferenceImagePath
        self.referenceMaskPath = referenceMaskPath
        self.seedFrameIndex = seedFrameIndex
        self.gapRanges = gapRanges
    }
}

public struct SCAIL2MaskQualityWarning: Codable, Hashable, Sendable {
    public let code: String
    public let subjectID: String?
    public let frameIndex: Int?
    public let range: ClosedRange<Int>?
    public let message: String

    enum CodingKeys: String, CodingKey {
        case code
        case subjectID = "subject_id"
        case frameIndex = "frame_index"
        case range
        case message
    }

    public init(
        code: String,
        subjectID: String? = nil,
        frameIndex: Int? = nil,
        range: ClosedRange<Int>? = nil,
        message: String
    ) {
        self.code = code
        self.subjectID = subjectID
        self.frameIndex = frameIndex
        self.range = range
        self.message = message
    }
}

public struct SCAIL2MaskQualityReport: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let blockingErrors: [String]
    public let warnings: [SCAIL2MaskQualityWarning]
    public let overlapPixelCount: Int
    public let paletteRoundTripValidated: Bool

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case blockingErrors = "blocking_errors"
        case warnings
        case overlapPixelCount = "overlap_pixel_count"
        case paletteRoundTripValidated = "palette_round_trip_validated"
    }

    public init(
        schemaVersion: Int = 1,
        blockingErrors: [String],
        warnings: [SCAIL2MaskQualityWarning],
        overlapPixelCount: Int,
        paletteRoundTripValidated: Bool
    ) {
        self.schemaVersion = schemaVersion
        self.blockingErrors = blockingErrors
        self.warnings = warnings
        self.overlapPixelCount = overlapPixelCount
        self.paletteRoundTripValidated = paletteRoundTripValidated
    }
}

public struct SCAIL2MaskCorrectionRecord: Codable, Hashable, Sendable {
    public let subjectID: String
    public let frameIndex: Int
    public let paintedMaskPath: String?
    public let positivePointCount: Int
    public let negativePointCount: Int
    public let hasBox: Bool

    enum CodingKeys: String, CodingKey {
        case subjectID = "subject_id"
        case frameIndex = "frame_index"
        case paintedMaskPath = "painted_mask_path"
        case positivePointCount = "positive_point_count"
        case negativePointCount = "negative_point_count"
        case hasBox = "has_box"
    }

    public init(correction: SCAIL2MaskCorrection) {
        subjectID = correction.subjectID
        frameIndex = correction.frameIndex
        paintedMaskPath = correction.paintedBinaryCorrectionPNG
        positivePointCount = correction.positivePoints.count
        negativePointCount = correction.negativePoints.count
        hasBox = correction.box != nil
    }
}

public struct SCAIL2MaskManifest: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let status: String
    public let previewFrame: Int?
    public let modelID: String
    public let modelRevision: String
    public let drivingSourcePath: String
    public let drivingProxyPath: String?
    public let drivingMaskPath: String?
    public let overlayPreviewPath: String
    public let contactSheetPath: String
    public let trackingPath: String?
    public let qualityPath: String
    public let frameCount: Int
    public let fps: Double
    public let width: Int
    public let height: Int
    public let subjects: [SCAIL2MaskSubjectManifest]
    public let corrections: [SCAIL2MaskCorrectionRecord]
    public let artifacts: [SCAIL2MaskArtifact]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case status
        case previewFrame = "preview_frame"
        case modelID = "model_id"
        case modelRevision = "model_revision"
        case drivingSourcePath = "driving_source_path"
        case drivingProxyPath = "driving_proxy_path"
        case drivingMaskPath = "driving_mask_path"
        case overlayPreviewPath = "overlay_preview_path"
        case contactSheetPath = "contact_sheet_path"
        case trackingPath = "tracking_path"
        case qualityPath = "quality_path"
        case frameCount = "frame_count"
        case fps
        case width
        case height
        case subjects
        case corrections
        case artifacts
    }

    public init(
        status: String,
        previewFrame: Int?,
        modelID: String,
        modelRevision: String,
        drivingSourcePath: String,
        drivingProxyPath: String?,
        drivingMaskPath: String?,
        overlayPreviewPath: String,
        contactSheetPath: String,
        trackingPath: String?,
        qualityPath: String,
        frameCount: Int,
        fps: Double,
        width: Int,
        height: Int,
        subjects: [SCAIL2MaskSubjectManifest],
        corrections: [SCAIL2MaskCorrectionRecord],
        artifacts: [SCAIL2MaskArtifact]
    ) {
        schemaVersion = 1
        self.status = status
        self.previewFrame = previewFrame
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.drivingSourcePath = drivingSourcePath
        self.drivingProxyPath = drivingProxyPath
        self.drivingMaskPath = drivingMaskPath
        self.overlayPreviewPath = overlayPreviewPath
        self.contactSheetPath = contactSheetPath
        self.trackingPath = trackingPath
        self.qualityPath = qualityPath
        self.frameCount = frameCount
        self.fps = fps
        self.width = width
        self.height = height
        self.subjects = subjects
        self.corrections = corrections
        self.artifacts = artifacts
    }
}

public struct SCAIL2MaskPreparationResult: Codable, Hashable, Sendable {
    public let manifestPath: String
    public let status: String
    public let preview: Bool
    public let frameCount: Int
    public let warnings: Int

    enum CodingKeys: String, CodingKey {
        case manifestPath = "manifest_path"
        case status
        case preview
        case frameCount = "frame_count"
        case warnings
    }

    public init(
        manifestPath: String,
        status: String,
        preview: Bool,
        frameCount: Int,
        warnings: Int
    ) {
        self.manifestPath = manifestPath
        self.status = status
        self.preview = preview
        self.frameCount = frameCount
        self.warnings = warnings
    }
}

public struct SCAIL2MaskTrackingSubject: Codable, Hashable, Sendable {
    public let id: String
    public let color: SCAIL2SubjectColor
    public let seedFrameIndex: Int?
    public let frames: [SAM31TrackingFrameResult]

    enum CodingKeys: String, CodingKey {
        case id
        case color
        case seedFrameIndex = "seed_frame_index"
        case frames
    }

    public init(
        id: String,
        color: SCAIL2SubjectColor,
        seedFrameIndex: Int?,
        frames: [SAM31TrackingFrameResult]
    ) {
        self.id = id
        self.color = color
        self.seedFrameIndex = seedFrameIndex
        self.frames = frames.map { frame in
            SAM31TrackingFrameResult(
                frameIndex: frame.frameIndex,
                timestampSeconds: frame.timestampSeconds,
                detections: frame.detections.map { detection in
                    SAM31TrackingObjectResult(
                        objectID: detection.objectID,
                        label: detection.label,
                        score: detection.score,
                        visible: detection.visible,
                        box: detection.box,
                        maskAreaPixels: detection.maskAreaPixels,
                        maskPath: nil
                    )
                }
            )
        }
    }
}

public struct SCAIL2MaskTrackingReport: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let modelID: String
    public let frameCount: Int
    public let fps: Double
    public let subjects: [SCAIL2MaskTrackingSubject]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelID = "model_id"
        case frameCount = "frame_count"
        case fps
        case subjects
    }

    public init(
        modelID: String,
        frameCount: Int,
        fps: Double,
        subjects: [SCAIL2MaskTrackingSubject]
    ) {
        schemaVersion = 1
        self.modelID = modelID
        self.frameCount = frameCount
        self.fps = fps
        self.subjects = subjects
    }
}
