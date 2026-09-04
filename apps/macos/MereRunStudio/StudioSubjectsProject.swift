import Foundation
import SwiftUI

// MARK: - Plan model (what the user authors)

struct StudioSCAILSubject: Identifiable, Equatable {
    let id = UUID()
    var name: String
    var color: String
    var referenceImage = ""
    var referencePrompt = ""
    var drivingPrompt = ""
    var referenceBox = ""
    var drivingBox = ""
    var referencePositivePoints = ""
    var drivingPositivePoints = ""
    var referenceNegativePoints = ""
    var drivingNegativePoints = ""
}

struct StudioSCAILCorrection: Identifiable, Equatable {
    let id = UUID()
    var subjectID: String
    var frameIndex = 0
    var box = ""
    var positivePoints = ""
    var negativePoints = ""
    var paintedMaskPath = ""
}

// MARK: - plan.json (what the CLI reads)

struct StudioSCAILSelectorPlan: Codable {
    let text: String?
    let box: StudioSCAILBoxPlan?
    let positivePoints: [StudioSCAILPointPlan]
    let negativePoints: [StudioSCAILPointPlan]

    enum CodingKeys: String, CodingKey {
        case text
        case box
        case positivePoints = "positive_points"
        case negativePoints = "negative_points"
    }
}

struct StudioSCAILPointPlan: Codable {
    let x: Float
    let y: Float
}

struct StudioSCAILBoxPlan: Codable {
    let x1: Float
    let y1: Float
    let x2: Float
    let y2: Float
}

struct StudioSCAILSubjectPlan: Codable {
    let id: String
    let color: String
    let referenceImage: String
    let referenceSelector: StudioSCAILSelectorPlan
    let drivingSelector: StudioSCAILSelectorPlan

    enum CodingKeys: String, CodingKey {
        case id
        case color
        case referenceImage = "reference_image"
        case referenceSelector = "reference_selector"
        case drivingSelector = "driving_selector"
    }
}

struct StudioSCAILMaskPlan: Codable {
    let schemaVersion = 1
    let mode: String
    let drivingVideo: String
    let inSeconds: Double?
    let outSeconds: Double?
    let width: Int
    let height: Int
    let fps: Double
    let subjects: [StudioSCAILSubjectPlan]
    let corrections: [StudioSCAILCorrectionPlan]
    let threshold: Float
    let resolution: Int
    let seedFrameSearchLimit: Int
    let paletteTolerance = 192

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
}

struct StudioSCAILCorrectionPlan: Codable {
    let subjectID: String
    let frameIndex: Int
    let box: StudioSCAILBoxPlan?
    let positivePoints: [StudioSCAILPointPlan]
    let negativePoints: [StudioSCAILPointPlan]
    let paintedBinaryCorrectionPNG: String?

    enum CodingKeys: String, CodingKey {
        case subjectID = "subject_id"
        case frameIndex = "frame_index"
        case box
        case positivePoints = "positive_points"
        case negativePoints = "negative_points"
        case paintedBinaryCorrectionPNG = "painted_binary_correction_png"
    }
}

enum StudioSCAILPlanError: LocalizedError {
    case invalidBox(String)
    case invalidPoints(String)

    var errorDescription: String? {
        switch self {
        case .invalidBox(let label):
            "\(label) box must be four comma-separated numbers with x2>x1 and y2>y1."
        case .invalidPoints(let label):
            "\(label) points must use x,y pairs separated by semicolons."
        }
    }
}

// MARK: - Mask artifacts (what `video prepare-masks` writes)

/// The subset of `SCAIL2MaskManifest` the app reads. Paths are relative to the output directory
/// unless absolute. `drivingMaskPath` is nil after a preview and set after a full track.
struct StudioSCAILManifest: Codable, Equatable {
    struct Subject: Codable, Equatable {
        let id: String
        let color: String
        let preparedReferenceImagePath: String
        let referenceMaskPath: String

        enum CodingKeys: String, CodingKey {
            case id
            case color
            case preparedReferenceImagePath = "prepared_reference_image_path"
            case referenceMaskPath = "reference_mask_path"
        }
    }

    struct Correction: Codable, Equatable {
        let subjectID: String
        let frameIndex: Int

        enum CodingKeys: String, CodingKey {
            case subjectID = "subject_id"
            case frameIndex = "frame_index"
        }
    }

    let status: String
    let previewFrame: Int?
    let drivingSourcePath: String
    let drivingProxyPath: String?
    let drivingMaskPath: String?
    let overlayPreviewPath: String
    let contactSheetPath: String
    let trackingPath: String?
    let qualityPath: String?
    let frameCount: Int
    let fps: Double?
    let subjects: [Subject]
    let corrections: [Correction]?

    enum CodingKeys: String, CodingKey {
        case status
        case previewFrame = "preview_frame"
        case drivingSourcePath = "driving_source_path"
        case drivingProxyPath = "driving_proxy_path"
        case drivingMaskPath = "driving_mask_path"
        case overlayPreviewPath = "overlay_preview_path"
        case contactSheetPath = "contact_sheet_path"
        case trackingPath = "tracking_path"
        case qualityPath = "quality_path"
        case frameCount = "frame_count"
        case fps
        case subjects
        case corrections
    }

    var isTracked: Bool { drivingMaskPath != nil }
}

/// The subset of `SCAIL2MaskTrackingReport` (`tracking.json`) the app reads: which frames each
/// subject was visible in.
struct StudioSCAILTrackingReport: Codable, Equatable {
    struct Detection: Codable, Equatable {
        let visible: Bool
        let score: Float
    }

    struct Frame: Codable, Equatable {
        let frameIndex: Int
        let detections: [Detection]

        enum CodingKeys: String, CodingKey {
            case frameIndex = "frame_index"
            case detections
        }
    }

    struct Subject: Codable, Equatable {
        let id: String
        let frames: [Frame]

        /// Frames in which the subject has at least one visible detection.
        var visibleFrameCount: Int {
            frames.filter { frame in frame.detections.contains { $0.visible } }.count
        }
    }

    let frameCount: Int
    let fps: Double
    let subjects: [Subject]

    enum CodingKeys: String, CodingKey {
        case frameCount = "frame_count"
        case fps
        case subjects
    }
}

/// The subset of `SCAIL2MaskQualityReport` (`quality.json`) the app reads.
struct StudioSCAILQualityReport: Codable, Equatable {
    struct Warning: Codable, Equatable {
        let code: String
        let subjectID: String?
        let frameIndex: Int?
        let message: String

        enum CodingKeys: String, CodingKey {
            case code
            case subjectID = "subject_id"
            case frameIndex = "frame_index"
            case message
        }
    }

    let blockingErrors: [String]
    let warnings: [Warning]

    enum CodingKeys: String, CodingKey {
        case blockingErrors = "blocking_errors"
        case warnings
    }

    /// The CLI flags a frame where a subject's mask changes abruptly between neighbours.
    static let driftCode = "abrupt_change"

    var driftFrameCount: Int {
        warnings.filter { $0.code == Self.driftCode }.count
    }
}

// MARK: - Stages

enum StudioSubjectsStage: Int, CaseIterable, Identifiable {
    case plan = 1
    case track
    case animate

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .plan: "Plan"
        case .track: "Track"
        case .animate: "Animate"
        }
    }

    var next: StudioSubjectsStage? {
        StudioSubjectsStage(rawValue: rawValue + 1)
    }
}

enum StudioSubjectsStageStatus: Equatable {
    case done
    case active
    case todo
}

/// What the project has achieved so far, derived from the plan, the mask manifest, and the
/// Library. Drives the rail, the header copy, and which actions are enabled.
struct StudioSubjectsProgress: Equatable {
    /// The plan validates: a driving clip and every subject has a reference and selectors.
    var planReady = false
    /// A preview or a full track has produced a manifest.
    var previewed = false
    /// A full track has produced a driving mask (`drivingMaskPath`).
    var tracked = false
    /// An animation run has produced a video.
    var animated = false
    /// The stage whose job is queued or running, if any.
    var runningStage: StudioSubjectsStage?

    func isDone(_ stage: StudioSubjectsStage) -> Bool {
        switch stage {
        case .plan: planReady
        case .track: tracked
        case .animate: animated
        }
    }

    /// The first stage that is not done; where a returning user should land.
    var defaultStage: StudioSubjectsStage {
        StudioSubjectsStage.allCases.first { !isDone($0) } ?? .animate
    }

    func status(of stage: StudioSubjectsStage, selected: StudioSubjectsStage) -> StudioSubjectsStageStatus {
        if stage == selected { return .active }
        return isDone(stage) ? .done : .todo
    }
}

/// The header of each stage: serif title, one-line description, and the two actions.
struct StudioSubjectsStageCopy: Equatable {
    enum Action: Equatable {
        case previewMasks
        case trackMasks
        case validateRun
        case animate
        case continueTo(StudioSubjectsStage)
    }

    struct Button: Equatable {
        let label: String
        let action: Action
        let isEnabled: Bool
    }

    let title: String
    let description: String
    let secondary: Button?
    let primary: Button

    static func copy(for stage: StudioSubjectsStage, progress: StudioSubjectsProgress) -> StudioSubjectsStageCopy {
        let busy = progress.runningStage != nil
        switch stage {
        case .plan:
            return StudioSubjectsStageCopy(
                title: "Plan the subjects",
                description: "Choose the driving clip, then give each subject a reference image and a selector.",
                secondary: Button(label: "Preview masks", action: .previewMasks, isEnabled: progress.planReady && !busy),
                primary: Button(label: "Continue to Track", action: .continueTo(.track), isEnabled: progress.planReady)
            )
        case .track:
            let tracking = progress.runningStage == .track
            if progress.tracked {
                return StudioSubjectsStageCopy(
                    title: "Track subjects through the clip",
                    description: "Review masks, add a correction where a mask slips, then continue to Animate.",
                    secondary: Button(label: tracking ? "Tracking…" : "Re-track", action: .trackMasks, isEnabled: !busy),
                    // Moving on is always allowed; the Animate stage gates its own action on the job.
                    primary: Button(label: "Continue to Animate", action: .continueTo(.animate), isEnabled: true)
                )
            }
            return StudioSubjectsStageCopy(
                title: "Track subjects through the clip",
                description: progress.previewed
                    ? "Check the preview frame, then track every subject through the whole clip."
                    : "Preview one frame to check the selectors, then track the whole clip.",
                secondary: Button(label: "Preview frame", action: .previewMasks, isEnabled: progress.planReady && !busy),
                primary: Button(
                    label: tracking ? "Tracking…" : "Track subjects",
                    action: .trackMasks,
                    isEnabled: progress.planReady && !busy
                )
            )
        case .animate:
            let animating = progress.runningStage == .animate
            return StudioSubjectsStageCopy(
                title: "Animate the reference",
                description: progress.tracked
                    ? "Direct the finished shot, pick a profile, and render with the tracked masks."
                    : "Track the clip first; animation renders from the tracked masks.",
                secondary: Button(label: "Validate run", action: .validateRun, isEnabled: progress.tracked && !busy),
                primary: Button(
                    label: animating ? "Animating…" : "Animate",
                    action: .animate,
                    isEnabled: progress.tracked && !busy
                )
            )
        }
    }
}

// MARK: - Subject rows

/// The 11pt meta line under a subject's name. The lead segment says how the subject is picked
/// in the driving clip (a box, a point, or by its reference), and the trailing segment is what
/// tracking found: corrections when there are any, otherwise the visible-frame count.
enum StudioSubjectsRowCopy {
    static func meta(
        subject: StudioSCAILSubject,
        trackedFrames: (visible: Int, total: Int)?,
        correctionFrames: [Int]
    ) -> String {
        var segments = [lead(for: subject)]
        if !correctionFrames.isEmpty {
            let frames = correctionFrames.sorted().map(String.init).joined(separator: ", ")
            segments.append("\(correctionFrames.count) correction\(correctionFrames.count == 1 ? "" : "s") at \(frames)")
        } else if let trackedFrames, trackedFrames.total > 0 {
            segments.append("\(trackedFrames.visible)/\(trackedFrames.total) frames")
        }
        return segments.joined(separator: " · ")
    }

    static func lead(for subject: StudioSCAILSubject) -> String {
        if !subject.drivingBox.isBlank {
            return "box"
        }
        if let point = firstPoint(in: subject.drivingPositivePoints) {
            return "point (\(point.x), \(point.y))"
        }
        if !subject.referenceImage.isBlank {
            return "ref: \(URL(fileURLWithPath: subject.referenceImage).lastPathComponent)"
        }
        if !subject.drivingPrompt.isBlank {
            return "“\(subject.drivingPrompt)”"
        }
        return "no selector yet"
    }

    private static func firstPoint(in raw: String) -> (x: Int, y: Int)? {
        let pair = raw.replacingOccurrences(of: "\n", with: ";").split(separator: ";").first ?? ""
        let values = pair.split(separator: ",").compactMap {
            Float($0.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard values.count == 2 else { return nil }
        return (Int(values[0].rounded()), Int(values[1].rounded()))
    }
}

// MARK: - Stats

/// The four stat panels under the preview. Only values the artifacts actually carry are set;
/// the view omits a panel whose value is nil.
struct StudioSubjectsStats: Equatable {
    var coverage: String?
    var drift: String?
    var corrections: String?
    var frames: String?

    var panels: [(label: String, value: String)] {
        [
            ("Coverage", coverage),
            ("Drift", drift),
            ("Corrections", corrections),
            ("Frames", frames),
        ].compactMap { label, value in value.map { (label, $0) } }
    }

    static func stats(
        manifest: StudioSCAILManifest?,
        tracking: StudioSCAILTrackingReport?,
        quality: StudioSCAILQualityReport?,
        correctionCount: Int
    ) -> StudioSubjectsStats {
        var stats = StudioSubjectsStats(corrections: String(correctionCount))
        if let tracking, tracking.frameCount > 0, !tracking.subjects.isEmpty {
            let possible = tracking.frameCount * tracking.subjects.count
            let visible = tracking.subjects.reduce(0) { $0 + $1.visibleFrameCount }
            stats.coverage = formatPercent(Double(visible) / Double(possible))
        }
        if let quality, manifest?.isTracked == true {
            stats.drift = quality.driftFrameCount == 0 ? "low" : "\(quality.driftFrameCount) flagged"
        }
        if let manifest, manifest.frameCount > 0 {
            if let fps = manifest.fps ?? tracking?.fps, fps > 0 {
                stats.frames = "\(manifest.frameCount) @ \(formatFPS(fps)) fps"
            } else {
                stats.frames = String(manifest.frameCount)
            }
        }
        return stats
    }

    static func formatPercent(_ fraction: Double) -> String {
        let clamped = min(1, max(0, fraction))
        let tenths = (clamped * 1_000).rounded() / 10
        if tenths == tenths.rounded() {
            return "\(Int(tenths))%"
        }
        return String(format: "%.1f%%", tenths)
    }

    static func formatFPS(_ fps: Double) -> String {
        if fps == fps.rounded() {
            return String(Int(fps))
        }
        return String(format: "%.2f", fps)
    }
}

// MARK: - Project rail

enum StudioSubjectsProjectCopy {
    /// The driving clip's file name without its extension names the project.
    static func name(drivingVideo: String) -> String {
        let trimmed = drivingVideo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Untitled project" }
        let stem = URL(fileURLWithPath: trimmed).deletingPathExtension().lastPathComponent
        return stem.isEmpty ? "Untitled project" : stem
    }

    /// `3 subjects · 240 frames`; the frame count is only known once a preview or track ran.
    static func summary(subjectCount: Int, frameCount: Int?) -> String {
        var parts = ["\(subjectCount) subject\(subjectCount == 1 ? "" : "s")"]
        if let frameCount, frameCount > 0 {
            parts.append("\(frameCount) frames")
        }
        return parts.joined(separator: " · ")
    }

    /// `Saved 1:26 PM` from the last plan.json write.
    static func saved(_ date: Date?) -> String {
        guard let date else { return "Not saved yet" }
        return "Saved \(timeFormatter.string(from: date))"
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "h:mm a"
        return formatter
    }()
}

// MARK: - Job bar

enum StudioSubjectsJobBar {
    enum Job: Equatable {
        case preview(frame: Int)
        case track
        case validate
        case animate

        var stage: StudioSubjectsStage {
            switch self {
            case .preview, .track: .track
            case .validate, .animate: .animate
            }
        }
    }

    enum Phase: Equatable {
        case idle
        case queued(Job)
        case running(Job)
        case ended(Job, exitCode: Int32?)
    }

    static func detail(
        phase: Phase,
        subjectCount: Int,
        frameCount: Int?,
        profile: String
    ) -> String {
        let subjects = "\(subjectCount) subject\(subjectCount == 1 ? "" : "s")"
        let frames = frameCount.map { " · \($0) frames" } ?? ""
        switch phase {
        case .idle:
            return "No job running"
        case .queued(let job):
            return "\(verb(for: job)) queued behind the active job"
        case .running(let job):
            switch job {
            case .preview(let frame):
                return "Previewing frame \(frame) · \(subjects)"
            case .track:
                return "Tracking \(subjects)\(frames)"
            case .validate:
                return "Validating the SCAIL-2 run · \(profileName(profile))"
            case .animate:
                return "Animating with SCAIL-2 · \(profileName(profile))"
            }
        case .ended(let job, let exitCode):
            if let exitCode, exitCode != 0 {
                return "\(verb(for: job)) failed · exit \(exitCode)"
            }
            switch job {
            case .preview(let frame):
                return "Frame \(frame) previewed · \(subjects)"
            case .track:
                return "Masks tracked · \(subjects)\(frames)"
            case .validate:
                return "Run validated · \(profileName(profile))"
            case .animate:
                return "Animation finished · \(profileName(profile))"
            }
        }
    }

    static func dotColor(phase: Phase) -> Color {
        switch phase {
        case .idle: MereRunTheme.textMuted
        case .queued: MereRunTheme.yellow
        case .running: MereRunTheme.accent
        case .ended(_, let exitCode):
            exitCode == 0 || exitCode == nil ? MereRunTheme.green : MereRunTheme.red
        }
    }

    private static func verb(for job: Job) -> String {
        switch job {
        case .preview: "Preview"
        case .track: "Tracking"
        case .validate: "Validation"
        case .animate: "Animation"
        }
    }

    private static func profileName(_ profile: String) -> String {
        profile == "quality" ? "quality profile" : "fast profile"
    }
}

// MARK: - Palette

enum StudioSubjectsPalette {
    static let names = ["blue", "red", "green", "magenta", "cyan", "yellow"]

    /// The mask palette the CLI paints with (`SCAIL2SubjectColor.rgba8`), so swatches match the
    /// overlay tint.
    static func color(named name: String) -> Color {
        switch name {
        case "blue": Color(hex: "0000FF")
        case "red": Color(hex: "FF0000")
        case "green": Color(hex: "00FF00")
        case "magenta": Color(hex: "FF00FF")
        case "cyan": Color(hex: "00FFFF")
        case "yellow": Color(hex: "FFFF00")
        default: MereRunTheme.textMuted
        }
    }
}

// MARK: - Test seam

/// When set in the environment, Video ▸ Subjects opens on this project instead of an empty
/// plan, so the snapshot harness can render the Track stage from artifacts it wrote itself.
/// Production never sets it.
struct StudioSubjectsProjectSeed {
    var mode = "animation"
    var drivingVideo: String
    var width = 832
    var height = 480
    var fps = 24
    var subjects: [StudioSCAILSubject]
    var corrections: [StudioSCAILCorrection] = []
    /// A directory holding `manifest.json` (and whatever it references).
    var preparedDirectory: URL?
    var planSavedAt: Date?
    var stage: StudioSubjectsStage = .track
    /// The Library row of a mask job that is still running.
    var maskRequestID: UUID?
}

private struct StudioSubjectsProjectSeedKey: EnvironmentKey {
    static let defaultValue: StudioSubjectsProjectSeed? = nil
}

extension EnvironmentValues {
    var studioSubjectsProjectSeed: StudioSubjectsProjectSeed? {
        get { self[StudioSubjectsProjectSeedKey.self] }
        set { self[StudioSubjectsProjectSeedKey.self] = newValue }
    }
}
