@testable import MereRunApp
import Foundation
import XCTest

final class StudioSubjectsProjectTests: XCTestCase {
    // MARK: - Stage state machine

    func testFreshProjectStartsAtPlanWithNothingDone() {
        let progress = StudioSubjectsProgress()
        XCTAssertEqual(progress.defaultStage, .plan)
        XCTAssertEqual(progress.status(of: .plan, selected: .plan), .active)
        XCTAssertEqual(progress.status(of: .track, selected: .plan), .todo)
        XCTAssertEqual(progress.status(of: .animate, selected: .plan), .todo)
    }

    func testPlannedProjectLandsOnTrackAndMarksPlanDone() {
        let progress = StudioSubjectsProgress(planReady: true)
        XCTAssertEqual(progress.defaultStage, .track)
        XCTAssertEqual(progress.status(of: .plan, selected: .track), .done)
        XCTAssertEqual(progress.status(of: .track, selected: .track), .active)
        XCTAssertEqual(progress.status(of: .animate, selected: .track), .todo)
    }

    func testTrackedProjectLandsOnAnimateAndSelectionOverridesDone() {
        let progress = StudioSubjectsProgress(planReady: true, previewed: true, tracked: true)
        XCTAssertEqual(progress.defaultStage, .animate)
        // Looking back at a finished stage shows it as the active row, not a check.
        XCTAssertEqual(progress.status(of: .track, selected: .track), .active)
        XCTAssertEqual(progress.status(of: .plan, selected: .track), .done)
    }

    func testFullyAnimatedProjectStaysOnAnimate() {
        let progress = StudioSubjectsProgress(planReady: true, previewed: true, tracked: true, animated: true)
        XCTAssertEqual(progress.defaultStage, .animate)
        XCTAssertEqual(progress.status(of: .animate, selected: .plan), .done)
    }

    func testStageOrderAndNext() {
        XCTAssertEqual(StudioSubjectsStage.allCases, [.plan, .track, .animate])
        XCTAssertEqual(StudioSubjectsStage.plan.next, .track)
        XCTAssertEqual(StudioSubjectsStage.track.next, .animate)
        XCTAssertNil(StudioSubjectsStage.animate.next)
    }

    // MARK: - Stage header copy

    func testPlanCopyGatesOnPlanReadiness() {
        let notReady = StudioSubjectsStageCopy.copy(for: .plan, progress: StudioSubjectsProgress())
        XCTAssertEqual(notReady.title, "Plan the subjects")
        XCTAssertEqual(notReady.primary.label, "Continue to Track")
        XCTAssertFalse(notReady.primary.isEnabled)
        XCTAssertEqual(notReady.secondary?.label, "Preview masks")
        XCTAssertFalse(notReady.secondary?.isEnabled ?? true)

        let ready = StudioSubjectsStageCopy.copy(for: .plan, progress: StudioSubjectsProgress(planReady: true))
        XCTAssertTrue(ready.primary.isEnabled)
        XCTAssertEqual(ready.primary.action, .continueTo(.track))
        XCTAssertEqual(ready.secondary?.action, .previewMasks)
        XCTAssertTrue(ready.secondary?.isEnabled ?? false)
    }

    func testTrackCopyBeforeAndAfterTracking() {
        let untracked = StudioSubjectsStageCopy.copy(for: .track, progress: StudioSubjectsProgress(planReady: true))
        XCTAssertEqual(untracked.title, "Track subjects through the clip")
        XCTAssertEqual(untracked.primary.label, "Track subjects")
        XCTAssertEqual(untracked.primary.action, .trackMasks)
        XCTAssertEqual(untracked.secondary?.label, "Preview frame")

        let tracked = StudioSubjectsStageCopy.copy(
            for: .track,
            progress: StudioSubjectsProgress(planReady: true, previewed: true, tracked: true)
        )
        XCTAssertEqual(tracked.description, "Review masks, add a correction where a mask slips, then continue to Animate.")
        XCTAssertEqual(tracked.secondary?.label, "Re-track")
        XCTAssertEqual(tracked.primary.label, "Continue to Animate")
        XCTAssertEqual(tracked.primary.action, .continueTo(.animate))
        XCTAssertTrue(tracked.primary.isEnabled)
    }

    func testTrackCopyWhileTrackingDisablesActions() {
        let tracking = StudioSubjectsStageCopy.copy(
            for: .track,
            progress: StudioSubjectsProgress(planReady: true, previewed: true, tracked: true, runningStage: .track)
        )
        XCTAssertEqual(tracking.secondary?.label, "Tracking…")
        XCTAssertFalse(tracking.secondary?.isEnabled ?? true)
        // Moving on to Animate stays possible; that stage disables its own actions while busy.
        XCTAssertTrue(tracking.primary.isEnabled)

        let firstTrack = StudioSubjectsStageCopy.copy(
            for: .track,
            progress: StudioSubjectsProgress(planReady: true, runningStage: .track)
        )
        XCTAssertEqual(firstTrack.primary.label, "Tracking…")
        XCTAssertFalse(firstTrack.primary.isEnabled)
    }

    func testAnimateCopyRequiresTrackedMasks() {
        let untracked = StudioSubjectsStageCopy.copy(for: .animate, progress: StudioSubjectsProgress(planReady: true))
        XCTAssertEqual(untracked.title, "Animate the reference")
        XCTAssertFalse(untracked.primary.isEnabled)
        XCTAssertFalse(untracked.secondary?.isEnabled ?? true)

        let tracked = StudioSubjectsStageCopy.copy(
            for: .animate,
            progress: StudioSubjectsProgress(planReady: true, previewed: true, tracked: true)
        )
        XCTAssertEqual(tracked.primary.label, "Animate")
        XCTAssertEqual(tracked.primary.action, .animate)
        XCTAssertTrue(tracked.primary.isEnabled)
        XCTAssertEqual(tracked.secondary?.action, .validateRun)

        let animating = StudioSubjectsStageCopy.copy(
            for: .animate,
            progress: StudioSubjectsProgress(planReady: true, previewed: true, tracked: true, runningStage: .animate)
        )
        XCTAssertEqual(animating.primary.label, "Animating…")
        XCTAssertFalse(animating.primary.isEnabled)
    }

    func testBoardStacksBelowTheSideBySideWidth() {
        // 1440 window with the Library open leaves ~690pt; the 1280 default leaves ~530pt.
        XCTAssertTrue(StudioSCAILView.isWide(columnWidth: 740))
        XCTAssertTrue(StudioSCAILView.isWide(columnWidth: 688))
        XCTAssertFalse(StudioSCAILView.isWide(columnWidth: 580))
    }

    // MARK: - Subject rows

    func testRowLeadPrefersBoxThenPointThenReference() {
        var subject = StudioSCAILSubject(name: "skater", color: "blue", referenceImage: "/refs/skater-front.png")
        XCTAssertEqual(StudioSubjectsRowCopy.lead(for: subject), "ref: skater-front.png")

        subject.drivingPositivePoints = "612.4,479.6; 10,10"
        XCTAssertEqual(StudioSubjectsRowCopy.lead(for: subject), "point (612, 480)")

        subject.drivingBox = "500,200,600,330"
        XCTAssertEqual(StudioSubjectsRowCopy.lead(for: subject), "box")
    }

    func testRowLeadFallsBackToDrivingPromptOrPlaceholder() {
        var subject = StudioSCAILSubject(name: "dancer", color: "red", drivingPrompt: "dancer")
        XCTAssertEqual(StudioSubjectsRowCopy.lead(for: subject), "“dancer”")
        subject.drivingPrompt = ""
        XCTAssertEqual(StudioSubjectsRowCopy.lead(for: subject), "no selector yet")
    }

    func testRowMetaShowsTrackedFramesOrCorrections() {
        let skater = StudioSCAILSubject(name: "skater", color: "blue", referenceImage: "skater-front.png")
        XCTAssertEqual(
            StudioSubjectsRowCopy.meta(subject: skater, trackedFrames: (visible: 240, total: 240), correctionFrames: []),
            "ref: skater-front.png · 240/240 frames"
        )
        XCTAssertEqual(
            StudioSubjectsRowCopy.meta(subject: skater, trackedFrames: nil, correctionFrames: []),
            "ref: skater-front.png"
        )

        var backpack = StudioSCAILSubject(name: "backpack", color: "green")
        backpack.drivingBox = "1,1,2,2"
        XCTAssertEqual(
            StudioSubjectsRowCopy.meta(subject: backpack, trackedFrames: (visible: 240, total: 240), correctionFrames: [142, 88]),
            "box · 2 corrections at 88, 142"
        )
        XCTAssertEqual(
            StudioSubjectsRowCopy.meta(subject: backpack, trackedFrames: nil, correctionFrames: [12]),
            "box · 1 correction at 12"
        )
    }

    // MARK: - Stats

    func testStatsFromTrackedArtifacts() {
        let manifest = Self.manifest(tracked: true, frameCount: 240, fps: 24)
        let tracking = StudioSCAILTrackingReport(
            frameCount: 240,
            fps: 24,
            subjects: [
                Self.trackedSubject("skater", visible: 240, total: 240),
                Self.trackedSubject("board", visible: 231, total: 240),
                Self.trackedSubject("backpack", visible: 240, total: 240),
            ]
        )
        let quality = StudioSCAILQualityReport(
            blockingErrors: [],
            warnings: [
                StudioSCAILQualityReport.Warning(code: "weak_score", subjectID: "board", frameIndex: 12, message: "weak"),
            ]
        )
        let stats = StudioSubjectsStats.stats(manifest: manifest, tracking: tracking, quality: quality, correctionCount: 2)
        XCTAssertEqual(stats.coverage, "98.8%")
        XCTAssertEqual(stats.drift, "low")
        XCTAssertEqual(stats.corrections, "2")
        XCTAssertEqual(stats.frames, "240 @ 24 fps")
        XCTAssertEqual(stats.panels.map(\.label), ["Coverage", "Drift", "Corrections", "Frames"])
    }

    func testDriftCountsAbruptChangesOnly() {
        let quality = StudioSCAILQualityReport(
            blockingErrors: [],
            warnings: [
                StudioSCAILQualityReport.Warning(code: "abrupt_change", subjectID: "board", frameIndex: 40, message: ""),
                StudioSCAILQualityReport.Warning(code: "abrupt_change", subjectID: "board", frameIndex: 41, message: ""),
                StudioSCAILQualityReport.Warning(code: "disappearance", subjectID: "board", frameIndex: 42, message: ""),
            ]
        )
        let stats = StudioSubjectsStats.stats(
            manifest: Self.manifest(tracked: true, frameCount: 100, fps: 23.976),
            tracking: nil,
            quality: quality,
            correctionCount: 0
        )
        XCTAssertEqual(stats.drift, "2 flagged")
        XCTAssertNil(stats.coverage)
        XCTAssertEqual(stats.frames, "100 @ 23.98 fps")
    }

    func testStatsAfterPreviewOnlyOmitCoverageAndDrift() {
        let quality = StudioSCAILQualityReport(blockingErrors: [], warnings: [])
        let stats = StudioSubjectsStats.stats(
            manifest: Self.manifest(tracked: false, frameCount: 240, fps: 24),
            tracking: nil,
            quality: quality,
            correctionCount: 0
        )
        XCTAssertNil(stats.coverage)
        XCTAssertNil(stats.drift)
        XCTAssertEqual(stats.panels.map(\.label), ["Corrections", "Frames"])
    }

    func testStatsWithoutArtifactsOnlyCountCorrections() {
        let stats = StudioSubjectsStats.stats(manifest: nil, tracking: nil, quality: nil, correctionCount: 1)
        XCTAssertEqual(stats.panels.map(\.label), ["Corrections"])
        XCTAssertEqual(stats.corrections, "1")
    }

    func testPercentFormatting() {
        XCTAssertEqual(StudioSubjectsStats.formatPercent(1), "100%")
        XCTAssertEqual(StudioSubjectsStats.formatPercent(0.5), "50%")
        XCTAssertEqual(StudioSubjectsStats.formatPercent(0.98750), "98.8%")
        XCTAssertEqual(StudioSubjectsStats.formatPercent(0.98611), "98.6%")
        XCTAssertEqual(StudioSubjectsStats.formatPercent(1.4), "100%")
    }

    // MARK: - Project rail

    func testProjectNameComesFromDrivingClip() {
        XCTAssertEqual(StudioSubjectsProjectCopy.name(drivingVideo: "/clips/skate-clip-01.mp4"), "skate-clip-01")
        XCTAssertEqual(StudioSubjectsProjectCopy.name(drivingVideo: "   "), "Untitled project")
    }

    func testProjectSummaryAndSavedCopy() {
        XCTAssertEqual(StudioSubjectsProjectCopy.summary(subjectCount: 3, frameCount: 240), "3 subjects · 240 frames")
        XCTAssertEqual(StudioSubjectsProjectCopy.summary(subjectCount: 1, frameCount: nil), "1 subject")
        XCTAssertEqual(StudioSubjectsProjectCopy.saved(nil), "Not saved yet")

        var components = DateComponents()
        components.year = 2_026
        components.month = 9
        components.day = 3
        components.hour = 13
        components.minute = 26
        let date = Calendar.current.date(from: components)!
        XCTAssertEqual(StudioSubjectsProjectCopy.saved(date), "Saved 1:26 PM")
    }

    // MARK: - Job bar

    func testJobBarDetailPerPhase() {
        XCTAssertEqual(
            StudioSubjectsJobBar.detail(phase: .running(.track), subjectCount: 3, frameCount: 240, profile: "fast"),
            "Tracking 3 subjects · 240 frames"
        )
        XCTAssertEqual(
            StudioSubjectsJobBar.detail(phase: .running(.track), subjectCount: 1, frameCount: nil, profile: "fast"),
            "Tracking 1 subject"
        )
        XCTAssertEqual(
            StudioSubjectsJobBar.detail(phase: .running(.preview(frame: 88)), subjectCount: 3, frameCount: 240, profile: "fast"),
            "Previewing frame 88 · 3 subjects"
        )
        XCTAssertEqual(
            StudioSubjectsJobBar.detail(phase: .running(.animate), subjectCount: 3, frameCount: 240, profile: "quality"),
            "Animating with SCAIL-2 · quality profile"
        )
        XCTAssertEqual(
            StudioSubjectsJobBar.detail(phase: .queued(.animate), subjectCount: 3, frameCount: 240, profile: "fast"),
            "Animation queued behind the active job"
        )
        XCTAssertEqual(
            StudioSubjectsJobBar.detail(phase: .ended(.track, exitCode: 0), subjectCount: 3, frameCount: 240, profile: "fast"),
            "Masks tracked · 3 subjects · 240 frames"
        )
        XCTAssertEqual(
            StudioSubjectsJobBar.detail(phase: .ended(.validate, exitCode: 2), subjectCount: 3, frameCount: 240, profile: "fast"),
            "Validation failed · exit 2"
        )
        XCTAssertEqual(
            StudioSubjectsJobBar.detail(phase: .idle, subjectCount: 3, frameCount: nil, profile: "fast"),
            "No job running"
        )
    }

    func testJobStageRouting() {
        XCTAssertEqual(StudioSubjectsJobBar.Job.preview(frame: 1).stage, .track)
        XCTAssertEqual(StudioSubjectsJobBar.Job.track.stage, .track)
        XCTAssertEqual(StudioSubjectsJobBar.Job.validate.stage, .animate)
        XCTAssertEqual(StudioSubjectsJobBar.Job.animate.stage, .animate)
    }

    // MARK: - Artifacts

    func testManifestDecodesCLIShape() throws {
        let json = """
        {
          "schema_version": 1,
          "status": "ready",
          "preview_frame": null,
          "model_id": "vision-segment-sam31",
          "model_revision": "r1",
          "driving_source_path": "/clips/skate-clip-01.mp4",
          "driving_proxy_path": "driving-proxy.mp4",
          "driving_mask_path": "driving-mask.mov",
          "overlay_preview_path": "overlay-preview.mp4",
          "contact_sheet_path": "contact-sheet.png",
          "tracking_path": "tracking.json",
          "quality_path": "quality.json",
          "frame_count": 240,
          "fps": 24,
          "width": 832,
          "height": 480,
          "subjects": [
            {
              "id": "skater",
              "color": "blue",
              "reference_image_path": "/refs/skater-front.png",
              "prepared_reference_image_path": "reference-skater-prepared.png",
              "reference_mask_path": "reference-skater-mask.png",
              "seed_frame_index": 0,
              "gap_ranges": []
            }
          ],
          "corrections": [
            {"subject_id": "skater", "frame_index": 88, "painted_mask_path": null,
             "positive_point_count": 1, "negative_point_count": 0, "has_box": false}
          ],
          "artifacts": []
        }
        """
        let manifest = try JSONDecoder().decode(StudioSCAILManifest.self, from: Data(json.utf8))
        XCTAssertTrue(manifest.isTracked)
        XCTAssertEqual(manifest.frameCount, 240)
        XCTAssertEqual(manifest.fps, 24)
        XCTAssertEqual(manifest.subjects.first?.id, "skater")
        XCTAssertEqual(manifest.corrections?.first?.frameIndex, 88)
        XCTAssertEqual(manifest.trackingPath, "tracking.json")
    }

    func testTrackingReportCountsVisibleFrames() throws {
        let json = """
        {
          "schema_version": 1,
          "model_id": "vision-segment-sam31",
          "frame_count": 3,
          "fps": 24,
          "subjects": [
            {
              "id": "board",
              "color": "red",
              "seed_frame_index": 0,
              "frames": [
                {"frame_index": 0, "timestamp_seconds": 0, "detections": [{"object_id": "1", "label": "board", "score": 0.9, "visible": true, "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}, "mask_area_pixels": 10}]},
                {"frame_index": 1, "timestamp_seconds": 0.04, "detections": [{"object_id": "1", "label": "board", "score": 0.2, "visible": false, "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}, "mask_area_pixels": 0}]},
                {"frame_index": 2, "timestamp_seconds": 0.08, "detections": []}
              ]
            }
          ]
        }
        """
        let report = try JSONDecoder().decode(StudioSCAILTrackingReport.self, from: Data(json.utf8))
        XCTAssertEqual(report.subjects.first?.visibleFrameCount, 1)
        XCTAssertEqual(report.frameCount, 3)
    }

    // MARK: - Helpers

    private static func manifest(tracked: Bool, frameCount: Int, fps: Double) -> StudioSCAILManifest {
        StudioSCAILManifest(
            status: tracked ? "ready" : "preview_ready",
            previewFrame: tracked ? nil : 0,
            drivingSourcePath: "/clips/skate-clip-01.mp4",
            drivingProxyPath: tracked ? "driving-proxy.mp4" : nil,
            drivingMaskPath: tracked ? "driving-mask.mov" : nil,
            overlayPreviewPath: tracked ? "overlay-preview.mp4" : "overlay-frame-0.png",
            contactSheetPath: "contact-sheet.png",
            trackingPath: tracked ? "tracking.json" : nil,
            qualityPath: "quality.json",
            frameCount: frameCount,
            fps: fps,
            subjects: [],
            corrections: nil
        )
    }

    private static func trackedSubject(_ id: String, visible: Int, total: Int) -> StudioSCAILTrackingReport.Subject {
        StudioSCAILTrackingReport.Subject(
            id: id,
            frames: (0..<total).map { index in
                StudioSCAILTrackingReport.Frame(
                    frameIndex: index,
                    detections: [StudioSCAILTrackingReport.Detection(visible: index < visible, score: 0.9)]
                )
            }
        )
    }
}
