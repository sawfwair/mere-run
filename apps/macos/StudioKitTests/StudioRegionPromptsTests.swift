@testable import StudioKit
import CoreGraphics
import Foundation
import XCTest

/// Boxes and points drawn on a picture: the text the CLI reads them as, the mapping between the
/// picture on screen and its pixels, and how Find's boxes become Segment's prompts.
final class StudioRegionPromptsTests: XCTestCase {
    private let cup = StudioRegionPrompt.box(CGRect(x: 40, y: 30, width: 120, height: 80), label: "coffee cup")
    private let handle = StudioRegionPrompt.point(CGPoint(x: 400.4, y: 259.6), isPositive: true)
    private let shadow = StudioRegionPrompt.point(CGPoint(x: 12, y: 18), isPositive: false, label: "shadow")

    // MARK: - The CLI's text

    /// `VisionSegment.parseBoxPrompt` takes `x1,y1,x2,y2[,label]` and splits on commas, so the
    /// label can never carry one; coordinates are whole pixels.
    func testBoxTextIsWhatTheCLIParses() {
        XCTAssertEqual(StudioRegionPromptText.boxLines([cup]), ["40,30,160,110,coffee cup"])
        let unlabeled = StudioRegionPrompt.box(CGRect(x: 160.6, y: 110.4, width: -120.6, height: -80.4))
        XCTAssertEqual(StudioRegionPromptText.boxLines([unlabeled]), ["40,30,161,110"])
        let comma = StudioRegionPrompt.box(CGRect(x: 0, y: 0, width: 10, height: 10), label: "  cup,  saucer , ")
        XCTAssertEqual(StudioRegionPromptText.boxLines([comma]), ["0,0,10,10,cup saucer"])
        XCTAssertNil(StudioRegionPrompt.box(.zero, label: " , ").label)
        XCTAssertEqual(StudioRegionPromptText.boxText([cup, handle, unlabeled]), "40,30,160,110,coffee cup\n40,30,161,110")
    }

    /// `VisionSegment.parsePointPrompt` takes `x,y,positive|negative[,label]`.
    func testPointTextIsWhatTheCLIParses() {
        XCTAssertEqual(StudioRegionPromptText.pointLines([handle, shadow]), ["400,260,positive", "12,18,negative,shadow"])
        XCTAssertEqual(StudioRegionPromptText.pointText([cup, handle]), "400,260,positive")
        XCTAssertEqual(StudioRegionPromptText.pointText([cup]), "")
    }

    func testDecodingAcceptsAndRejectsExactlyWhatTheCLIDoes() throws {
        let box = try XCTUnwrap(StudioRegionPromptText.box("161, 110, 40, 30, person"))
        XCTAssertEqual(box.rect, CGRect(x: 40, y: 30, width: 121, height: 80))
        XCTAssertEqual(box.label, "person")
        XCTAssertNil(StudioRegionPromptText.box("1,2,3"))
        XCTAssertNil(StudioRegionPromptText.box("a,b,c,d"))
        XCTAssertNil(StudioRegionPromptText.box("1,2,3,4,5,6"))

        for polarity in ["positive", "pos", "p", "1", "POS"] {
            XCTAssertEqual(StudioRegionPromptText.point("10,20,\(polarity)")?.isPositivePoint, true, polarity)
        }
        for polarity in ["negative", "neg", "n", "0"] {
            XCTAssertEqual(StudioRegionPromptText.point("10,20,\(polarity)")?.isNegativePoint, true, polarity)
        }
        XCTAssertEqual(StudioRegionPromptText.point("10,20,n,face")?.label, "face")
        XCTAssertNil(StudioRegionPromptText.point("10,20"))
        XCTAssertNil(StudioRegionPromptText.point("10,20,maybe"))
        XCTAssertNil(StudioRegionPromptText.point("x,20,positive"))

        let decoded = StudioRegionPromptText.prompts(
            boxText: "40,30,160,110,coffee cup\n\n1,2,3",
            pointText: "400,260,positive\n12,18,negative,shadow"
        )
        XCTAssertEqual(decoded.map(\.isBox), [true, false, false])
        XCTAssertEqual(StudioRegionPromptText.boxText(decoded), "40,30,160,110,coffee cup")
        XCTAssertEqual(StudioRegionPromptText.pointText(decoded), "400,260,positive\n12,18,negative,shadow")
    }

    func testCountAndAccessibilityDescriptions() {
        XCTAssertEqual([cup, handle, shadow].countDescription, "1 box · 2 points")
        XCTAssertEqual([cup, cup].countDescription, "2 boxes")
        XCTAssertEqual([StudioRegionPrompt]().countDescription, "")
        XCTAssertEqual(cup.accessibilityDescription(ordinal: 1), "Box 1, coffee cup, 120 by 80 at 40, 30")
        XCTAssertEqual(handle.accessibilityDescription(ordinal: 2), "Positive point 2 at 400, 260")
        XCTAssertEqual(shadow.accessibilityDescription(ordinal: 3), "Negative point 3, shadow at 12, 18")
    }

    // MARK: - Screen to pixels

    /// A 1000×500 picture in a 400×400 view is letterboxed to 400×200 at y 100; every mapping
    /// goes through that rect and one scale.
    func testViewAndImageCoordinatesMapThroughTheFittedRect() {
        let imageSize = CGSize(width: 1_000, height: 500)
        let viewSize = CGSize(width: 400, height: 400)
        let fitted = StudioAnalyzeGeometry.fittedRect(imageSize: imageSize, in: viewSize)
        XCTAssertEqual(fitted, CGRect(x: 0, y: 100, width: 400, height: 200))
        XCTAssertEqual(StudioRegionGeometry.scale(imageSize: imageSize, fitted: fitted), 0.4)

        XCTAssertEqual(
            StudioRegionGeometry.imagePoint(fromView: CGPoint(x: 200, y: 200), imageSize: imageSize, fitted: fitted),
            CGPoint(x: 500, y: 250)
        )
        XCTAssertEqual(
            StudioRegionGeometry.viewPoint(fromImage: .zero, imageSize: imageSize, fitted: fitted),
            CGPoint(x: 0, y: 100)
        )
        XCTAssertEqual(
            StudioRegionGeometry.viewRect(
                fromImage: CGRect(x: 100, y: 50, width: 200, height: 100), imageSize: imageSize, fitted: fitted
            ),
            CGRect(x: 40, y: 120, width: 80, height: 40)
        )
        // Dragging off the picture clamps to its edge instead of producing coordinates the CLI
        // would draw outside the image.
        XCTAssertEqual(
            StudioRegionGeometry.imagePoint(fromView: CGPoint(x: -50, y: 20), imageSize: imageSize, fitted: fitted),
            CGPoint(x: 0, y: 0)
        )
        XCTAssertEqual(
            StudioRegionGeometry.imagePoint(fromView: CGPoint(x: 900, y: 900), imageSize: imageSize, fitted: fitted),
            CGPoint(x: 1_000, y: 500)
        )
        // A drag from bottom-right to top-left is the same box.
        XCTAssertEqual(
            StudioRegionGeometry.imageRect(
                fromView: CGPoint(x: 120, y: 160), to: CGPoint(x: 40, y: 120), imageSize: imageSize, fitted: fitted
            ),
            CGRect(x: 100, y: 50, width: 200, height: 100)
        )
    }

    func testMovingKeepsTheShapeAndStaysInsideThePicture() {
        let imageSize = CGSize(width: 1_000, height: 500)
        let slid = cup.moved(by: CGVector(dx: -100, dy: 1_000), within: imageSize)
        XCTAssertEqual(slid.rect, CGRect(x: 0, y: 420, width: 120, height: 80))
        XCTAssertEqual(slid.id, cup.id)
        XCTAssertEqual(slid.label, "coffee cup")
        let point = handle.moved(by: CGVector(dx: 5_000, dy: -5_000), within: imageSize)
        XCTAssertEqual(point.point, CGPoint(x: 1_000, y: 0))
        XCTAssertTrue(point.isPositivePoint)
    }

    /// The anchor is captured once when the drag starts, so a corner dragged across the box and
    /// on keeps growing from the same fixed corner instead of collapsing on itself.
    func testResizingKeepsTheAnchorAcrossACrossingDrag() throws {
        let imageSize = CGSize(width: 1_000, height: 500)
        let anchor = try XCTUnwrap(cup.anchor(for: .bottomTrailing))
        XCTAssertEqual(anchor, CGPoint(x: 40, y: 30))
        let grown = cup.resizingBox(anchor: anchor, to: CGPoint(x: 300, y: 200), within: imageSize)
        XCTAssertEqual(grown.rect, CGRect(x: 40, y: 30, width: 260, height: 170))

        // Step one crosses the anchor; step two keeps going. Re-deriving the anchor from the
        // flipped box after step one would pin (20, 10) instead and shrink the box to 10×5.
        let crossed = cup.resizingBox(anchor: anchor, to: CGPoint(x: 20, y: 10), within: imageSize)
        XCTAssertEqual(crossed.rect, CGRect(x: 20, y: 10, width: 20, height: 20))
        let further = crossed.resizingBox(anchor: anchor, to: CGPoint(x: 10, y: 5), within: imageSize)
        XCTAssertEqual(further.rect, CGRect(x: 10, y: 5, width: 30, height: 25))
        XCTAssertEqual(further.id, cup.id)
        XCTAssertEqual(further.label, "coffee cup")

        XCTAssertEqual(handle.resizingBox(anchor: .zero, to: CGPoint(x: 5, y: 5), within: imageSize), handle)
        XCTAssertNil(handle.anchor(for: .topLeading))
    }

    /// Handles of the selected box win over the box, points win over boxes, and the prompt drawn
    /// last wins among overlapping boxes.
    func testHitTestingPrefersHandlesThenPointsThenTheNewestBox() {
        let imageSize = CGSize(width: 1_000, height: 500)
        let fitted = CGRect(x: 0, y: 0, width: 1_000, height: 500)
        let other = StudioRegionPrompt.box(CGRect(x: 100, y: 60, width: 200, height: 200))
        let inside = StudioRegionPrompt.point(CGPoint(x: 150, y: 100), isPositive: true)
        let prompts = [cup, other, inside]
        func hit(_ x: CGFloat, _ y: CGFloat, selected: UUID? = nil) -> StudioRegionHit? {
            StudioRegionHit.hit(
                in: prompts, at: CGPoint(x: x, y: y), imageSize: imageSize, fitted: fitted,
                selectedID: selected, handleRadius: 8, pointRadius: 10
            )
        }
        XCTAssertEqual(hit(150, 100), .point(id: inside.id))
        XCTAssertEqual(hit(120, 90), .box(id: other.id))
        XCTAssertEqual(hit(50, 40), .box(id: cup.id))
        XCTAssertEqual(hit(160, 110), .box(id: other.id), "the newer box wins where they overlap")
        XCTAssertEqual(hit(160, 110, selected: cup.id), .handle(id: cup.id, corner: .bottomTrailing))
        XCTAssertEqual(hit(41, 31, selected: cup.id), .handle(id: cup.id, corner: .topLeading))
        XCTAssertEqual(hit(41, 31, selected: other.id), .box(id: cup.id), "only the selected box offers handles")
        XCTAssertNil(hit(900, 400))
    }

    // MARK: - Find to Segment

    func testHandoffCarriesFindBoxesIntoSegmentAsDrawnPrompts() {
        let detections = [
            StudioAnalyzeDetection(id: 0, label: "coffee cup", confidence: 0.94, box: CGRect(x: 246, y: 307, width: 471, height: 451), maskURL: nil),
            StudioAnalyzeDetection(id: 1, label: "saucer", confidence: 0.81, box: CGRect(x: 82, y: 757, width: 184, height: 164), maskURL: nil),
        ]
        let toSegment = StudioAnalyzeHandoff.make(
            to: .visionSegment, inputPath: "/tmp/mug.png", prompt: "every coffee cup", detections: detections
        )
        XCTAssertEqual(toSegment.regionPrompts.map(\.label), ["coffee cup", "saucer"])
        XCTAssertEqual(toSegment.regionPrompts.map(\.rect), detections.map(\.box))

        var draft = StudioDraft()
        draft.reset(for: .segment)
        draft.visionRegionPrompts = [handle]
        toSegment.apply(to: &draft)
        XCTAssertEqual(draft.inputPath, "/tmp/mug.png")
        XCTAssertEqual(StudioRegionPromptText.boxText(draft.visionRegionPrompts ?? []), "246,307,717,758,coffee cup\n82,757,266,921,saucer")

        // Track cannot take the picture, so it cannot take boxes drawn in that picture's pixels.
        let toTrack = StudioAnalyzeHandoff.make(
            to: .visionTrack, inputPath: "/tmp/mug.png", prompt: "every coffee cup", detections: detections
        )
        XCTAssertEqual(toTrack.inputPath, "")
        XCTAssertTrue(toTrack.regionPrompts.isEmpty)
        var track = StudioDraft()
        track.reset(for: .track)
        track.visionRegionPrompts = [handle]
        toTrack.apply(to: &track)
        XCTAssertEqual(track.visionRegionPrompts, [handle], "prompts on Track's own clip are left alone")

        // Read takes the picture but draws nothing on it.
        let toRead = StudioAnalyzeHandoff.make(
            to: .visionRead, inputPath: "/tmp/mug.png", prompt: "", detections: detections
        )
        XCTAssertEqual(toRead.inputPath, "/tmp/mug.png")
        XCTAssertTrue(toRead.regionPrompts.isEmpty)
    }

    /// Every way an input arrives goes through `replaceInput`, and a new picture takes the old
    /// one's prompts and frames with it; the same path keeps them.
    func testReplacingTheInputClearsPromptsAndFramesDrawnOnThePreviousOne() throws {
        var draft = StudioDraft()
        draft.reset(for: .track)
        draft.inputPath = "/tmp/a.mp4"
        draft.visionRegionPrompts = [cup]
        draft.visionInitFrame = 12
        draft.visionEndFrame = 40

        draft.replaceInput("/tmp/a.mp4")
        XCTAssertEqual(draft.visionRegionPrompts, [cup])
        XCTAssertEqual(draft.visionInitFrame, 12)

        draft.replaceInput("/tmp/b.mp4")
        XCTAssertEqual(draft.inputPath, "/tmp/b.mp4")
        XCTAssertNil(draft.visionRegionPrompts)
        XCTAssertNil(draft.visionInitFrame)
        XCTAssertNil(draft.visionEndFrame)

        // The composer's well (a drop, a paste, a click to pick) attaches through the slot.
        let slot = try XCTUnwrap(StudioMode.segment.attachmentSlots.first)
        var segment = StudioDraft()
        segment.reset(for: .segment)
        segment.inputPath = "/tmp/mug.png"
        segment.visionRegionPrompts = [cup]
        slot.attach([URL(fileURLWithPath: "/tmp/mug.png")], to: &segment)
        XCTAssertEqual(segment.visionRegionPrompts, [cup], "re-attaching the same picture keeps the drawing")
        XCTAssertTrue(segment.attach(dropped: [URL(fileURLWithPath: "/tmp/other.png")], for: .segment))
        XCTAssertEqual(segment.inputPath, "/tmp/other.png")
        XCTAssertNil(segment.visionRegionPrompts)
        segment.visionRegionPrompts = [handle]
        slot.clear(in: &segment)
        XCTAssertEqual(segment.inputPath, "")
        XCTAssertNil(segment.visionRegionPrompts)
    }

    /// The Command view's `--box`, `--point`, `--init-frame`, and `--end-frame` flow back into the
    /// drawing through the binding table, so the two never disagree and the validation that
    /// accepts a drawn prompt accepts a typed one.
    func testCommandViewOverridesFlowBackIntoTheDrawing() throws {
        let bindings = StudioContractBindings.bindings(for: .track)
        var draft = StudioDraft()
        draft.reset(for: .track)
        draft.prompt = ""
        draft.inputPath = "/tmp/clip.mp4"
        draft.visionRegionPrompts = [handle]

        try XCTUnwrap(bindings["--box"]).write(&draft, .text("40,30,160,110,coffee cup\n1,2,3"))
        XCTAssertEqual(draft.visionRegionPrompts?.map(\.isBox), [true, false], "boxes replaced, points kept")
        XCTAssertEqual(draft.visionRegionPrompts?.first?.label, "coffee cup")
        XCTAssertEqual(try XCTUnwrap(bindings["--box"]).read(draft), .text("40,30,160,110,coffee cup"))

        try XCTUnwrap(bindings["--point"]).write(&draft, .text("12,18,negative,shadow"))
        XCTAssertEqual(StudioRegionPromptText.pointText(draft.visionRegionPrompts ?? []), "12,18,negative,shadow")
        XCTAssertEqual(draft.visionRegionPrompts?.boxes.count, 1)

        try XCTUnwrap(bindings["--init-frame"]).write(&draft, .integer(12))
        XCTAssertEqual(draft.visionInitFrame, 12)
        try XCTUnwrap(bindings["--init-frame"]).write(&draft, .integer(0))
        XCTAssertNil(draft.visionInitFrame, "0 is the CLI's default and reads as unset")
        try XCTUnwrap(bindings["--end-frame"]).write(&draft, .integer(40))
        XCTAssertEqual(draft.visionEndFrame, 40)
        XCTAssertEqual(try XCTUnwrap(bindings["--end-frame"]).read(draft), .integer(40))
        try XCTUnwrap(bindings["--end-frame"]).write(&draft, .unset)
        XCTAssertNil(draft.visionEndFrame)

        // A prompt typed in the Command view satisfies the run's validation like a drawn one.
        XCTAssertNoThrow(try StudioCommandAdapter.makeRequest(mode: .track, draft: draft))
        try XCTUnwrap(bindings["--box"]).write(&draft, .text(""))
        try XCTUnwrap(bindings["--point"]).write(&draft, .text(""))
        XCTAssertNil(draft.visionRegionPrompts)
        XCTAssertThrowsError(try StudioCommandAdapter.makeRequest(mode: .track, draft: draft))
    }

    func testHandoffWithoutDetectionsClearsPromptsDrawnOnThePreviousPicture() {
        var draft = StudioDraft()
        draft.reset(for: .segment)
        draft.visionRegionPrompts = [cup]
        StudioAnalyzeHandoff.make(to: .visionSegment, inputPath: "/tmp/other.png", prompt: "the saucer")
            .apply(to: &draft)
        XCTAssertNil(draft.visionRegionPrompts)
    }

    // MARK: - Into the command

    func testDrawnPromptsBecomeTheSegmentAndTrackCommandsBoxAndPointFlags() throws {
        var segment = StudioDraft()
        segment.reset(for: .segment)
        segment.prompt = ""
        segment.inputPath = "/tmp/mug.png"
        segment.visionRegionPrompts = [cup, handle, shadow]
        let request = try StudioCommandAdapter.makeRequest(mode: .segment, draft: segment)
        XCTAssertEqual(request.draft.visionBoxPrompts, "40,30,160,110,coffee cup")
        XCTAssertEqual(request.draft.visionPointPrompts, "400,260,positive\n12,18,negative,shadow")
        let arguments = request.template.arguments(from: request.draft)
        XCTAssertTrue(arguments.contains("--box"))
        XCTAssertTrue(arguments.contains("40,30,160,110,coffee cup"))
        XCTAssertEqual(arguments.filter { $0 == "--point" }.count, 2)

        segment.visionRegionPrompts = nil
        XCTAssertThrowsError(try StudioCommandAdapter.makeRequest(mode: .segment, draft: segment)) { error in
            XCTAssertEqual(error as? StudioCommandError, .missingPrompt("A prompt or a drawn box or point"))
        }

        var track = StudioDraft()
        track.reset(for: .track)
        track.prompt = "the skater"
        track.inputPath = "/tmp/clip.mp4"
        track.visionRegionPrompts = [cup]
        track.visionInitFrame = 12
        track.visionEndFrame = 200
        let tracked = try StudioCommandAdapter.makeRequest(mode: .track, draft: track)
        XCTAssertEqual(tracked.draft.visionInitFrame, 12)
        XCTAssertEqual(tracked.draft.visionEndFrame, "200")
        XCTAssertEqual(tracked.draft.visionBoxPrompts, "40,30,160,110,coffee cup")
        let trackArguments = tracked.template.arguments(from: tracked.draft)
        XCTAssertEqual(trackArguments.firstIndex(of: "--init-frame").map { trackArguments[$0 + 1] }, "12")
        XCTAssertEqual(trackArguments.firstIndex(of: "--end-frame").map { trackArguments[$0 + 1] }, "200")

        track.visionEndFrame = 3
        XCTAssertThrowsError(try StudioCommandAdapter.makeRequest(mode: .track, draft: track))

        var find = StudioDraft()
        find.reset(for: .findObjects)
        find.prompt = "cups"
        find.inputPath = "/tmp/mug.png"
        find.visionRegionPrompts = [cup]
        let found = try StudioCommandAdapter.makeRequest(mode: .findObjects, draft: find)
        XCTAssertEqual(found.draft.visionBoxPrompts, "", "Find takes no box prompts")
    }

    func testSavedDraftsFromBeforeDrawnPromptsStillDecode() throws {
        var draft = StudioDraft()
        draft.reset(for: .segment)
        draft.visionRegionPrompts = [cup]
        draft.visionInitFrame = 4
        let encoded = try JSONEncoder().encode(draft)
        var json = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        XCTAssertTrue(json.contains("visionRegionPrompts"))
        json = json.replacingOccurrences(of: "\"visionRegionPrompts\"", with: "\"legacyRegionPrompts\"")
        let decoded = try JSONDecoder().decode(StudioDraft.self, from: Data(json.utf8))
        XCTAssertNil(decoded.visionRegionPrompts)
        XCTAssertEqual(decoded.visionInitFrame, 4)
    }

    // MARK: - Subjects selectors

    /// Video ▸ Subjects keeps one box as `x1,y1,x2,y2` and point lists as `x,y; x,y`.
    func testSubjectSelectorTextRoundTrips() {
        let prompts = StudioSubjectSelectorText.prompts(
            box: "500,200,600,330", positivePoints: "612.4,479.6; 10,10", negativePoints: ""
        )
        XCTAssertEqual(prompts.count, 3)
        XCTAssertEqual(prompts[0].rect, CGRect(x: 500, y: 200, width: 100, height: 130))
        XCTAssertEqual(prompts.filter(\.isPositivePoint).count, 2)
        XCTAssertEqual(StudioSubjectSelectorText.boxText(prompts), "500,200,600,330")
        XCTAssertEqual(StudioSubjectSelectorText.positivePointsText(prompts), "612,480; 10,10")
        XCTAssertEqual(StudioSubjectSelectorText.negativePointsText(prompts), "")
        XCTAssertEqual(StudioSubjectSelectorText.negativePointsText([shadow]), "12,18")
        XCTAssertTrue(StudioSubjectSelectorText.prompts(box: "1,2,3", positivePoints: "1,2,3", negativePoints: "x").isEmpty)
        XCTAssertEqual(StudioSubjectSelectorText.prompts(box: "", positivePoints: "1,2\n3,4", negativePoints: "").count, 2)
    }

    /// `MediaImageIO.centerCropped` scales by the larger ratio and crops the overflow evenly, so a
    /// 1920×1080 frame fills an 832×480 plan canvas as 853×480 shifted 10 pixels left.
    func testSubjectFrameCoverRectMatchesTheCLIsCenterCrop() {
        let rect = StudioSubjectFrameGeometry.coverRect(
            source: CGSize(width: 1_920, height: 1_080), canvas: CGSize(width: 832, height: 480)
        )
        XCTAssertEqual(rect, CGRect(x: -10, y: 0, width: 853, height: 480))
        let portrait = StudioSubjectFrameGeometry.coverRect(
            source: CGSize(width: 480, height: 960), canvas: CGSize(width: 832, height: 480)
        )
        XCTAssertEqual(portrait, CGRect(x: 0, y: -592, width: 832, height: 1_664))
        XCTAssertEqual(StudioSubjectFrameGeometry.coverRect(source: .zero, canvas: CGSize(width: 1, height: 1)), .zero)
    }

    // MARK: - Frames

    /// `AppleMediaVideoIO.extractFrames`: nominal rate or 30, floored at 1, and
    /// `(duration × fps).rounded(.toNearestOrEven)` frames.
    func testVideoFrameGridCountsFramesTheWayTheTrackerDoes() {
        let grid = StudioVideoFrameGrid(duration: 10, frameRate: 24)
        XCTAssertEqual(grid.frameCount, 240)
        XCTAssertEqual(grid.lastFrame, 239)
        XCTAssertEqual(grid.time(ofFrame: 3), 3.5 / 24, accuracy: 1e-9)
        XCTAssertEqual(grid.clamped(-4), 0)
        XCTAssertEqual(grid.clamped(1_000), 239)
        XCTAssertEqual(grid.timeDescription(ofFrame: 36), "0:01.5")
        XCTAssertEqual(grid.timeDescription(ofFrame: 239), "0:09.9")

        XCTAssertEqual(StudioVideoFrameGrid(duration: 2.5, frameRate: 29.97).frameCount, 75, "74.925 rounds to nearest")
        XCTAssertEqual(StudioVideoFrameGrid(duration: 2.5, frameRate: 29).frameCount, 72, "72.5 rounds to even")
        XCTAssertEqual(StudioVideoFrameGrid(duration: 10, frameRate: 0.25).frameRate, 1, "floored at 1 fps")

        let unknown = StudioVideoFrameGrid(duration: 2, frameRate: 0)
        XCTAssertEqual(unknown.frameRate, 30)
        XCTAssertEqual(unknown.frameCount, 60)
        XCTAssertEqual(StudioVideoFrameGrid(duration: 0, frameRate: 0).frameCount, 1)
        XCTAssertEqual(StudioVideoFrameGrid(duration: .nan, frameRate: 30).frameCount, 1)
        XCTAssertEqual(StudioVideoFrameGrid(duration: 1, frameRate: .nan).frameRate, 30)
    }

    /// `SCAIL2MaskPreparer` takes source frame `round(t × sourceFPS)` for plan time `t`.
    func testSubjectPlanTimeSnapsToTheSourceFrameGrid() {
        // Plan frame 1 at 24 fps (t = 1/24) on a 30 fps clip is source frame round(1.25) = 1.
        XCTAssertEqual(StudioVideoFrameGrid.sourceTime(forPlanTime: 1.0 / 24, sourceFrameRate: 30), 1.5 / 30, accuracy: 1e-9)
        // Plan frame 3 (t = 0.125) is source frame round(3.75) = 4.
        XCTAssertEqual(StudioVideoFrameGrid.sourceTime(forPlanTime: 0.125, sourceFrameRate: 30), 4.5 / 30, accuracy: 1e-9)
        XCTAssertEqual(StudioVideoFrameGrid.sourceTime(forPlanTime: 0, sourceFrameRate: 0), 0.5 / 30, accuracy: 1e-9)
    }
}
