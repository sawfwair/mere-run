@testable import MereRunApp
import XCTest

final class StudioTypesTests: XCTestCase {
    func testStudioModesMapToPublicTemplates() {
        XCTAssertEqual(StudioMode.createImage.defaultTemplateID, .imageGenerate)
        XCTAssertEqual(StudioMode.chat.defaultTemplateID, .textChat)
        XCTAssertEqual(StudioMode.code.defaultTemplateID, .textCode)
        XCTAssertEqual(StudioMode.speak.defaultTemplateID, .speechSynthesize)
        XCTAssertEqual(StudioMode.listen.defaultTemplateID, .speechTranscribe)
        XCTAssertEqual(StudioMode.readImage.defaultTemplateID, .visionInspect)
        XCTAssertEqual(StudioMode.findObjects.defaultTemplateID, .visionGround)
        XCTAssertEqual(StudioMode.segment.defaultTemplateID, .visionSegment)
        XCTAssertEqual(StudioMode.track.defaultTemplateID, .visionTrack)
        XCTAssertEqual(StudioMode.music.defaultTemplateID, .musicGenerate)
        XCTAssertEqual(StudioMode.video.defaultTemplateID, .videoGenerate)
    }

    func testStudioRunRequestBuildsImageCommandWithDefaultOutput() throws {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "a porcelain teapot on linen"
        draft.secondaryText = "blurry"
        draft.width = 768
        draft.height = 512
        draft.steps = 8

        let request = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: draft)

        XCTAssertEqual(request.templateID, .imageGenerate)
        XCTAssertEqual(request.draft.prompt, "a porcelain teapot on linen")
        XCTAssertEqual(request.draft.secondaryText, "blurry")
        XCTAssertEqual(request.draft.width, 768)
        XCTAssertEqual(request.draft.height, 512)
        XCTAssertEqual(request.draft.steps, 8)
        XCTAssertEqual(request.expectedOutputURL?.pathExtension, "png")
    }

    func testReadImageVariantsResolveToDistinctTemplates() throws {
        var draft = StudioDraft()
        draft.reset(for: .readImage)
        draft.inputPath = "/tmp/image.png"
        draft.prompt = "What text is visible?"

        draft.readImageAction = .inspect
        XCTAssertEqual(try StudioCommandAdapter.makeRequest(mode: .readImage, draft: draft).templateID, .visionInspect)

        draft.readImageAction = .ocr
        XCTAssertEqual(try StudioCommandAdapter.makeRequest(mode: .readImage, draft: draft).templateID, .visionOCR)

        draft.readImageAction = .caption
        XCTAssertEqual(try StudioCommandAdapter.makeRequest(mode: .readImage, draft: draft).templateID, .visionCaption)
    }

    func testMissingInputValidationIsFriendly() {
        var draft = StudioDraft()
        draft.reset(for: .listen)

        XCTAssertThrowsError(try StudioCommandAdapter.makeRequest(mode: .listen, draft: draft)) { error in
            XCTAssertEqual(error as? StudioCommandError, .missingInput("audio"))
        }
    }

    func testModelReadinessParserFindsInstalledMissingAndUnknown() {
        let output = """
        ID Category Status Size
        image-zimage-nano image installed 4 GB
        video-ltx-av media missing 12 GB
        vision-segment-sam31 vision unsupported 7 GB
        """

        XCTAssertEqual(
            ModelReadinessParser.state(for: "image-zimage-nano", modelListOutput: output),
            .ready
        )
        XCTAssertEqual(
            ModelReadinessParser.state(for: "video-ltx-av", modelListOutput: output),
            .missingModel("video-ltx-av")
        )
        XCTAssertEqual(
            ModelReadinessParser.state(for: "vision-segment-sam31", modelListOutput: output),
            .unsupported("vision-segment-sam31 is listed as unsupported on this Mac.")
        )

        if case .unknown = ModelReadinessParser.state(for: "missing", modelListOutput: output) {
            // expected
        } else {
            XCTFail("Expected unknown readiness for a model absent from the list.")
        }
    }

    func testAPIKeyPreviewMasking() {
        let args = ["api", "serve", "--api-key", "secret-token", "--port", "8080"]
        XCTAssertEqual(
            args.maskingSecrets(),
            ["api", "serve", "--api-key", "••••••••", "--port", "8080"]
        )
    }
}
