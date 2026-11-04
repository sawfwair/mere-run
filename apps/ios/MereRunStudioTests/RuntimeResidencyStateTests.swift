import XCTest

final class RuntimeResidencyStateTests: XCTestCase {
    func testReleaseWaitsForActiveInferenceThenBlocksNewWork() throws {
        var state = RuntimeResidencyState()

        let chat = try XCTUnwrap(state.begin(.chatting, modelID: "chat"))
        XCTAssertFalse(state.requestRelease())
        XCTAssertEqual(state.activity, .chatting)
        XCTAssertTrue(state.releasePending)

        XCTAssertTrue(state.completeActivity(chat))
        XCTAssertEqual(state.activity, .releasing)
        XCTAssertNil(state.begin(.generatingImage, modelID: "image"))

        state.completeRelease()
        XCTAssertEqual(state.activity, .idle)
        XCTAssertNotNil(state.begin(.generatingImage, modelID: "image"))
    }

    func testIdleReleaseStartsImmediately() {
        var state = RuntimeResidencyState()

        XCTAssertTrue(state.requestRelease())
        XCTAssertEqual(state.activity, .releasing)
        XCTAssertFalse(state.releasePending)
        XCTAssertFalse(state.requestRelease())
        XCTAssertFalse(state.releasePending)
    }

    func testMatchingChatClaimsWarmupAndReusesPreparedModel() throws {
        var state = RuntimeResidencyState()
        let warmup = try XCTUnwrap(state.beginWarmup(modelID: "chat-a"))
        XCTAssertFalse(state.isIdle)
        XCTAssertNil(state.beginWarmup(modelID: "chat-a"))
        XCTAssertNil(state.begin(.chatting, modelID: "chat-b"))
        let chat = try XCTUnwrap(state.begin(.chatting, modelID: "chat-a"))

        XCTAssertTrue(state.completeWarmup(warmup, succeeded: true))
        XCTAssertEqual(state.activity, .chatting)
        XCTAssertEqual(state.warmChatModelID, "chat-a")
        XCTAssertFalse(state.completeActivity(chat))
        XCTAssertTrue(state.isIdle)
        XCTAssertNil(state.beginWarmup(modelID: "chat-a"), "prepared weights need no second warmup")
    }

    func testImageAdmissionInvalidatesWarmupPublication() throws {
        var state = RuntimeResidencyState()
        let warmup = try XCTUnwrap(state.beginWarmup(modelID: "chat"))
        let image = try XCTUnwrap(state.begin(.generatingImage, modelID: "image"))

        XCTAssertFalse(state.completeWarmup(warmup, succeeded: true))
        XCTAssertNil(state.warmChatModelID)
        XCTAssertTrue(state.isCurrent(image))
        XCTAssertFalse(state.completeActivity(image))
        XCTAssertTrue(state.isIdle)
    }

    func testReleaseInvalidatesWarmupAndBlocksNewAdmissionsUntilUnloaded() throws {
        var state = RuntimeResidencyState()
        let old = try XCTUnwrap(state.beginWarmup(modelID: "chat-a"))
        XCTAssertTrue(state.requestRelease())
        XCTAssertFalse(state.completeWarmup(old, succeeded: true))
        XCTAssertNil(state.warmChatModelID)
        XCTAssertNil(state.beginWarmup(modelID: "chat-b"))
        XCTAssertNil(state.begin(.chatting, modelID: "chat-b"))

        state.completeRelease()
        let current = try XCTUnwrap(state.beginWarmup(modelID: "chat-b"))
        XCTAssertFalse(state.completeWarmup(old, succeeded: true))
        XCTAssertTrue(state.completeWarmup(current, succeeded: true))
        XCTAssertEqual(state.warmChatModelID, "chat-b")
    }

    func testReleaseDuringChatWarmupKeepsTheChatLeaseButRejectsReadiness() throws {
        var state = RuntimeResidencyState()
        let warmup = try XCTUnwrap(state.beginWarmup(modelID: "chat-a"))
        let chat = try XCTUnwrap(state.begin(.chatting, modelID: "chat-a"))
        XCTAssertFalse(state.requestRelease())
        XCTAssertTrue(state.isCurrent(chat))
        XCTAssertTrue(state.releasePending)
        XCTAssertFalse(state.completeWarmup(warmup, succeeded: true))
        state.recordChatPrepared(chat)
        XCTAssertNil(state.warmChatModelID)
        XCTAssertTrue(state.completeActivity(chat))
        XCTAssertEqual(state.activity, .releasing)
    }

    func testOldActivityCannotFinishOrPublishIntoANewerRequest() throws {
        var state = RuntimeResidencyState()
        let old = try XCTUnwrap(state.begin(.chatting, modelID: "chat"))
        XCTAssertFalse(state.completeActivity(old))
        let current = try XCTUnwrap(state.begin(.chatting, modelID: "chat"))

        XCTAssertFalse(state.isCurrent(old), "model identity alone does not identify a chat turn")
        XCTAssertFalse(state.completeActivity(old))
        state.recordChatPrepared(old)
        XCTAssertNil(state.warmChatModelID)
        XCTAssertTrue(state.isCurrent(current))
        state.recordChatPrepared(current)
        XCTAssertEqual(state.warmChatModelID, "chat")
    }

    func testFailedWarmupCanRetryWithoutPublishingPreparedWeights() throws {
        var state = RuntimeResidencyState()
        let old = try XCTUnwrap(state.beginWarmup(modelID: "chat"))
        XCTAssertTrue(state.completeWarmup(old, succeeded: false))
        XCTAssertTrue(state.isIdle)
        XCTAssertNil(state.warmChatModelID)
        let current = try XCTUnwrap(state.beginWarmup(modelID: "chat"))
        XCTAssertFalse(state.completeWarmup(old, succeeded: true))
        XCTAssertTrue(state.completeWarmup(current, succeeded: true))
    }

    func testDeletionCanReserveReleaseDuringWarmupButNotDuringInference() throws {
        var state = RuntimeResidencyState()
        let warmup = try XCTUnwrap(state.beginWarmup(modelID: "chat"))
        XCTAssertTrue(state.beginRelease())
        XCTAssertFalse(state.completeWarmup(warmup, succeeded: true))
        state.completeRelease()
        _ = try XCTUnwrap(state.begin(.generatingImage, modelID: "image"))
        XCTAssertFalse(state.beginRelease())
        state.completeRelease()
        XCTAssertEqual(state.activity, .generatingImage)
    }
}
