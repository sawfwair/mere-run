import Foundation
@testable import StudioKit
@testable import StudioUI
import XCTest

final class StudioConversationAnnouncementTests: XCTestCase {
    func testCleanExitAfterCancellationDoesNotAnnounceSuccess() {
        XCTAssertEqual(StudioConversationAnnouncement.completion(for: .cancelled(exit: 0, at: .distantPast)), "Reply stopped.")
        XCTAssertEqual(StudioConversationAnnouncement.completion(for: .cancelled(exit: 15, at: .distantPast)), "Reply stopped.")
    }

    func testSuccessAndFailureHaveDistinctAnnouncements() {
        XCTAssertEqual(StudioConversationAnnouncement.completion(for: .finished(exit: 0, at: .distantPast)), "Reply ready.")
        XCTAssertEqual(StudioConversationAnnouncement.completion(for: .finished(exit: 1, at: .distantPast)),
                       "Reply failed. Review the conversation for details.")
    }

    func testActiveJobsDoNotAnnounceCompletion() {
        XCTAssertNil(StudioConversationAnnouncement.completion(for: .queued))
        XCTAssertNil(StudioConversationAnnouncement.completion(for: .running(since: .distantPast)))
    }
}
