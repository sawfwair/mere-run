import XCTest
@testable import MereRunCLI

final class TextChatCommandParsingTests: XCTestCase {
    func testTextChatDefaultsToNonStreamingCLIOutput() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Say hello",
        ])

        XCTAssertFalse(cmd.stream)
        XCTAssertEqual(cmd.prompt, "Say hello")
    }

    func testTextChatParsesStreamingFlag() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Stream this",
            "--stream",
        ])

        XCTAssertTrue(cmd.stream)
    }
}
