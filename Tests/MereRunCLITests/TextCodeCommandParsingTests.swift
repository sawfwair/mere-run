import XCTest
@testable import MereRunCLI

final class TextCodeCommandParsingTests: XCTestCase {
    func testCodeGenerationHidesReasoningUnlessRequested() throws {
        let defaultCommand = try TextCode.parse(["--prompt", "Return a Swift function."])
        XCTAssertFalse(defaultCommand.makeRequest().showThinking)

        let reasoningCommand = try TextCode.parse(["--prompt", "Return a Swift function.", "--thinking"])
        XCTAssertTrue(reasoningCommand.makeRequest().showThinking)
    }
}
