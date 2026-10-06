import Foundation
import XCTest
@testable import MereRunCore

final class EmbeddingGemma2MediaInputTests: XCTestCase {
    func testOrderedContentResolvesRelativeFiles() throws {
        let json = #"{"inputs":[{"content":[{"type":"text","text":"before"},{"type":"image","path":"photo.png"},{"type":"audio","path":"/clips/sound.wav"},{"type":"video-frames","frames":["first.png","second.png"]},{"type":"text","text":"after"}]}]}"#
        let inputs = try JSONDecoder().decode(EmbeddingGemma2InputDocument.self, from: Data(json.utf8)).resolved(relativeTo: URL(fileURLWithPath: "/corpus"))
        XCTAssertEqual(inputs.count, 1)
        guard case .text("before") = inputs[0].content[0],
              case .image(let image) = inputs[0].content[1],
              case .audio(let audio) = inputs[0].content[2],
              case .videoFrames(let frames) = inputs[0].content[3],
              case .text("after") = inputs[0].content[4] else { return XCTFail("Content order changed") }
        XCTAssertEqual(image.path, "/corpus/photo.png")
        XCTAssertEqual(audio.path, "/clips/sound.wav")
        XCTAssertEqual(frames.map(\.path), ["/corpus/first.png", "/corpus/second.png"])
    }

    func testMalformedEmptyAndRemoteInputsAreRejected() {
        for json in [#"{"inputs":[]}"#, #"{"inputs":[{"content":[{"type":"video-frames","frames":[]}]}]}"#, #"{"inputs":[{"content":[]}]}"#,
                     #"{"inputs":[{"content":[{"type":"image","path":"https://example.com/a.png"}]}]}"#,
                     #"{"inputs":[{"content":[{"type":"audio","path":""}]}]}"#,
                     #"{"inputs":[{"content":[{"type":"other","path":"x"}]}]}"#] {
            XCTAssertThrowsError(try JSONDecoder().decode(EmbeddingGemma2InputDocument.self, from: Data(json.utf8)).resolved(relativeTo: URL(fileURLWithPath: "/")))
        }
    }
}
