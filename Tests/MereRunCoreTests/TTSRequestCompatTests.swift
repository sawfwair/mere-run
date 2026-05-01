import XCTest
@testable import MereRunCore
import AudioCore

final class TTSRequestCompatTests: XCTestCase {
    func testLegacyStyleInitializerDefaultsRemainCompatible() {
        let outputURL = URL(fileURLWithPath: "/tmp/legacy.wav")
        let request = TTSRequest(
            text: "Hello",
            voiceDescription: "A calm narrator",
            speed: 1.0,
            temperature: 0.6,
            outputURL: outputURL
        )

        XCTAssertEqual(request.text, "Hello")
        XCTAssertEqual(request.voiceDescription, "A calm narrator")
        XCTAssertEqual(request.voiceMode, .style)
        XCTAssertNil(request.cloneReference)
        XCTAssertEqual(request.language, "auto")
        XCTAssertEqual(request.outputURL, outputURL)
    }

    func testCloneInitializerStoresReference() {
        let outputURL = URL(fileURLWithPath: "/tmp/clone.wav")
        let reference = TTSCloneReference(
            audioURL: URL(fileURLWithPath: "/tmp/ref.wav"),
            transcript: "Reference transcript",
            language: "en"
        )

        let request = TTSRequest(
            text: "Speak this",
            voiceDescription: "ignored in clone mode",
            voiceMode: .clone,
            cloneReference: reference,
            language: "en",
            outputURL: outputURL
        )

        XCTAssertEqual(request.voiceMode, .clone)
        XCTAssertEqual(request.cloneReference?.audioURL.path, "/tmp/ref.wav")
        XCTAssertEqual(request.cloneReference?.transcript, "Reference transcript")
        XCTAssertEqual(request.language, "en")
    }
}
