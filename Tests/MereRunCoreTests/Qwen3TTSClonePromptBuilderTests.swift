import XCTest
@testable import MereRunCore
import AudioCore

final class Qwen3TTSClonePromptBuilderTests: XCTestCase {
    func testCloneStagesAreExposed() {
        XCTAssertEqual(TTSStage.preprocessingReference.rawValue, "preprocessingReference")
        XCTAssertEqual(TTSStage.encodingReference.rawValue, "encodingReference")
        XCTAssertEqual(TTSStage.buildingPrompt.rawValue, "buildingPrompt")
    }

    func testCloneReferenceCarriesTranscriptAndLanguage() {
        let reference = TTSCloneReference(
            audioURL: URL(fileURLWithPath: "/tmp/ref.wav"),
            transcript: "Reference words",
            language: "en"
        )

        let request = TTSRequest(
            text: "Target text",
            voiceMode: .clone,
            cloneReference: reference,
            language: "en",
            outputURL: URL(fileURLWithPath: "/tmp/out.wav")
        )

        XCTAssertEqual(request.voiceMode, .clone)
        XCTAssertEqual(request.cloneReference?.transcript, "Reference words")
        XCTAssertEqual(request.cloneReference?.language, "en")
    }
}
