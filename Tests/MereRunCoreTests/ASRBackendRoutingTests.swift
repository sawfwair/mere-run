import XCTest
@testable import MereRunCore
import AudioCore

final class ASRBackendRoutingTests: XCTestCase {
    func testTranslateAlwaysPrefersQwen() {
        let decision = ASRBackendRouting.select(
            task: .translate,
            languageHint: "en",
            preferredBackend: .auto,
            availableBackends: ASRBackendAvailability(parakeetAvailable: true, qwenAvailable: true)
        )

        XCTAssertEqual(decision.backend, .qwen)
        XCTAssertEqual(decision.reason, "translate_requires_qwen")
    }

    func testTranslateDoesNotFallbackToParakeetWhenQwenUnavailable() {
        let decision = ASRBackendRouting.select(
            task: .translate,
            languageHint: "en",
            preferredBackend: .auto,
            availableBackends: ASRBackendAvailability(parakeetAvailable: true, qwenAvailable: false)
        )

        XCTAssertEqual(decision.backend, .qwen)
        XCTAssertEqual(decision.reason, "translate_requires_qwen_preferred_unavailable")
    }

    func testUnsupportedLanguageHintRoutesToQwen() {
        let decision = ASRBackendRouting.select(
            task: .transcribe,
            languageHint: "en",
            preferredBackend: .auto,
            availableBackends: ASRBackendAvailability(parakeetAvailable: true, qwenAvailable: true),
            parakeetSupportedLanguageCodes: ["fr"]
        )

        XCTAssertEqual(decision.backend, .qwen)
        XCTAssertEqual(decision.reason, "unsupported_language_for_parakeet")
    }

    func testUnknownExplicitLanguageHintRoutesToQwen() {
        let decision = ASRBackendRouting.select(
            task: .transcribe,
            languageHint: "klingon-ish",
            preferredBackend: .auto,
            availableBackends: ASRBackendAvailability(parakeetAvailable: true, qwenAvailable: true)
        )

        XCTAssertEqual(decision.backend, .qwen)
        XCTAssertEqual(decision.reason, "unknown_explicit_language_hint")
    }

    func testAutoTranscriptionPrefersParakeet() {
        let decision = ASRBackendRouting.select(
            task: .transcribe,
            languageHint: nil,
            preferredBackend: .auto,
            availableBackends: ASRBackendAvailability(parakeetAvailable: true, qwenAvailable: true)
        )

        XCTAssertEqual(decision.backend, .parakeet)
        XCTAssertEqual(decision.reason, "auto_prefers_parakeet_for_transcription")
    }

    func testPreferredQwenFallsBackToParakeet() {
        let decision = ASRBackendRouting.select(
            task: .transcribe,
            languageHint: nil,
            preferredBackend: .qwen,
            availableBackends: ASRBackendAvailability(parakeetAvailable: true, qwenAvailable: false)
        )

        XCTAssertEqual(decision.backend, .parakeet)
        XCTAssertEqual(decision.reason, "preferred_qwen_fallback_to_parakeet")
    }

    func testNormalizeLanguageHintSupportsTagsAndNames() {
        XCTAssertEqual(ASRBackendRouting.normalizeLanguageHint("<|EN|>"), "en")
        XCTAssertEqual(ASRBackendRouting.normalizeLanguageHint("English"), "en")
        XCTAssertEqual(ASRBackendRouting.normalizeLanguageHint("zh-CN"), "zh")
        XCTAssertNil(ASRBackendRouting.normalizeLanguageHint("auto"))
    }
}
