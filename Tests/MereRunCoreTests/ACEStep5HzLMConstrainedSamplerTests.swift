import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class ACEStep5HzLMConstrainedSamplerTests: MereRunCoreTestCase {

    func testUserMetadataInjectionWhitelistsNextToken() throws {
        let env = ProcessInfo.processInfo.environment
        guard let root = env["MERERUN_TEST_ACESTEP_5HZ_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_5HZ_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-5Hz-lm-1.7B to run this test.")
        }

        let resources = ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: root))
        let configData = try Data(contentsOf: resources.configURL)
        let modelConfig = try JSONDecoder().decode(ACEStep5HzLMConfig.self, from: configData)
        let tokenizer = try ACEStep5HzLMTokenizer.load(from: resources.modelRootURL)

        let sampler = ACEStep5HzLMConstrainedSampler(
            tokenizer: tokenizer,
            vocabSize: modelConfig.vocabSize,
            enabled: true,
            debug: false,
            skipCaption: true,
            skipLanguage: true,
            stopAtReasoning: false,
            targetDurationSeconds: nil,
            userMetadata: .init(bpm: "120")
        )

        func feed(_ text: String) {
            for id in tokenizer.encode(text, addSpecialTokens: false) {
                sampler.update(with: id)
            }
        }

        feed("<think>")
        feed("\n")
        feed("bpm:")
        XCTAssertEqual(sampler.state, .bpmValue)

        let logits = MLXArray.zeros([modelConfig.vocabSize], dtype: .float32)
        let expected = tokenizer.encode(" 120\n", addSpecialTokens: false)
        let first = try XCTUnwrap(expected.first)

        let masked = sampler.processLogits(logits, tokens: [])
        XCTAssertTrue(masked[first].item(Float.self).isFinite)

        let other = (first == 0) ? 1 : 0
        XCTAssertEqual(masked[other].item(Float.self), -Float.infinity)
    }

    func testCodesGenerationBlocksEOSUntilTargetCodes() throws {
        let env = ProcessInfo.processInfo.environment
        guard let root = env["MERERUN_TEST_ACESTEP_5HZ_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_5HZ_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-5Hz-lm-1.7B to run this test.")
        }

        let resources = ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: root))
        let configData = try Data(contentsOf: resources.configURL)
        let modelConfig = try JSONDecoder().decode(ACEStep5HzLMConfig.self, from: configData)
        let tokenizer = try ACEStep5HzLMTokenizer.load(from: resources.modelRootURL)

        let sampler = ACEStep5HzLMConstrainedSampler(
            tokenizer: tokenizer,
            vocabSize: modelConfig.vocabSize,
            enabled: true,
            debug: false,
            skipCaption: true,
            skipLanguage: true,
            stopAtReasoning: false,
            targetDurationSeconds: 1.0,
            userMetadata: .init()
        )

        func feed(_ text: String) {
            for id in tokenizer.encode(text, addSpecialTokens: false) {
                sampler.update(with: id)
            }
        }

        feed("<think>")
        feed("\n")
        feed("bpm:")
        feed(" 120\n")
        feed("duration:")
        feed(" 60\n")
        feed("keyscale:")
        feed(" C major\n")
        feed("timesignature:")
        feed(" 4\n")
        feed("</think>")

        XCTAssertEqual(sampler.state, .codesGeneration)

        let eos = try XCTUnwrap(tokenizer.eosTokenId)
        let audio0 = try XCTUnwrap(tokenizer.convertTokenToId("<|audio_code_0|>"))

        let logits = MLXArray.zeros([modelConfig.vocabSize], dtype: .float32)

        let masked0 = sampler.processLogits(logits, tokens: [])
        XCTAssertEqual(masked0[eos].item(Float.self), -Float.infinity)
        XCTAssertTrue(masked0[audio0].item(Float.self).isFinite)

        for _ in 0..<5 {
            sampler.update(with: audio0)
        }

        let maskedAfter = sampler.processLogits(logits, tokens: [])
        XCTAssertTrue(maskedAfter[eos].item(Float.self).isFinite)
        XCTAssertEqual(maskedAfter[audio0].item(Float.self), -Float.infinity)

        sampler.update(with: eos)
        XCTAssertEqual(sampler.state, .completed)
    }

    func testUnderstandPhaseBlocksAudioCodesAfterReasoning() throws {
        let env = ProcessInfo.processInfo.environment
        guard let root = env["MERERUN_TEST_ACESTEP_5HZ_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_5HZ_ROOT=/path/to/ACE-Step-1.5/checkpoints/acestep-5Hz-lm-1.7B to run this test.")
        }

        let resources = ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: root))
        let configData = try Data(contentsOf: resources.configURL)
        let modelConfig = try JSONDecoder().decode(ACEStep5HzLMConfig.self, from: configData)
        let tokenizer = try ACEStep5HzLMTokenizer.load(from: resources.modelRootURL)

        let sampler = ACEStep5HzLMConstrainedSampler(
            tokenizer: tokenizer,
            vocabSize: modelConfig.vocabSize,
            enabled: true,
            debug: false,
            skipCaption: true,
            skipLanguage: true,
            stopAtReasoning: false,
            generationPhase: .understand,
            targetDurationSeconds: nil,
            userMetadata: .init()
        )

        func feed(_ text: String) {
            for id in tokenizer.encode(text, addSpecialTokens: false) {
                sampler.update(with: id)
            }
        }

        feed("<think>")
        feed("\n")
        feed("bpm:")
        feed(" 120\n")
        feed("duration:")
        feed(" 60\n")
        feed("keyscale:")
        feed(" C major\n")
        feed("timesignature:")
        feed(" 4\n")
        feed("</think>")

        XCTAssertEqual(sampler.state, .freeTextGeneration)

        let eos = try XCTUnwrap(tokenizer.eosTokenId)
        let audio0 = try XCTUnwrap(tokenizer.convertTokenToId("<|audio_code_0|>"))

        let logits = MLXArray.zeros([modelConfig.vocabSize], dtype: .float32)
        let masked = sampler.processLogits(logits, tokens: [])
        XCTAssertTrue(masked[eos].item(Float.self).isFinite)
        XCTAssertEqual(masked[audio0].item(Float.self), -Float.infinity)
    }
}
