import Foundation
import MLX
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel
@testable import MereRunTensor

final class Q35DFlash2PolicyTests: MereRunCoreTestCase {
    func testUnconstrainedTextSupportsGreedyAndSampledRequests() {
        let greedy = GenerationConfig(temperature: 0, repetitionPenalty: 1)
        XCTAssertTrue(eligible(greedy))
        var sampled = greedy
        sampled.temperature = 0.7
        XCTAssertTrue(eligible(sampled))
        sampled.topP = 0.8
        sampled.topK = 20
        sampled.presencePenalty = 1.5
        sampled.repetitionPenalty = 1.1
        sampled.frequencyPenalty = 0.2
        sampled.bannedTokens = [1]
        XCTAssertTrue(eligible(sampled))
        var penalized = greedy
        penalized.presencePenalty = 0.2
        XCTAssertFalse(eligible(penalized))
        var banned = greedy
        banned.bannedTokens = [1]
        XCTAssertFalse(eligible(banned))
        XCTAssertFalse(eligible(greedy, tokenBudget: 1))
        XCTAssertFalse(eligible(greedy, images: true))
        XCTAssertFalse(eligible(greedy, tools: true))
        XCTAssertFalse(eligible(greedy, json: true))
        XCTAssertFalse(eligible(greedy, logprobs: .tokens))
        XCTAssertFalse(eligible(greedy, batching: true))
    }

    func testInstalledGreedyParityAndRepeat() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MERERUN_TEST_Q35_DFLASH2"] == "1", "Installed Qwen DFlash 2 qualification is opt-in.")
        let targetRoot = try XCTUnwrap(env["MERERUN_TEST_Q35_DFLASH2_TARGET_ROOT"])
        let draftRoot = try XCTUnwrap(env["MERERUN_TEST_Q35_DFLASH2_DRAFT_ROOT"])
        let generator = Q35Generator(
            modelId: Q35Resources.q38TwentySevenB4BitModelId,
            prefixKVCacheEnabled: false, continuousBatchingEnabled: false, dflash2ModelPath: draftRoot
        )
        do {
            for prompt in ["Write a Python LRU cache with get and put methods.",
                           "Explain how rain forms in three paragraphs."] {
                var request = ChatRequest(
                    messages: [ChatMessage(role: .user, content: prompt)], maxTokens: 128,
                    temperature: 0, topP: 1, showThinking: false, maxContextTokens: 8192
                )
                let candidate = try await generator.chat(request, modelPath: targetRoot, progressHandler: nil)
                #if DEBUG
                let candidateTokens = await generator.lastGeneratedTokenIDsForTesting
                XCTAssertEqual(candidateTokens.count, candidate.tokensGenerated)
                XCTAssertGreaterThan(candidateTokens.count, 0)
                #endif
                if Q35Generator.shouldUseM5Verification(
                    modelID: Q35Resources.q38TwentySevenB4BitModelId, draftAvailable: true
                ) {
                    #if os(macOS)
                    XCTAssertTrue(M5AffineQ4Verification.isSupported)
                    try await generator.assertM5HeadDispatchForTest()
                    #endif
                    let backends = await generator.verificationBackendsForTest()
                    XCTAssertEqual(backends.target, .m5Matrix)
                    XCTAssertFalse(backends.draft.isEmpty)
                    XCTAssertTrue(backends.draft.allSatisfy { $0 == .reference })
                    if !Q35Generator.shouldUseM5Projection(path: "model.layers.0.mlp.gate_proj") {
                        XCTAssertFalse(backends.targetBody.isEmpty)
                        XCTAssertTrue(backends.targetBody.allSatisfy { $0 == .reference })
                    }
                }
                XCTAssertEqual(candidate.acceleration?.route, "dflash2")
                XCTAssertGreaterThan(candidate.acceleration?.draftedTokens ?? 0, 0)
                let repeatRun = try await generator.chat(request, modelPath: targetRoot, progressHandler: nil)
                XCTAssertEqual(candidate.response, repeatRun.response)
                XCTAssertEqual(candidate.tokensGenerated, repeatRun.tokensGenerated)
                #if DEBUG
                let repeatTokens = await generator.lastGeneratedTokenIDsForTesting
                XCTAssertEqual(candidateTokens, repeatTokens)
                #endif
                // Log-probability capture forces the serial target oracle.
                request.logprobCapture = .tokens
                let oracle = try await generator.chat(request, modelPath: targetRoot, progressHandler: nil)
                XCTAssertEqual(oracle.acceleration?.route, "final-target-pipelined")
                XCTAssertEqual(candidate.response, oracle.response)
                XCTAssertEqual(candidate.tokensGenerated, oracle.tokensGenerated)
                #if DEBUG
                let oracleTokens = await generator.lastGeneratedTokenIDsForTesting
                XCTAssertEqual(candidateTokens, oracleTokens)
                print("[q35-installed-parity] tokens=\(candidateTokens.count) repeat=\(candidateTokens == repeatTokens) oracle=\(candidateTokens == oracleTokens)")
                #endif
            }
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private func eligible(
        _ config: GenerationConfig, tokenBudget: Int = 128, images: Bool = false, tools: Bool = false,
        json: Bool = false, logprobs: ChatLogprobCapture = .none, batching: Bool = false
    ) -> Bool {
        Q35Generator.dflash2Eligible(
            generationConfig: config, tokenBudget: tokenBudget, hasImages: images, hasTools: tools,
            jsonConstrained: json, logprobCapture: logprobs, continuousBatchingEnabled: batching
        )
    }
}

private extension Q35Generator {
    #if os(macOS)
    func assertM5HeadDispatchForTest() throws {
        let target = try XCTUnwrap(model)
        let head = try XCTUnwrap(target.lmHead as? PortableQuantizedLinear)
        let biases = try XCTUnwrap(head.biases)
        let input = MLXArray([Int32](repeating: 1, count: 8)).reshaped(1, 8)
        let hidden = target.forward(input, cache: nil).hidden
        XCTAssertEqual(hidden.dtype, .bfloat16)
        let expected: MLXArray
        if PortableQuantizedLinear.m5TiledWeightsEnabled {
            let packed = try XCTUnwrap(M5TiledAffineQ4.Weights(weight: head.weight, scales: head.scales, biases: biases))
            expected = try XCTUnwrap(M5TiledAffineQ4.matmul(hidden, weights: packed))
        } else {
            expected = try XCTUnwrap(M5AffineQ4Verification.matmul(
                hidden, weight: head.weight, scales: head.scales, biases: biases
            ))
        }
        let actual = head(hidden)
        let reference = PortableQuantizedLinear(
            weight: head.weight, bias: head.bias, scales: head.scales, biases: biases,
            groupSize: head.groupSize, bits: head.bits, mode: head.mode
        )(hidden)
        let actualDifference = (actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self)
        let referenceDifference = (reference.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self)
        XCTAssertEqual(actualDifference, 0, "Installed head did not execute the selected matrix arithmetic")
        XCTAssertGreaterThan(referenceDifference, 0, "Dispatch probe must distinguish reference and matrix arithmetic")
        print("[q35-m5-head-dispatch] actualDifference=\(actualDifference) referenceDifference=\(referenceDifference) "
              + "tiled=\(PortableQuantizedLinear.m5TiledWeightsEnabled) gpuActiveBytes=\(MLX.Memory.activeMemory)")
    }
    #endif

    func verificationBackendsForTest() -> (
        target: QuantizedVerificationBackend?, draft: [QuantizedVerificationBackend], targetBody: [QuantizedVerificationBackend]
    ) {
        let target = (model?.lmHead as? PortableQuantizedLinear)?.verificationBackend
        let draft = dflash2Model?.leafModules().flattened().compactMap {
            ($0.1 as? PortableQuantizedLinear)?.verificationBackend
        } ?? []
        let targetBody = model?.leafModules().flattened().compactMap { path, module in
            path == "lm_head" ? nil : (module as? PortableQuantizedLinear)?.verificationBackend
        } ?? []
        return (target, draft, targetBody)
    }
}
