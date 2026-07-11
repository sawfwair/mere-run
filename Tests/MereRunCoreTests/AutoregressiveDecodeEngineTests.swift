import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// Contract tests for the shared pipelined decode loop, driven by a scripted
/// fake model: one-hot logits make greedy sampling deterministic, so the
/// engine's token stream, EOS handling, budget cutoff, and lag semantics are
/// all directly assertable without a real model.
final class AutoregressiveDecodeEngineTests: XCTestCase {
    private let vocab = 16
    private let eos = 15

    override class func setUp() {
        super.setUp()
        MLXTestSupport.ensureMetalLibraryAvailable()
    }

    private func oneHotLogits(_ token: Int) -> MLXArray {
        var values = [Float](repeating: -10, count: vocab)
        values[token] = 10
        return MLXArray(values).reshaped(1, 1, vocab)
    }

    /// Scripted model: emits the given tokens in order regardless of input,
    /// then EOS forever.
    private func scriptedModel(_ script: [Int]) -> (MLXArray) -> MLXArray {
        var upcoming = script
        return { _ in
            let next = upcoming.isEmpty ? self.eos : upcoming.removeFirst()
            return self.oneHotLogits(next)
        }
    }

    private func request(firstToken: Int, budget: Int) -> AutoregressiveDecodeRequest {
        AutoregressiveDecodeRequest(
            initialLogits: oneHotLogits(firstToken),
            generationConfig: GenerationConfig(
                maxTokens: budget,
                temperature: 0,
                topP: 1.0,
                repetitionPenalty: nil,
                repetitionContextSize: 0
            ),
            eosTokens: [eos],
            tokenBudget: budget
        )
    }

    func testDecodesScriptedSequenceUntilEOS() throws {
        let result = try AutoregressiveDecodeEngine.decode(
            request(firstToken: 3, budget: 32),
            stepForward: scriptedModel([7, 2, 9, eos])
        )
        XCTAssertEqual(result.generatedTokens, [3, 7, 2, 9])
        let firstToken = try XCTUnwrap(result.firstTokenSeconds)
        XCTAssertLessThanOrEqual(firstToken, result.decodeSeconds)
    }

    func testImmediateEOSReportsNoFirstToken() throws {
        var forwardCalls = 0
        let model = scriptedModel([1, 2, 3])
        let result = try AutoregressiveDecodeEngine.decode(
            request(firstToken: eos, budget: 8),
            stepForward: { input in
                forwardCalls += 1
                return model(input)
            }
        )
        XCTAssertNil(result.firstTokenSeconds)
        XCTAssertEqual(forwardCalls, 1)
    }

    func testBudgetCutsGenerationExactly() throws {
        var forwardCalls = 0
        let model = scriptedModel([7, 2, 9, 4])
        let result = try AutoregressiveDecodeEngine.decode(
            request(firstToken: 3, budget: 2),
            stepForward: { input in
                forwardCalls += 1
                return model(input)
            }
        )
        XCTAssertEqual(result.generatedTokens, [3, 7])
        XCTAssertEqual(forwardCalls, 1)
    }

    func testSingleTokenBudgetDoesNotScheduleThrowawayForward() throws {
        var forwardCalls = 0
        let result = try AutoregressiveDecodeEngine.decode(
            request(firstToken: 3, budget: 1),
            stepForward: { _ in
                forwardCalls += 1
                return self.oneHotLogits(7)
            }
        )
        XCTAssertEqual(result.generatedTokens, [3])
        XCTAssertEqual(forwardCalls, 0)
    }

    func testSteadyStateQueuesForwardBeforeConfirmingPriorToken() throws {
        var events: [String] = []
        var forwardCalls = 0
        let model = scriptedModel([7, 2, 9, eos])
        let result = try AutoregressiveDecodeEngine.decode(
            request(firstToken: 3, budget: 4),
            stepForward: { input in
                forwardCalls += 1
                events.append("forward")
                return model(input)
            },
            decodeToken: { String($0) },
            emitPiece: { token, _ in events.append("emit-\(token)") }
        )

        XCTAssertEqual(result.generatedTokens, [3, 7, 2, 9])
        XCTAssertEqual(
            Array(events.prefix(5)),
            ["forward", "emit-3", "forward", "forward", "emit-7"]
        )
        XCTAssertEqual(forwardCalls, 3)
    }

    func testImmediateEOSProducesNothing() throws {
        let result = try AutoregressiveDecodeEngine.decode(
            request(firstToken: eos, budget: 8),
            stepForward: scriptedModel([1, 2, 3])
        )
        XCTAssertEqual(result.generatedTokens, [])
    }

    func testZeroBudgetShortCircuits() throws {
        var forwardCalls = 0
        let result = try AutoregressiveDecodeEngine.decode(
            request(firstToken: 3, budget: 0),
            stepForward: { _ in
                forwardCalls += 1
                return self.oneHotLogits(1)
            }
        )
        XCTAssertEqual(result.generatedTokens, [])
        XCTAssertEqual(forwardCalls, 0)
    }

    func testPieceEmissionBuffersWhitespace() throws {
        var pieces: [String] = []
        let names = [3: " ", 7: "\n", 2: "hello", 9: "world"]
        _ = try AutoregressiveDecodeEngine.decode(
            request(firstToken: 3, budget: 8),
            stepForward: scriptedModel([7, 2, 9, eos]),
            decodeToken: { names[$0] ?? "" },
            emitPiece: { _, piece in pieces.append(piece) }
        )
        // Whitespace-only pieces buffer until visible text arrives.
        XCTAssertEqual(pieces, [" \nhello", "world"])
    }

    func testStatefulDecodeProcessesConfirmedTokenHistory() {
        var processedHistories: [[Int]] = []
        var sampled: [Int] = []
        let result = AutoregressiveDecodeEngine.decodeStateful(
            request(firstToken: 3, budget: 8),
            processLogits: { logits, tokens in
                processedHistories.append(tokens)
                return logits
            },
            stepForward: scriptedModel([7, 2, eos]),
            didSampleToken: { sampled.append($0) }
        )

        XCTAssertEqual(result.generatedTokens, [3, 7, 2])
        XCTAssertEqual(processedHistories, [[], [3], [3, 7], [3, 7, 2]])
        XCTAssertEqual(sampled, [3, 7, 2, eos])
    }

    func testStatefulDecodeCanRejectSampleBeforeAppendingIt() {
        let result = AutoregressiveDecodeEngine.decodeStateful(
            request(firstToken: 3, budget: 8),
            stepForward: scriptedModel([7, 2, eos]),
            shouldContinue: { $0 != 7 }
        )

        XCTAssertEqual(result.generatedTokens, [3])
    }
}
