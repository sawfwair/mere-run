import MLX
import MereRunDecode
import MereRunMLXTestSupport
import XCTest

final class DecodeLifecycleTests: MLXTestCase {
    private enum Failure: Error, Equatable {
        case cancelled
        case forward
    }

    private func logits(_ token: Int) -> MLXArray {
        var values = [Float](repeating: -10, count: 4)
        values[token] = 10
        return MLXArray(values).reshaped(1, 1, 4)
    }

    private func request(budget: Int = 6) -> AutoregressiveDecodeRequest {
        AutoregressiveDecodeRequest(
            initialLogits: logits(0),
            generationConfig: GenerationConfig(temperature: 0, topP: 1, repetitionPenalty: nil),
            eosTokens: [3], tokenBudget: budget
        )
    }

    func testCancellationBeforeSchedulingDoesNotForwardOrEmit() {
        var events: [String] = []
        XCTAssertThrowsError(try AutoregressiveDecodeEngine.decode(
            request(),
            stepForward: { _ in events.append("forward"); return self.logits(1) },
            decodeToken: { String($0) },
            emitPiece: { _, _ in events.append("emit") },
            checkCancellation: { throw Failure.cancelled }
        )) { XCTAssertEqual($0 as? Failure, .cancelled) }
        XCTAssertEqual(events, [])
    }

    func testCancellationWithPendingWorkDoesNotConfirmThatSample() {
        var checks = 0
        var events: [String] = []
        XCTAssertThrowsError(try AutoregressiveDecodeEngine.decode(
            request(),
            stepForward: { _ in events.append("forward"); return self.logits(1) },
            decodeToken: { String($0) },
            emitPiece: { token, _ in events.append("emit-\(token)") },
            checkCancellation: {
                checks += 1
                if checks == 3 { throw Failure.cancelled }
            }
        )) { XCTAssertEqual($0 as? Failure, .cancelled) }
        XCTAssertEqual(events, ["forward", "emit-0", "forward"])
    }

    func testForwardFailurePropagatesWithoutConfirmingQueuedSample() {
        var forwards = 0
        var emitted: [Int] = []
        XCTAssertThrowsError(try AutoregressiveDecodeEngine.decode(
            request(),
            stepForward: { _ in
                forwards += 1
                if forwards == 3 { throw Failure.forward }
                return self.logits(forwards)
            },
            decodeToken: { String($0) },
            emitPiece: { token, _ in emitted.append(token) }
        )) { XCTAssertEqual($0 as? Failure, .forward) }
        XCTAssertEqual(forwards, 3)
        XCTAssertEqual(emitted, [0])
    }

    func testRejectedFirstTokenKeepsDistinctSerialAndStatefulContracts() throws {
        let serial = try AutoregressiveDecodeEngine.decode(
            request(), stepForward: { _ in self.logits(1) },
            shouldContinue: { _, _ in false }
        )
        var events: [String] = []
        let stateful = AutoregressiveDecodeEngine.decodeStateful(
            request(),
            stepForward: { _ in events.append("forward"); return self.logits(1) },
            didSampleToken: { events.append("sample-\($0)") },
            shouldContinue: { token in events.append("continue-\(token)"); return false }
        )
        XCTAssertEqual(serial.generatedTokens, [0])
        XCTAssertEqual(stateful.generatedTokens, [])
        XCTAssertEqual(events, ["forward", "sample-0", "continue-0"])
    }

    func testStatefulFinalTokenPreservesOneStepAheadForward() {
        var forwards = 0
        let result = AutoregressiveDecodeEngine.decodeStateful(
            request(budget: 1),
            stepForward: { _ in forwards += 1; return self.logits(1) }
        )
        XCTAssertEqual(result.generatedTokens, [0])
        XCTAssertEqual(forwards, 1)
    }

    func testNonpositiveBudgetDoesNotInvokeCallbacksInEitherLoop() throws {
        for budget in [-1, 0] {
            var callbacks = 0
            let serial = try AutoregressiveDecodeEngine.decode(
                request(budget: budget),
                stepForward: { _ in callbacks += 1; return self.logits(1) },
                checkCancellation: { callbacks += 1 }
            )
            let stateful = AutoregressiveDecodeEngine.decodeStateful(
                request(budget: budget),
                stepForward: { _ in callbacks += 1; return self.logits(1) },
                didSampleToken: { _ in callbacks += 1 }
            )
            XCTAssertEqual(callbacks, 0)
            XCTAssertEqual(serial.generatedTokens, [])
            XCTAssertEqual(stateful.generatedTokens, [])
        }
    }
}
