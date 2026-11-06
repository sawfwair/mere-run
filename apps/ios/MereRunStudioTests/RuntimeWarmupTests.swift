import XCTest

@MainActor
final class RuntimeWarmupTests: XCTestCase {
    func testCancellationWaitsForUncooperativePreparationBeforeReturning() async throws {
        let warmup = RuntimeWarmup()
        let gate = PreparationGate()
        var state = RuntimeResidencyState()
        let lease = try XCTUnwrap(state.beginWarmup(modelID: "chat"))
        var outcomes: [Bool] = []
        warmup.start(lease, prepare: { await gate.pause() }, completed: { _, success in outcomes.append(success) })
        await gate.waitUntilStarted()
        let entered = AsyncStream<Void>.makeStream()
        defer { entered.continuation.finish() }
        var drained = false
        let waiter = Task {
            entered.continuation.yield(())
            let result = await warmup.drain(cancel: true)
            drained = true
            return result
        }
        var iterator = entered.stream.makeAsyncIterator()
        _ = await iterator.next()

        XCTAssertTrue(warmup.isRunning)
        XCTAssertFalse(drained, "cancellation cannot stand in for native preparation finishing")
        XCTAssertTrue(outcomes.isEmpty)
        gate.resume()
        let result = await waiter.value
        XCTAssertEqual(result, false, "ignoring cancellation cannot publish a successful warmup")
        XCTAssertEqual(outcomes, [false])
        XCTAssertFalse(warmup.isRunning)
    }

    func testChatHandoffWaitsForAndReusesSuccessfulPreparation() async throws {
        let warmup = RuntimeWarmup()
        let gate = PreparationGate()
        var state = RuntimeResidencyState()
        let preparation = try XCTUnwrap(state.beginWarmup(modelID: "chat"))
        warmup.start(preparation, prepare: { await gate.pause() }, completed: { lease, success in
            XCTAssertTrue(state.completeWarmup(lease, succeeded: success))
        })
        await gate.waitUntilStarted()
        let chat = try XCTUnwrap(state.begin(.chatting, modelID: "chat"))
        let entered = AsyncStream<Void>.makeStream()
        defer { entered.continuation.finish() }
        var joined = false
        let waiter = Task {
            entered.continuation.yield(())
            let result = await warmup.drain(cancel: false)
            joined = true
            return result
        }
        var iterator = entered.stream.makeAsyncIterator()
        _ = await iterator.next()
        XCTAssertFalse(joined)

        gate.resume()
        let result = await waiter.value
        XCTAssertEqual(result, true)
        XCTAssertTrue(state.isCurrent(chat))
        XCTAssertEqual(state.warmChatModelID, "chat")
        XCTAssertFalse(warmup.isRunning)
    }

    func testOldDrainCannotClearANewPreparationStartedFromCompletion() async throws {
        let warmup = RuntimeWarmup()
        let oldGate = PreparationGate()
        let newGate = PreparationGate()
        let old = RuntimeResidencyState.WarmupLease(modelID: "old")
        let current = RuntimeResidencyState.WarmupLease(modelID: "current")
        let finished = AsyncStream<Void>.makeStream()
        defer { finished.continuation.finish() }
        warmup.start(old, prepare: { await oldGate.pause() }, completed: { _, _ in
            warmup.start(current, prepare: { await newGate.pause() }, completed: { _, success in
                XCTAssertTrue(success)
                finished.continuation.yield(())
            })
        })
        await oldGate.waitUntilStarted()
        let entered = AsyncStream<Void>.makeStream()
        defer { entered.continuation.finish() }
        let waiter = Task {
            entered.continuation.yield(())
            return await warmup.drain(cancel: false)
        }
        var iterator = entered.stream.makeAsyncIterator()
        _ = await iterator.next()
        oldGate.resume()
        let result = await waiter.value
        XCTAssertEqual(result, true)
        await newGate.waitUntilStarted()
        XCTAssertTrue(warmup.isRunning, "an old awaiter cannot clear the new owner's task")

        newGate.resume()
        var completion = finished.stream.makeAsyncIterator()
        _ = await completion.next()
        XCTAssertFalse(warmup.isRunning)
    }

    func testReleaseRejectsLateWarmupReadinessAndWaitsBeforeUnloading() async throws {
        let warmup = RuntimeWarmup()
        let gate = PreparationGate()
        var state = RuntimeResidencyState()
        let lease = try XCTUnwrap(state.beginWarmup(modelID: "old"))
        var prepared = false
        var unloaded = false
        warmup.start(lease, prepare: {
            await gate.pause()
            prepared = true
        }, completed: { ticket, success in
            XCTAssertFalse(state.completeWarmup(ticket, succeeded: success))
        })
        await gate.waitUntilStarted()
        XCTAssertTrue(state.requestRelease())
        let entered = AsyncStream<Void>.makeStream()
        defer { entered.continuation.finish() }
        let release = Task {
            entered.continuation.yield(())
            await warmup.drain(cancel: true)
            XCTAssertTrue(prepared, "unload follows actual preparation completion")
            unloaded = true
            state.completeRelease()
        }
        var iterator = entered.stream.makeAsyncIterator()
        _ = await iterator.next()
        XCTAssertFalse(unloaded)
        XCTAssertEqual(state.activity, .releasing)
        XCTAssertNil(state.beginWarmup(modelID: "new"))

        gate.resume()
        await release.value
        XCTAssertTrue(unloaded)
        XCTAssertNil(state.warmChatModelID)
        XCTAssertTrue(state.isIdle)
        XCTAssertNotNil(state.beginWarmup(modelID: "new"))
    }
}

@MainActor
private final class PreparationGate {
    private var started = false
    private var startWaiter: CheckedContinuation<Void, Never>?
    private var continuation: CheckedContinuation<Void, Never>?

    func pause() async {
        started = true
        startWaiter?.resume()
        startWaiter = nil
        await withCheckedContinuation { continuation = $0 }
    }

    func waitUntilStarted() async {
        if started { return }
        await withCheckedContinuation { startWaiter = $0 }
    }

    func resume() {
        continuation?.resume()
        continuation = nil
    }
}
