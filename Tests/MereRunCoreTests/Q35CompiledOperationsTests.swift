import MLX
import XCTest
@testable import MereRunCore

final class Q35CompiledOperationsTests: MereRunCoreTestCase {
    func testRepeatedRequestsOwnDistinctCompiledGraphsOnTheirActiveStream() async throws {
        var owners: [Q35CompiledOperations] = []
        var reference: [Float]?
        for _ in 0..<4 {
            let (owner, values) = await Q35CompiledOperations.withNewDefaultStream(scoped: true) {
                let operations = Q35CompiledOperations.current
                XCTAssertEqual(operations.stream, StreamOrDevice.default.stream)
                let gate = MLXArray([Float(-2), -1, 0, 1, 2, 3, 4, 5]).reshaped(1, 1, 8)
                let other = MLXArray.ones([1, 1, 8])
                let silu = q35Silu(gate)
                let swiglu = operations.swiglu(gate, other)
                let precise = operations.preciseSwiglu(other, gate, other)
                let decay = operations.computeG(MLXArray.zeros([8]), gate, MLXArray.zeros([8]))
                MLX.eval(silu, swiglu, precise, decay)
                return (operations, silu.asArray(Float.self) + swiglu.asArray(Float.self) + precise.asArray(Float.self) + decay.asArray(Float.self))
            }
            XCTAssertFalse(owners.contains { $0 === owner }, "A new request must not reuse an earlier stream's graph owner")
            owners.append(owner)
            if let reference { XCTAssertEqual(values, reference) } else { reference = values }
        }
    }

    func testNestedRequestRestoresItsParentsCompilerScopeAfterSuspension() async throws {
        await Q35CompiledOperations.withNewDefaultStream(scoped: true) {
            let parent = Q35CompiledOperations.current
            await Q35CompiledOperations.withNewDefaultStream(scoped: true) {
                let child = Q35CompiledOperations.current
                XCTAssertFalse(parent === child)
                await Task.yield()
                XCTAssertTrue(child === Q35CompiledOperations.current)
                XCTAssertEqual(child.stream, StreamOrDevice.default.stream)
            }
            XCTAssertTrue(parent === Q35CompiledOperations.current)
            XCTAssertEqual(parent.stream, StreamOrDevice.default.stream)
        }
    }
}
