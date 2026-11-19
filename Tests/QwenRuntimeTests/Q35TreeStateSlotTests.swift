import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel

final class Q35TreeStateSlotTests: MLXTestCase {
    func testEveryParentSurvivesUntilItsFinalChild() throws {
        var rngState: UInt64 = 733
        var topologies = Self.topologies.map(\.1) + [[-1], [-1] + Array(repeating: 0, count: 15)]
        for count in 2...16 {
            for _ in 0..<32 {
                var parents = [-1]
                for row in 1..<count {
                    rngState = rngState &* 6_364_136_223_846_793_005 &+ 1
                    parents.append(Int(rngState % UInt64(row)))
                }
                topologies.append(parents)
            }
        }
        for parents in topologies {
            let tree = try Q35VerificationTree(parents: parents)
            let plan = tree.recurrentSlots
            XCTAssertLessThanOrEqual(plan.count, 3)
            XCTAssertEqual(plan.order.sorted(), Array(parents.indices))
            var snapshots = Array<[Int]?>(repeating: nil, count: plan.count)
            for row in plan.order {
                let parentPath: [Int]
                if row == 0 { parentPath = [] }
                else { parentPath = try XCTUnwrap(snapshots[plan.reads[row]]) }
                XCTAssertEqual(parentPath, row == 0 ? [] : tree.paths[parents[row]])
                let actual = parentPath + [row]
                XCTAssertEqual(actual, tree.paths[row])
                if plan.writes[row] >= 0 { snapshots[plan.writes[row]] = actual }
                else { XCTAssertFalse(parents.dropFirst().contains(row)) }
            }
        }
        XCTAssertEqual(try Q35VerificationTree(parents: [-1]).recurrentSlots.count, 0)
        XCTAssertEqual(try Q35VerificationTree(parents: Self.topologies[0].1).recurrentSlots.count, 1)
        XCTAssertEqual(try Q35VerificationTree(parents: Self.topologies[3].1).recurrentSlots.count, 2)
    }

    func testCompactOutputsMatchIndependentAndSerialPathsExactly() throws {
        #if os(macOS)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Compact recurrent Metal verification requires GPU.")
        MLXRandom.seed(737)
        for dtype in [DType.bfloat16, .float32] {
            for (_, parents) in Self.topologies + [("root", [-1])] {
                let tree = try Q35VerificationTree(parents: parents)
                let work = Q35TreeDeltaFixture(tree: tree, dtype: dtype)
                let actual = work.compact()
                exact(actual, work.independent())
                for row in parents.indices {
                    let path = tree.paths[row]
                    let serial = q35GatedDeltaUpdate(
                        q: Q35VerificationTree.take(work.q, rows: path),
                        k: Q35VerificationTree.take(work.k, rows: path),
                        v: Q35VerificationTree.take(work.v, rows: path),
                        a: Q35VerificationTree.take(work.a, rows: path),
                        b: Q35VerificationTree.take(work.b, rows: path),
                        aLog: work.aLog, dtBias: work.dtBias, state: work.state,
                        numKeyHeads: 16, numValueHeads: 48, valueHeadDim: 128
                    ).0
                    exact(actual[0..., row..<(row + 1), 0..., 0...], serial[0..., (path.count - 1)..., 0..., 0...])
                }
            }
        }
        #else
        throw XCTSkip("Compact recurrent Metal verification requires macOS.")
        #endif
    }

    static let topologies: [(String, [Int])] = [
        ("chain", [-1] + Array(0..<15)),
        ("balanced", [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]),
        ("forked-deep", [-1, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
        ("peak-eight", [-1] + Array(repeating: 0, count: 8) + Array(1...7))
    ]

    private func exact(_ actual: MLXArray, _ expected: MLXArray) {
        XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0)
    }
}

#if os(macOS)
struct Q35TreeDeltaFixture {
    let tree: Q35VerificationTree
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let a: MLXArray
    let b: MLXArray
    let aLog: MLXArray
    let dtBias: MLXArray
    let state: MLXArray
    let g: MLXArray
    let beta: MLXArray

    init(tree: Q35VerificationTree, dtype: DType) {
        self.tree = tree
        self.q = (MLXRandom.normal([1, tree.count, 16, 128]) * 0.05).asType(dtype)
        self.k = (MLXRandom.normal(q.shape) * 0.05).asType(dtype)
        self.v = (MLXRandom.normal([1, tree.count, 48, 128]) * 0.1).asType(dtype)
        self.a = MLXRandom.normal([1, tree.count, 48]).asType(dtype)
        self.b = MLXRandom.normal(a.shape).asType(dtype)
        self.aLog = MLXRandom.normal([48]).asType(dtype)
        self.dtBias = MLXRandom.normal([48]).asType(dtype)
        self.state = MLXRandom.normal([1, 48, 128, 128]) * 0.01
        self.g = Q35CompiledOperations.current.computeG(aLog, a, dtBias)
        self.beta = sigmoid(b).asType(.float32)
        eval(q, k, v, g, beta, state)
    }

    func compiled(compact: Bool) -> () -> MLXArray {
        let tree = self.tree
        let operation = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
            if compact {
                return [q35TreeCompactDelta(q: inputs[0], k: inputs[1], v: inputs[2], g: inputs[3],
                    beta: inputs[4], state: inputs[5], numKeyHeads: 16, numValueHeads: 48,
                    valueHeadDim: 128, tree: tree)]
            }
            return [q35TreeIndependentDelta(q: inputs[0], k: inputs[1], v: inputs[2], g: inputs[3],
                beta: inputs[4], state: inputs[5], numKeyHeads: 16, numValueHeads: 48,
                valueHeadDim: 128, tree: tree)]
        }
        let inputs = [q, k, v, g, beta, state]
        return { operation(inputs)[0] }
    }

    func compact() -> MLXArray {
        q35TreeCompactDelta(q: q, k: k, v: v, g: g, beta: beta, state: state,
                           numKeyHeads: 16, numValueHeads: 48, valueHeadDim: 128, tree: tree)
    }

    func independent() -> MLXArray {
        q35TreeIndependentDelta(q: q, k: k, v: v, g: g, beta: beta, state: state,
                               numKeyHeads: 16, numValueHeads: 48, valueHeadDim: 128, tree: tree)
    }
}

#endif
