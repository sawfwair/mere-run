import Foundation
import Testing
@testable import MereRunCore

@Suite("TripoSR checkpoint inventory")
struct TripoSRCheckpointInventoryTests {
    @Test("Frozen conversion inventory has exact production totals")
    func frozenConversionInventory() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = root.appendingPathComponent("scripts/model-conversion/triposr-tensor-inventory.json")
        let object = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: [String: Any]]
        )
        #expect(object.count == TripoSRWeights.sourceTensorCount)
        let scalarCount = try object.values.reduce(0) { partial, record in
            let shape = try #require(record["shape"] as? [Int])
            return partial + shape.reduce(1, *)
        }
        #expect(scalarCount == TripoSRWeights.sourceScalarCount)
        #expect(
            object["post_processor.upsample.weight"]?["shape"] as? [Int]
                == [1_024, 40, 2, 2]
        )
    }

    @Test("Pinned checkpoint is accepted by the restricted state-dict reader")
    func pinnedCheckpointInventory() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_TEST_TRIPOSR_CKPT"] else {
            return
        }
        let archive = try PyTorchStateDictArchive(url: URL(fileURLWithPath: path))
        #expect(archive.tensors.count == TripoSRWeights.sourceTensorCount)
        #expect(archive.tensors.reduce(0) { $0 + $1.elementCount } == TripoSRWeights.sourceScalarCount)
        #expect(
            archive.descriptor(named: "decoder.layers.18.weight")?.shape == [4, 64]
        )
        if ProcessInfo.processInfo.environment["MERERUN_DUMP_TRIPOSR_INVENTORY"] == "1" {
            for tensor in archive.tensors {
                print("\(tensor.name)\t\(tensor.dataType.rawValue)\t\(tensor.shape.map(String.init).joined(separator: ","))")
            }
        }
    }
}
