import Foundation
import Testing
@testable import MereRunCore

@Suite("InstantMesh Base checkpoint inventory")
struct InstantMeshCheckpointInventoryTests {
    @Test("Frozen reconstruction-only inventory has exact totals")
    func frozenConversionInventory() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = root.appendingPathComponent(
            "scripts/model-conversion/instantmesh-base-tensor-inventory.json"
        )
        let object = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: [String: Any]]
        )
        #expect(object.count == InstantMeshWeights.sourceTensorCount)
        let scalarCount = try object.values.reduce(0) { partial, record in
            let shape = try #require(record["shape"] as? [Int])
            return partial + shape.reduce(1, *)
        }
        #expect(scalarCount == InstantMeshWeights.sourceScalarCount)
        #expect(object["transformer.deconv.weight"]?["shape"] as? [Int] == [1_024, 40, 2, 2])
        #expect(object["synthesizer.decoder.net_weight.6.weight"]?["shape"] as? [Int] == [21, 64])
        #expect(object.keys.allSatisfy { !$0.contains("zero123") && !$0.contains("diffusion") })
    }
}
