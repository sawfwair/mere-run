import XCTest
@testable import MereRunCore

final class TerraMindFloodResourcesTests: XCTestCase {
    func testPinsExactFP32ConversionContract() throws {
        let configuration = try decodeConfiguration(dtype: "float32")

        XCTAssertNoThrow(try TerraMindFloodResources.validateConfiguration(configuration))
        XCTAssertEqual(TerraMindFloodResources.tensorCount, 171)
        XCTAssertEqual(TerraMindFloodResources.scalarCount, 168_416_386)
        XCTAssertEqual(TerraMindFloodModel.requiredTensorNames.count, 171)
        XCTAssertEqual(
            TerraMindFloodResources.weightsArtifact.sha256,
            "4940ad94df06a923e3a919f944a71ad01892872e89c428abe718eefc44d0f95a"
        )
    }

    func testRejectsFP16BecauseItChangesFloodBoundary() throws {
        let configuration = try decodeConfiguration(dtype: "float16")

        XCTAssertThrowsError(try TerraMindFloodResources.validateConfiguration(configuration)) { error in
            XCTAssertEqual(error as? TerraMindFloodResourceError, .unsupportedPrecision("float16"))
        }
    }

    private func decodeConfiguration(dtype: String) throws -> TerraMindFloodConversionConfiguration {
        let json = """
        {
          "converter": "scripts/convert-terramind-flood-mlx.py@v1",
          "dtype": "\(dtype)",
          "format": "mere.run/terramind-flood-mlx-v1",
          "model_id": "vision-flood-terramind-base",
          "scalar_count": 168416386,
          "source_checkpoint": "TerraMind_v1_base_ImpactMesh_flood.pt",
          "source_checkpoint_sha256": "22627584c2db618c2f6ddb64b411a95762a893becb25104e3f66bfebecaa71e9",
          "source_configuration_sha256": "d6c74ef58085a6d3f27bca2d570d84b9256100b885e7c51521c9d0cf7f335282",
          "source_repository": "ibm-esa-geospatial/TerraMind-base-Flood",
          "source_revision": "1e4b2429d17234922f8d92beb0d725af4db85c08",
          "tensor_count": 171,
          "tile_size": 256,
          "timestamps": 4
        }
        """
        return try JSONDecoder().decode(
            TerraMindFloodConversionConfiguration.self,
            from: Data(json.utf8)
        )
    }
}
