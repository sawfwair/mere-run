import Foundation
import XCTest

final class SCAIL2ConverterContractTests: XCTestCase {
    func testConverterPinsEveryExecutableAndCheckpointBoundary() throws {
        let scriptURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("scripts/model-conversion/convert_scail2.py")
        let script = try String(contentsOf: scriptURL, encoding: .utf8)

        XCTAssertTrue(script.contains("150cc0ca4e98e50e60b9295dacde39442fdccab2"))
        XCTAssertTrue(script.contains("5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"))
        XCTAssertTrue(script.contains("d6c73e94c57eb36e6351c800d1228e41ed7e45db1ccf410dd875bcfdd2945e7f"))
        XCTAssertTrue(script.contains("7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"))
        XCTAssertTrue(script.contains("020daacf4c0ce94284df584d243cefb6ddfadefff8772226a8e85431df5de2da"))
        XCTAssertTrue(script.contains("38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"))
        XCTAssertTrue(script.contains("6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"))
        XCTAssertTrue(script.contains("weights_only=True"))
        XCTAssertFalse(script.contains("weights_only=False"))
        XCTAssertTrue(script.contains("torch.serialization.safe_globals"))
        XCTAssertTrue(script.contains("np.uint32"))
        XCTAssertTrue(script.contains("mmap=True"))
        XCTAssertTrue(script.contains("Duplicate converted tensor key"))
        XCTAssertTrue(script.contains("provenance.json"))
        XCTAssertTrue(script.contains("mererun_model.json"))
        XCTAssertTrue(script.contains("video-scail2-14b-mlx"))
        XCTAssertTrue(script.contains("upstreamRepoId"))
        XCTAssertTrue(script.contains("is_float32_transformer_tensor"))
        XCTAssertTrue(script.contains("key.startswith(\"time_embedding_\")"))
        XCTAssertTrue(script.contains("key.startswith(\"time_projection.\")"))
        XCTAssertTrue(script.contains("key.startswith(\"head.\")"))
    }
}
