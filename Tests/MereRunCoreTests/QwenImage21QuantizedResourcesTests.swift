import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class QwenImage21QuantizedResourcesTests: MereRunCoreTestCase {
    func testComponentLoaderPreservesPackedIntegersAndReadsComponentPrecision() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let component = root.appending(path: "transformer")
        try FileManager.default.createDirectory(at: component, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"quantization":{"bits":4,"group_size":64,"mode":"affine"}}"#.utf8)
            .write(to: component.appending(path: "config.json"))
        try MLX.save(arrays: ["layer.weight": MLXArray([UInt32.max, 0]).reshaped(1, 2),
                              "layer.scales": MLXArray([Float(0.125)])],
                     url: component.appending(path: "diffusion_pytorch_model.safetensors"))
        let resources = QwenImage21Resources(rootURL: root)
        let loaded = try resources.arrays("transformer", stem: "diffusion_pytorch_model")
        XCTAssertEqual(loaded["layer.weight"]?.dtype, .uint32)
        XCTAssertEqual(loaded["layer.weight"]?.asArray(UInt32.self), [UInt32.max, 0])
        XCTAssertEqual(loaded["layer.scales"]?.dtype, .bfloat16)
        XCTAssertEqual(try resources.quantization("transformer")?.bits, 4)
        try Data("{}".utf8).write(to: component.appending(path: "config.json"))
        XCTAssertNil(try resources.quantization("transformer"))
    }
}
