import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class SafetensorsSubsetWriterTests: MereRunCoreTestCase {
    func testWriterCopiesSelectedTensorPayloadWithoutMaterializingConversion() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "safetensors-subset-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("source.safetensors")
        let destinationURL = root.appendingPathComponent("subset.safetensors")
        try MLX.save(
            arrays: [
                "drop.weight": MLXArray([Float(1), Float(2)]),
                "keep.bias": MLXArray([Int32(7), Int32(8), Int32(9)]),
                "keep.weight": MLXArray([Float(3), Float(4), Float(5), Float(6)])
                    .asType(.bfloat16),
            ],
            metadata: ["source": "fixture"],
            url: sourceURL
        )

        let result = try SafetensorsSubsetWriter.write(
            sourceURL: sourceURL,
            destinationURL: destinationURL,
            fileMetadata: ["format": "fixture-v1"],
            include: { $0.hasPrefix("keep.") }
        )

        let sourceMetadata = try SafetensorsStreamingLoader.metadata(url: sourceURL)
        let outputMetadata = try SafetensorsStreamingLoader.metadata(url: destinationURL)
        XCTAssertEqual(Set(outputMetadata.keys), ["keep.bias", "keep.weight"])
        XCTAssertEqual(result.tensorCount, 2)
        XCTAssertEqual(
            result.payloadBytes,
            Int64(
                sourceMetadata["keep.bias"]!.endOffset
                    - sourceMetadata["keep.bias"]!.startOffset
                    + sourceMetadata["keep.weight"]!.endOffset
                    - sourceMetadata["keep.weight"]!.startOffset
            )
        )
        XCTAssertEqual(
            try SafetensorsStreamingLoader.fileMetadata(url: destinationURL)["format"],
            "fixture-v1"
        )
        let arrays = try MLX.loadArrays(url: destinationURL)
        XCTAssertEqual(arrays["keep.bias"]?.asArray(Int32.self), [7, 8, 9])
        XCTAssertEqual(
            arrays["keep.weight"]?.asType(.float32).asArray(Float.self),
            [3, 4, 5, 6]
        )
    }
}
