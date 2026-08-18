import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class SafetensorsStreamingLoaderTests: XCTestCase {
    func testMetadataParsesTypedHeaderAndSkipsFileMetadata() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let url = temp.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            header: """
            {
              "__metadata__": { "format": "pt" },
              "linear.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8]
              }
            }
            """,
            payloadByteCount: 8,
            to: url
        )

        let metadata = try SafetensorsStreamingLoader.metadata(url: url)
        let tensor = try XCTUnwrap(metadata["linear.weight"])
        let fileMetadata = try SafetensorsStreamingLoader.fileMetadata(url: url)
        XCTAssertEqual(metadata.count, 1)
        XCTAssertEqual(tensor.shape, [2])
        XCTAssertEqual(tensor.dtype, .float32)
        XCTAssertEqual(tensor.endOffset - tensor.startOffset, 8)
        XCTAssertEqual(fileMetadata, ["format": "pt"])
    }

    func testMetadataRejectsMalformedTensorHeader() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let url = temp.appendingPathComponent("malformed.safetensors")
        try writeSafetensors(
            header: """
            {
              "linear.weight": {
                "dtype": "F16",
                "shape": [2],
                "data_offsets": [0]
              }
            }
            """,
            payloadByteCount: 8,
            to: url
        )

        XCTAssertThrowsError(try SafetensorsStreamingLoader.metadata(url: url)) { error in
            XCTAssertTrue(String(describing: error).contains("malformedTensorMetadata"))
        }
    }

    func testMetadataRejectsOutOfRangeTensorData() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let url = temp.appendingPathComponent("out-of-range.safetensors")
        try writeSafetensors(
            header: """
            {
              "linear.weight": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 16]
              }
            }
            """,
            payloadByteCount: 8,
            to: url
        )

        XCTAssertThrowsError(try SafetensorsStreamingLoader.metadata(url: url)) { error in
            XCTAssertTrue(String(describing: error).contains("invalidTensorDataRange"))
        }
    }

    func testMetadataUsesTargetSizeForSymbolicLink() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let target = temp.appendingPathComponent("target.safetensors")
        try writeSafetensors(
            header: """
            {
              "linear.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8]
              }
            }
            """,
            payloadByteCount: 8,
            to: target
        )
        let link = temp.appendingPathComponent("model.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

        let metadata = try SafetensorsStreamingLoader.metadata(url: link)

        XCTAssertEqual(metadata["linear.weight"]?.shape, [2])
    }

    private func writeSafetensors(header: String, payloadByteCount: Int, to url: URL) throws {
        let headerData = Data(header.utf8)
        var headerSize = UInt64(headerData.count).littleEndian
        var fileData = Data()
        withUnsafeBytes(of: &headerSize) { bytes in
            fileData.append(contentsOf: bytes)
        }
        fileData.append(headerData)
        fileData.append(Data(repeating: 0, count: payloadByteCount))
        try fileData.write(to: url)
    }
}
