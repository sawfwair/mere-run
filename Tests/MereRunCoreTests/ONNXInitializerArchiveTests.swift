import Foundation
import MereRunCore
import XCTest

final class ONNXInitializerArchiveTests: XCTestCase {
    func testReadsEmbeddedFloatInitializerWithoutExecutingGraph() throws {
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let floats: [Float] = [1.5, -2]
        let tensor = message([
            field(1, packedVarints: [1, 2]),
            field(2, varint: 1),
            field(8, bytes: Data("weight".utf8)),
            field(9, bytes: floatBytes(floats)),
        ])
        let graph = message([field(5, bytes: tensor)])
        try message([field(7, bytes: graph)]).write(to: url)

        let archive = try ONNXInitializerArchive(url: url)
        XCTAssertEqual(archive.tensors.count, 1)
        let descriptor = try XCTUnwrap(archive.descriptor(named: "weight"))
        XCTAssertEqual(descriptor.shape, [1, 2])
        XCTAssertEqual(descriptor.dataType, .float32)
        XCTAssertEqual(archive.rawData(for: descriptor), floatBytes(floats))
    }

    func testRejectsRawByteCountMismatch() throws {
        let url = temporaryFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let tensor = message([
            field(1, packedVarints: [2]),
            field(2, varint: 1),
            field(8, bytes: Data("bad".utf8)),
            field(9, bytes: Data([0, 0, 0, 0])),
        ])
        try message([field(7, bytes: message([field(5, bytes: tensor)]))]).write(to: url)
        XCTAssertThrowsError(try ONNXInitializerArchive(url: url)) { error in
            XCTAssertEqual(
                error as? ONNXInitializerError,
                .tensorByteCountMismatch(name: "bad", expected: 8, actual: 4)
            )
        }
    }

    func testPinnedMoGeArchiveInventoryWhenFixtureIsAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_MOGE_ONNX"] ?? ""
        try XCTSkipIf(path.isEmpty || !FileManager.default.fileExists(atPath: path), "Set MERERUN_TEST_MOGE_ONNX to the pinned model.onnx fixture.")
        let archive = try ONNXInitializerArchive(url: URL(fileURLWithPath: path))
        XCTAssertEqual(archive.tensors.count, 336)
        XCTAssertEqual(archive.tensors.reduce(0) { $0 + $1.elementCount }, 35_102_894)
        let patch = try XCTUnwrap(archive.descriptor(named: "encoder.backbone.patch_embed.proj.weight"))
        XCTAssertEqual(patch.shape, [384, 3, 14, 14])
        XCTAssertEqual(try XCTUnwrap(archive.descriptor(named: "scale_head.4.weight")).shape, [1, 384])
    }

    private func temporaryFile() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("onnx-\(UUID().uuidString).onnx")
    }

    private func message(_ fields: [Data]) -> Data {
        fields.reduce(into: Data()) { $0.append($1) }
    }

    private func field(_ number: Int, varint: UInt64) -> Data {
        var data = encodeVarint(UInt64(number << 3))
        data.append(encodeVarint(varint))
        return data
    }

    private func field(_ number: Int, bytes: Data) -> Data {
        var data = encodeVarint(UInt64((number << 3) | 2))
        data.append(encodeVarint(UInt64(bytes.count)))
        data.append(bytes)
        return data
    }

    private func field(_ number: Int, packedVarints: [UInt64]) -> Data {
        field(number, bytes: packedVarints.reduce(into: Data()) { $0.append(encodeVarint($1)) })
    }

    private func encodeVarint(_ value: UInt64) -> Data {
        var value = value
        var data = Data()
        repeat {
            var byte = UInt8(value & 0x7F)
            value >>= 7
            if value != 0 { byte |= 0x80 }
            data.append(byte)
        } while value != 0
        return data
    }

    private func floatBytes(_ values: [Float]) -> Data {
        values.reduce(into: Data()) { data, value in
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
    }
}
