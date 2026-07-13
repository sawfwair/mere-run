import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class PyTorchStateDictArchiveTests: XCTestCase {
    func testLoadsWhitelistedProtocolTwoFloatTensor() throws {
        let url = try writeCheckpoint(
            pickle: stateDictPickle(shape: [2], stride: [1], storageElementCount: 2),
            storage: floatData([1.5, -2])
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let archive = try PyTorchStateDictArchive(url: url)
        XCTAssertEqual(archive.tensors.count, 1)
        let descriptor = try XCTUnwrap(archive.descriptor(named: "weight"))
        XCTAssertEqual(descriptor.shape, [2])
        XCTAssertEqual(descriptor.stride, [1])
        XCTAssertEqual(descriptor.dataType, .float32)
        XCTAssertEqual(descriptor.storageKey, "0")
        XCTAssertEqual(descriptor.elementCount, 2)
        XCTAssertEqual(try archive.rawData(for: descriptor), floatData([1.5, -2]))

        let array = try archive.loadArray(for: descriptor)
        MLX.eval(array)
        XCTAssertEqual(array.shape, [2])
        XCTAssertEqual(array.asArray(Float.self), [1.5, -2])
    }

    func testAcceptsExactGeneratorStateDictWrapper() throws {
        let url = try writeCheckpoint(
            pickle: generatorStateDictPickle(shape: [2], stride: [1], storageElementCount: 2),
            storage: floatData([1.5, -2])
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let archive = try PyTorchStateDictArchive(url: url)
        XCTAssertEqual(archive.tensors.map(\.name), ["weight"])
        XCTAssertEqual(try archive.rawData(named: "weight"), floatData([1.5, -2]))
    }

    func testAcceptsCUDAStorageLocationWithoutExecutingDeviceCode() throws {
        let url = try writeCheckpoint(
            pickle: stateDictPickle(
                shape: [1],
                stride: [1],
                storageElementCount: 1,
                storageDevice: "cuda:0"
            ),
            storage: floatData([2])
        )
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertEqual(try PyTorchStateDictArchive(url: url).rawData(named: "weight"), floatData([2]))
    }

    func testAcceptsExactModernTorchSerializationIdentifier() throws {
        let url = try writeCheckpoint(
            pickle: stateDictPickle(shape: [1], stride: [1], storageElementCount: 1),
            storage: floatData([3]),
            serializationID: "1828009421549095952311728918578000915067"
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let archive = try PyTorchStateDictArchive(url: url)
        XCTAssertEqual(archive.tensors.count, 1)
    }

    func testRejectsNonDecimalModernTorchSerializationIdentifier() throws {
        let url = try writeCheckpoint(
            pickle: stateDictPickle(shape: [1], stride: [1], storageElementCount: 1),
            storage: floatData([3]),
            serializationID: "../../payload"
        )
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try PyTorchStateDictArchive(url: url)) { error in
            XCTAssertEqual(
                error as? PyTorchStateDictError,
                .malformedZIP("invalid serialization identifier")
            )
        }
    }

    func testRejectsUnknownPickleOpcode() throws {
        var pickle = stateDictPickle(shape: [1], stride: [1], storageElementCount: 1)
        pickle[2] = 0x4E // NONE is deliberately outside the state-dict whitelist.
        let url = try writeCheckpoint(pickle: pickle, storage: floatData([0]))
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try PyTorchStateDictArchive(url: url)) { error in
            XCTAssertEqual(error as? PyTorchStateDictError, .unsupportedPickleOpcode(0x4E, offset: 2))
        }
    }

    func testRejectsUnknownPickleGlobalWithoutImportingIt() throws {
        let pickle = stateDictPickle(
            shape: [1],
            stride: [1],
            storageElementCount: 1,
            rebuildGlobal: ("os", "system")
        )
        let url = try writeCheckpoint(pickle: pickle, storage: floatData([0]))
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try PyTorchStateDictArchive(url: url)) { error in
            XCTAssertEqual(
                error as? PyTorchStateDictError,
                .unsupportedPickleGlobal(module: "os", name: "system")
            )
        }
    }

    func testRejectsUnicodeDigitStorageIdentifier() throws {
        let pickle = stateDictPickle(
            shape: [1],
            stride: [1],
            storageElementCount: 1,
            storageKey: "١"
        )
        let url = try writeCheckpoint(pickle: pickle, storage: floatData([0]))
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try PyTorchStateDictArchive(url: url)) { error in
            XCTAssertEqual(
                error as? PyTorchStateDictError,
                .malformedPickle("invalid persistent storage identifier")
            )
        }
    }

    func testRejectsNonContiguousTensor() throws {
        let url = try writeCheckpoint(
            pickle: stateDictPickle(shape: [2, 2], stride: [1, 2], storageElementCount: 4),
            storage: floatData([0, 1, 2, 3])
        )
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try PyTorchStateDictArchive(url: url)) { error in
            XCTAssertEqual(
                error as? PyTorchStateDictError,
                .nonContiguousTensor(name: "weight", shape: [2, 2], stride: [1, 2])
            )
        }
    }

    func testRejectsStorageByteCountMismatch() throws {
        let url = try writeCheckpoint(
            pickle: stateDictPickle(shape: [2], stride: [1], storageElementCount: 3),
            storage: floatData([0, 1])
        )
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try PyTorchStateDictArchive(url: url)) { error in
            XCTAssertEqual(
                error as? PyTorchStateDictError,
                .storageByteCountMismatch(key: "0", expected: 12, actual: 8)
            )
        }
    }

    func testRejectsStorageWithBadCRC() throws {
        let url = try writeCheckpoint(
            pickle: stateDictPickle(shape: [1], stride: [1], storageElementCount: 1),
            storage: floatData([0]),
            corruptStorageCRC: true
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let archive = try PyTorchStateDictArchive(url: url)
        XCTAssertThrowsError(try archive.rawData(named: "weight")) { error in
            XCTAssertEqual(error as? PyTorchStateDictError, .checksumMismatch("checkpoint/data/0"))
        }
    }

    func testWholeFileDigestCallersCanSkipRedundantEntryCRC() throws {
        let storage = floatData([7])
        let url = try writeCheckpoint(
            pickle: stateDictPickle(shape: [1], stride: [1], storageElementCount: 1),
            storage: storage,
            corruptStorageCRC: true
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let archive = try PyTorchStateDictArchive(
            url: url,
            verifyEntryChecksums: false
        )
        XCTAssertEqual(try archive.rawData(named: "weight"), storage)
    }

    func testPinnedVideoDepthAnythingInventoryWhenFixtureIsAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_PTH"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_VDA_PTH to the pinned video_depth_anything_vits.pth fixture."
        )

        let archive = try PyTorchStateDictArchive(url: URL(fileURLWithPath: path))
        XCTAssertEqual(archive.tensors.count, 351)
        XCTAssertEqual(archive.tensors.reduce(0) { $0 + $1.elementCount }, 29_080_193)
        XCTAssertEqual(Set(archive.tensors.map(\.dataType)), [.float32])
        XCTAssertEqual(Set(archive.tensors.map(\.storageKey)).count, 351)

        let patch = try XCTUnwrap(archive.descriptor(named: "pretrained.patch_embed.proj.weight"))
        XCTAssertEqual(patch.shape, [384, 3, 14, 14])
        let temporal = try XCTUnwrap(archive.descriptor(
            named: "head.motion_modules.3.temporal_transformer.transformer_blocks.0.ff.net.0.proj.weight"
        ))
        XCTAssertEqual(temporal.shape, [512, 64])

        let cls = try archive.loadArray(named: "pretrained.cls_token")
        MLX.eval(cls)
        XCTAssertEqual(cls.shape, [1, 1, 384])
        XCTAssertEqual(cls.size, 384)
    }

    func testPinnedMMAudioBigVGANInventoryWhenFixtureIsAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_MMAUDIO_BIGVGAN"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_MMAUDIO_BIGVGAN to the pinned bigvgan_generator.pt fixture."
        )

        let archive = try PyTorchStateDictArchive(url: URL(fileURLWithPath: path))
        XCTAssertEqual(archive.tensors.count, 783)
        XCTAssertNotNil(archive.descriptor(named: "conv_pre.bias"))
        XCTAssertNotNil(archive.descriptor(named: "conv_post.weight_g"))
        XCTAssertNotNil(archive.descriptor(named: "conv_post.weight_v"))
    }

    private func writeCheckpoint(
        pickle: Data,
        storage: Data,
        corruptStorageCRC: Bool = false,
        serializationID: String? = nil
    ) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("state-dict-\(UUID().uuidString).pth")
        var entries = [
            ZIPFixtureEntry(name: "checkpoint/data.pkl", data: pickle),
            ZIPFixtureEntry(name: "checkpoint/data/0", data: storage, corruptCRC: corruptStorageCRC),
            ZIPFixtureEntry(name: "checkpoint/version", data: Data("3\n".utf8)),
        ]
        if let serializationID {
            entries.append(
                ZIPFixtureEntry(
                    name: "checkpoint/.data/serialization_id",
                    data: Data(serializationID.utf8)
                )
            )
        }
        let archive = storedZIP(entries: entries)
        try archive.write(to: url)
        return url
    }
}

private struct ZIPFixtureEntry {
    let name: String
    let data: Data
    var corruptCRC = false
}

private func stateDictPickle(
    shape: [Int],
    stride: [Int],
    storageElementCount: Int,
    storageKey: String = "0",
    storageDevice: String = "cpu",
    rebuildGlobal: (module: String, name: String) = ("torch._utils", "_rebuild_tensor_v2")
) -> Data {
    var data = Data([0x80, 0x02, 0x7D, 0x28]) // PROTO 2, EMPTY_DICT, MARK
    appendUnicode("weight", to: &data)
    appendGlobal(rebuildGlobal.module, rebuildGlobal.name, to: &data)
    data.append(0x28) // REDUCE argument MARK
    data.append(0x28) // persistent storage tuple MARK
    appendUnicode("storage", to: &data)
    appendGlobal("torch", "FloatStorage", to: &data)
    appendUnicode(storageKey, to: &data)
    appendUnicode(storageDevice, to: &data)
    appendInteger(storageElementCount, to: &data)
    data.append(contentsOf: [0x74, 0x51]) // TUPLE, BINPERSID
    appendInteger(0, to: &data)
    appendIntegerTuple(shape, to: &data)
    appendIntegerTuple(stride, to: &data)
    data.append(0x89) // NEWFALSE
    appendGlobal("collections", "OrderedDict", to: &data)
    data.append(contentsOf: [0x29, 0x52]) // EMPTY_TUPLE, REDUCE
    data.append(contentsOf: [0x74, 0x52]) // TUPLE, REDUCE
    data.append(contentsOf: [0x75, 0x2E]) // SETITEMS, STOP
    return data
}

private func generatorStateDictPickle(
    shape: [Int],
    stride: [Int],
    storageElementCount: Int
) -> Data {
    var inner = stateDictPickle(
        shape: shape,
        stride: stride,
        storageElementCount: storageElementCount
    )
    inner.removeFirst(2) // The outer container owns PROTO 2.
    inner.removeLast() // The outer container owns STOP.

    var data = Data([0x80, 0x02, 0x7D, 0x28]) // PROTO 2, EMPTY_DICT, MARK
    appendUnicode("generator", to: &data)
    data.append(inner)
    data.append(contentsOf: [0x75, 0x2E]) // SETITEMS, STOP
    return data
}

private func appendUnicode(_ value: String, to data: inout Data) {
    let bytes = Data(value.utf8)
    data.append(0x58)
    append(UInt32(bytes.count), to: &data)
    data.append(bytes)
}

private func appendGlobal(_ module: String, _ name: String, to data: inout Data) {
    data.append(0x63)
    data.append(Data("\(module)\n\(name)\n".utf8))
}

private func appendInteger(_ value: Int, to data: inout Data) {
    precondition(value >= 0)
    if value <= 0xFF {
        data.append(contentsOf: [0x4B, UInt8(value)])
    } else if value <= 0xFFFF {
        data.append(0x4D)
        append(UInt16(value), to: &data)
    } else {
        data.append(0x4A)
        append(UInt32(value), to: &data)
    }
}

private func appendIntegerTuple(_ values: [Int], to data: inout Data) {
    for value in values { appendInteger(value, to: &data) }
    switch values.count {
    case 0: data.append(0x29)
    case 1: data.append(0x85)
    case 2: data.append(0x86)
    case 3: data.append(0x87)
    default: preconditionFailure("Fixture only needs tuple ranks up to three.")
    }
}

private func floatData(_ values: [Float]) -> Data {
    values.reduce(into: Data()) { data, value in append(value.bitPattern, to: &data) }
}

private func storedZIP(entries: [ZIPFixtureEntry]) -> Data {
    struct CentralEntry {
        let name: String
        let size: UInt32
        let crc: UInt32
        let localOffset: UInt32
    }

    var archive = Data()
    var central: [CentralEntry] = []
    for entry in entries {
        let name = Data(entry.name.utf8)
        let size = UInt32(entry.data.count)
        let actualCRC = fixtureCRC32(entry.data)
        let crc = entry.corruptCRC ? actualCRC ^ UInt32.max : actualCRC
        let localOffset = UInt32(archive.count)
        append(UInt32(0x0403_4B50), to: &archive)
        append(UInt16(20), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(crc, to: &archive)
        append(size, to: &archive)
        append(size, to: &archive)
        append(UInt16(name.count), to: &archive)
        append(UInt16(0), to: &archive)
        archive.append(name)
        archive.append(entry.data)
        central.append(CentralEntry(name: entry.name, size: size, crc: crc, localOffset: localOffset))
    }

    let centralOffset = UInt32(archive.count)
    for entry in central {
        let name = Data(entry.name.utf8)
        append(UInt32(0x0201_4B50), to: &archive)
        append(UInt16(20), to: &archive)
        append(UInt16(20), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(entry.crc, to: &archive)
        append(entry.size, to: &archive)
        append(entry.size, to: &archive)
        append(UInt16(name.count), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt32(0), to: &archive)
        append(entry.localOffset, to: &archive)
        archive.append(name)
    }
    let centralSize = UInt32(archive.count) - centralOffset
    append(UInt32(0x0605_4B50), to: &archive)
    append(UInt16(0), to: &archive)
    append(UInt16(0), to: &archive)
    append(UInt16(central.count), to: &archive)
    append(UInt16(central.count), to: &archive)
    append(centralSize, to: &archive)
    append(centralOffset, to: &archive)
    append(UInt16(0), to: &archive)
    return archive
}

private func fixtureCRC32(_ data: Data) -> UInt32 {
    var crc = UInt32.max
    for byte in data {
        crc ^= UInt32(byte)
        for _ in 0..<8 { crc = (crc >> 1) ^ (0xEDB8_8320 & (0 &- (crc & 1))) }
    }
    return ~crc
}

private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
}
