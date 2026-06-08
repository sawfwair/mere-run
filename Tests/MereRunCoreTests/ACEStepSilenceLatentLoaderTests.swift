import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class ACEStepSilenceLatentLoaderTests: XCTestCase {
    func testLoadsPyTorchZipSilenceLatentAsFrameMajorMLXArray() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("acestep-silence-latent-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: root)
        }

        var tensorData = Data()
        for value in [Float](arrayLiteral: 1, 2, 3, 4, 5, 6) {
            appendFloat32(value, to: &tensorData)
        }

        let archive = makeStoredZip(entries: [
            ("silence_latent/byteorder", Data("little".utf8)),
            ("silence_latent/data/0", tensorData)
        ])
        try archive.write(to: root.appendingPathComponent("silence_latent.pt"))

        let loaded = try XCTUnwrap(
            try ACEStepCheckpointLoader.loadSilenceLatentIfPresent(
                resources: ACEStepResources(rootURL: root),
                latentDim: 2,
                dtype: .float32
            )
        )
        MLX.eval(loaded)

        XCTAssertEqual(loaded.shape, [1, 3, 2])
        XCTAssertEqual(loaded.asArray(Float.self), [1, 4, 2, 5, 3, 6])
    }
}

private func makeStoredZip(entries: [(name: String, data: Data)]) -> Data {
    struct CentralDirectoryEntry {
        var name: String
        var size: UInt32
        var localHeaderOffset: UInt32
    }

    var archive = Data()
    var centralDirectoryEntries: [CentralDirectoryEntry] = []

    for entry in entries {
        let nameData = Data(entry.name.utf8)
        let size = UInt32(entry.data.count)
        let localHeaderOffset = UInt32(archive.count)

        append(UInt32(0x0403_4b50), to: &archive)
        append(UInt16(20), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt32(0), to: &archive)
        append(size, to: &archive)
        append(size, to: &archive)
        append(UInt16(nameData.count), to: &archive)
        append(UInt16(0), to: &archive)
        archive.append(nameData)
        archive.append(entry.data)

        centralDirectoryEntries.append(
            CentralDirectoryEntry(
                name: entry.name,
                size: size,
                localHeaderOffset: localHeaderOffset
            )
        )
    }

    let centralDirectoryOffset = UInt32(archive.count)
    for entry in centralDirectoryEntries {
        let nameData = Data(entry.name.utf8)

        append(UInt32(0x0201_4b50), to: &archive)
        append(UInt16(20), to: &archive)
        append(UInt16(20), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt32(0), to: &archive)
        append(entry.size, to: &archive)
        append(entry.size, to: &archive)
        append(UInt16(nameData.count), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt16(0), to: &archive)
        append(UInt32(0), to: &archive)
        append(entry.localHeaderOffset, to: &archive)
        archive.append(nameData)
    }

    let centralDirectorySize = UInt32(archive.count) - centralDirectoryOffset
    append(UInt32(0x0605_4b50), to: &archive)
    append(UInt16(0), to: &archive)
    append(UInt16(0), to: &archive)
    append(UInt16(centralDirectoryEntries.count), to: &archive)
    append(UInt16(centralDirectoryEntries.count), to: &archive)
    append(centralDirectorySize, to: &archive)
    append(centralDirectoryOffset, to: &archive)
    append(UInt16(0), to: &archive)

    return archive
}

private func appendFloat32(_ value: Float, to data: inout Data) {
    append(value.bitPattern, to: &data)
}

private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { bytes in
        data.append(bytes.bindMemory(to: UInt8.self))
    }
}
