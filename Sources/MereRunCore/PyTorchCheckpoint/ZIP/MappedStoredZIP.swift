import Foundation

struct StoredZIPEntry {
    let name: String
    let flags: UInt16
    let compressionMethod: UInt16
    let crc32: UInt32
    let uncompressedSize: Int
    let dataRange: Range<Int>
}

final class MappedStoredZIP: @unchecked Sendable {
    private static let crc32Table: [UInt32] = (0..<256).map { value in
        var entry = UInt32(value)
        for _ in 0..<8 {
            entry = (entry >> 1) ^ (0xEDB8_8320 & (0 &- (entry & 1)))
        }
        return entry
    }

    let mappedData: Data
    let entries: [String: StoredZIPEntry]
    private let verifyEntryChecksums: Bool

    init(url: URL, verifyEntryChecksums: Bool = true) throws {
        let data = try Data(contentsOf: url, options: [.alwaysMapped])
        let eocd = try Self.endOfCentralDirectory(in: data)
        guard try data.uint16(at: eocd + 4) == 0, try data.uint16(at: eocd + 6) == 0 else {
            throw PyTorchStateDictError.malformedZIP("multi-disk archives are rejected")
        }
        let entryCount = Int(try data.uint16(at: eocd + 10))
        let centralSize = Int(try data.uint32(at: eocd + 12))
        let centralOffset = Int(try data.uint32(at: eocd + 16))
        guard entryCount != 0xFFFF, centralSize != Int(UInt32.max), centralOffset != Int(UInt32.max),
              centralOffset <= data.count - centralSize else {
            throw PyTorchStateDictError.malformedZIP("unsupported ZIP64 central-directory sentinel")
        }
        try Self.validateZIP64Trailer(data, centralEnd: centralOffset + centralSize, eocd: eocd,
                                      count: entryCount, size: centralSize, offset: centralOffset)

        var result: [String: StoredZIPEntry] = [:]
        var cursor = centralOffset
        for _ in 0..<entryCount {
            guard try data.uint32(at: cursor) == 0x0201_4B50 else {
                throw PyTorchStateDictError.malformedZIP("invalid central-directory signature")
            }
            let flags = try data.uint16(at: cursor + 8)
            let method = try data.uint16(at: cursor + 10)
            let crc = try data.uint32(at: cursor + 16)
            let compressed = Int(try data.uint32(at: cursor + 20))
            let uncompressed = Int(try data.uint32(at: cursor + 24))
            let nameLength = Int(try data.uint16(at: cursor + 28))
            let extraLength = Int(try data.uint16(at: cursor + 30))
            let commentLength = Int(try data.uint16(at: cursor + 32))
            let localOffset = Int(try data.uint32(at: cursor + 42))
            let nameStart = cursor + 46
            guard nameStart <= data.count - nameLength else {
                throw PyTorchStateDictError.malformedZIP("central filename exceeds archive")
            }
            let nameData = data.subdata(in: nameStart..<(nameStart + nameLength))
            guard let name = String(data: nameData, encoding: .utf8), Self.safe(name) else {
                throw PyTorchStateDictError.unsafeArchivePath(String(decoding: nameData, as: UTF8.self))
            }
            guard result[name] == nil else { throw PyTorchStateDictError.duplicateArchiveEntry(name) }
            guard flags & ~UInt16(0x0808) == 0 else {
                throw PyTorchStateDictError.malformedZIP("entry '\(name)' uses unsupported flags")
            }
            guard method == 0, compressed == uncompressed else {
                throw PyTorchStateDictError.unsupportedCompression(name: name, method: method)
            }
            guard try data.uint32(at: localOffset) == 0x0403_4B50,
                  try data.uint16(at: localOffset + 6) == flags,
                  try data.uint16(at: localOffset + 8) == method else {
                throw PyTorchStateDictError.malformedZIP("entry '\(name)' has an inconsistent local header")
            }
            let localNameLength = Int(try data.uint16(at: localOffset + 26))
            let localExtraLength = Int(try data.uint16(at: localOffset + 28))
            let dataStart = localOffset + 30 + localNameLength + localExtraLength
            guard localNameLength == nameLength, dataStart <= data.count - uncompressed,
                  data.subdata(in: (localOffset + 30)..<(localOffset + 30 + localNameLength)) == nameData else {
                throw PyTorchStateDictError.malformedZIP("entry '\(name)' has an invalid local data range")
            }
            result[name] = StoredZIPEntry(
                name: name,
                flags: flags,
                compressionMethod: method,
                crc32: crc,
                uncompressedSize: uncompressed,
                dataRange: dataStart..<(dataStart + uncompressed)
            )
            cursor = nameStart + nameLength + extraLength + commentLength
        }
        guard cursor == centralOffset + centralSize else {
            throw PyTorchStateDictError.malformedZIP("central-directory size mismatch")
        }
        self.mappedData = data
        self.entries = result
        self.verifyEntryChecksums = verifyEntryChecksums
    }

    func utf8Entry(named name: String) throws -> String {
        guard let entry = entries[name] else { throw PyTorchStateDictError.malformedZIP("missing \(name)") }
        let bytes = try data(for: entry)
        guard let value = String(data: bytes, encoding: .utf8) else {
            throw PyTorchStateDictError.malformedZIP("entry '\(name)' is not UTF-8")
        }
        return value
    }

    func data(for entry: StoredZIPEntry) throws -> Data {
        mappedData.subdata(in: try validatedDataRange(for: entry))
    }

    func validatedDataRange(for entry: StoredZIPEntry) throws -> Range<Int> {
        guard !verifyEntryChecksums || Self.crc32(mappedData, range: entry.dataRange) == entry.crc32 else {
            throw PyTorchStateDictError.checksumMismatch(entry.name)
        }
        return entry.dataRange
    }

    private static func safe(_ name: String) -> Bool {
        !name.isEmpty && !name.hasPrefix("/") && !name.contains("\\") && !name.contains("\0")
            && name.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { $0 != "" && $0 != "." && $0 != ".." }
    }

    private static func endOfCentralDirectory(in data: Data) throws -> Int {
        guard data.count >= 22 else { throw PyTorchStateDictError.malformedZIP("archive is too small") }
        let minimum = max(0, data.count - 22 - 65_535)
        var cursor = data.count - 22
        while cursor >= minimum {
            if try data.uint32(at: cursor) == 0x0605_4B50 {
                let commentLength = Int(try data.uint16(at: cursor + 20))
                if cursor + 22 + commentLength == data.count { return cursor }
            }
            cursor -= 1
        }
        throw PyTorchStateDictError.malformedZIP("missing end-of-central-directory record")
    }

    private static func validateZIP64Trailer(
        _ data: Data,
        centralEnd: Int,
        eocd: Int,
        count: Int,
        size: Int,
        offset: Int
    ) throws {
        if centralEnd == eocd { return }
        guard centralEnd + 76 == eocd,
              try data.uint32(at: centralEnd) == 0x0606_4B50,
              try data.uint64(at: centralEnd + 4) == 44,
              try data.uint64(at: centralEnd + 24) == UInt64(count),
              try data.uint64(at: centralEnd + 32) == UInt64(count),
              try data.uint64(at: centralEnd + 40) == UInt64(size),
              try data.uint64(at: centralEnd + 48) == UInt64(offset),
              try data.uint32(at: centralEnd + 56) == 0x0706_4B50,
              try data.uint64(at: centralEnd + 64) == UInt64(centralEnd),
              try data.uint32(at: centralEnd + 72) == 1 else {
            throw PyTorchStateDictError.malformedZIP("invalid ZIP64 trailer")
        }
    }

    private static func crc32(_ data: Data, range: Range<Int>) -> UInt32 {
        var crc = UInt32.max
        for index in range {
            let tableIndex = Int((crc ^ UInt32(data[index])) & 0xFF)
            crc = (crc >> 8) ^ crc32Table[tableIndex]
        }
        return ~crc
    }
}

private extension Data {
    func uint16(at offset: Int) throws -> UInt16 {
        guard offset >= 0, offset <= count - 2 else {
            throw PyTorchStateDictError.malformedZIP("unexpected end of archive")
        }
        return UInt16(self[offset]) | (UInt16(self[offset + 1]) << 8)
    }

    func uint32(at offset: Int) throws -> UInt32 {
        UInt32(try uint16(at: offset)) | (UInt32(try uint16(at: offset + 2)) << 16)
    }

    func uint64(at offset: Int) throws -> UInt64 {
        UInt64(try uint32(at: offset)) | (UInt64(try uint32(at: offset + 4)) << 32)
    }
}
