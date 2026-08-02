import Foundation
import MLX

public enum PyTorchTensorDataType: String, Codable, Equatable, Hashable, Sendable {
    case bfloat16
    case bool
    case float16
    case float32
    case int8
    case int16
    case int32
    case int64
    case uint8

    public var byteCount: Int {
        switch self {
        case .bool, .int8, .uint8: 1
        case .bfloat16, .float16, .int16: 2
        case .float32, .int32: 4
        case .int64: 8
        }
    }

    fileprivate var mlxDType: DType {
        switch self {
        case .bfloat16: .bfloat16
        case .bool: .bool
        case .float16: .float16
        case .float32: .float32
        case .int8: .int8
        case .int16: .int16
        case .int32: .int32
        case .int64: .int64
        case .uint8: .uint8
        }
    }
}

public struct PyTorchTensorDescriptor: Equatable, Sendable {
    public let name: String
    public let shape: [Int]
    public let stride: [Int]
    public let dataType: PyTorchTensorDataType
    public let storageKey: String
    public let storageOffset: Int
    public let storageElementCount: Int
    public let requiresGradient: Bool

    public var elementCount: Int { shape.reduce(1, *) }
    public var byteCount: Int { elementCount * dataType.byteCount }
}

public enum PyTorchStateDictError: Error, Equatable, LocalizedError, Sendable {
    case malformedZIP(String)
    case unsafeArchivePath(String)
    case duplicateArchiveEntry(String)
    case unsupportedCompression(name: String, method: UInt16)
    case checksumMismatch(String)
    case missingDataPickle
    case unsupportedArchiveVersion(String)
    case unsupportedByteOrder(String)
    case pickleTooLarge(Int)
    case unsupportedPickleProtocol(Int)
    case unsupportedPickleOpcode(UInt8, offset: Int)
    case unsupportedPickleGlobal(module: String, name: String)
    case malformedPickle(String)
    case duplicateTensorName(String)
    case unsupportedStorageType(String)
    case missingStorage(String)
    case unusedStorage(String)
    case storageByteCountMismatch(key: String, expected: Int, actual: Int)
    case nonContiguousTensor(name: String, shape: [Int], stride: [Int])
    case tensorOutOfBounds(String)
    case unknownTensorDescriptor(String)

    public var errorDescription: String? {
        switch self {
        case .malformedZIP(let detail): "Malformed PyTorch ZIP archive: \(detail)"
        case .unsafeArchivePath(let path): "Unsafe PyTorch ZIP entry path '\(path)'."
        case .duplicateArchiveEntry(let name): "Duplicate PyTorch ZIP entry '\(name)'."
        case .unsupportedCompression(let name, let method):
            "PyTorch ZIP entry '\(name)' uses unsupported compression method \(method)."
        case .checksumMismatch(let name): "PyTorch ZIP entry '\(name)' failed its CRC-32 check."
        case .missingDataPickle: "PyTorch ZIP archive contains no unique data.pkl entry."
        case .unsupportedArchiveVersion(let version): "Unsupported PyTorch archive version '\(version)'."
        case .unsupportedByteOrder(let order): "Unsupported PyTorch checkpoint byte order '\(order)'."
        case .pickleTooLarge(let count): "PyTorch data.pkl exceeds the 64 MiB metadata limit (\(count) bytes)."
        case .unsupportedPickleProtocol(let version): "Unsupported pickle protocol \(version); expected protocol 2."
        case .unsupportedPickleOpcode(let opcode, let offset):
            "Unsupported pickle opcode 0x\(String(opcode, radix: 16)) at offset \(offset)."
        case .unsupportedPickleGlobal(let module, let name): "Rejected pickle global '\(module).\(name)'."
        case .malformedPickle(let detail): "Malformed restricted PyTorch state-dict pickle: \(detail)"
        case .duplicateTensorName(let name): "Duplicate state-dict tensor name '\(name)'."
        case .unsupportedStorageType(let name): "Unsupported PyTorch storage type '\(name)'."
        case .missingStorage(let key): "Missing PyTorch storage blob '\(key)'."
        case .unusedStorage(let key): "Unreferenced PyTorch storage blob '\(key)'."
        case .storageByteCountMismatch(let key, let expected, let actual):
            "PyTorch storage '\(key)' expected \(expected) bytes but contains \(actual)."
        case .nonContiguousTensor(let name, _, _): "PyTorch tensor '\(name)' is not C-contiguous."
        case .tensorOutOfBounds(let name): "PyTorch tensor '\(name)' exceeds its backing storage."
        case .unknownTensorDescriptor(let name): "Tensor descriptor '\(name)' does not belong to this archive."
        }
    }
}

/// A non-executing reader for the narrow ZIP + protocol-2 state-dict format emitted by `torch.save`.
///
/// The archive is memory mapped. Pickle globals are symbolic and whitelisted; no Python callable,
/// reducer, object constructor, or import is ever executed.
public final class PyTorchStateDictArchive: @unchecked Sendable {
    private let zip: MappedStoredZIP
    private let storageEntryByKey: [String: StoredZIPEntry]
    private let tensorByName: [String: PyTorchTensorDescriptor]
    public let tensors: [PyTorchTensorDescriptor]

    /// `verifyEntryChecksums` must remain enabled for arbitrary input. It may
    /// be disabled only when the caller has just verified a trusted whole-file
    /// digest, which supersedes the much slower per-entry CRC32 pass.
    public init(url: URL, verifyEntryChecksums: Bool = true) throws {
        let zip = try MappedStoredZIP(
            url: url,
            verifyEntryChecksums: verifyEntryChecksums
        )
        let pickleEntries = zip.entries.values.filter { $0.name.hasSuffix("/data.pkl") }
        guard pickleEntries.count == 1, let pickleEntry = pickleEntries.first else {
            throw PyTorchStateDictError.missingDataPickle
        }
        let root = String(pickleEntry.name.dropLast("/data.pkl".count))
        guard !root.isEmpty else { throw PyTorchStateDictError.missingDataPickle }

        let version = try zip.utf8Entry(named: "\(root)/version").trimmingCharacters(in: .whitespacesAndNewlines)
        guard version == "3" else { throw PyTorchStateDictError.unsupportedArchiveVersion(version) }
        if zip.entries["\(root)/byteorder"] != nil {
            let byteOrder = try zip.utf8Entry(named: "\(root)/byteorder")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard byteOrder == "little" else { throw PyTorchStateDictError.unsupportedByteOrder(byteOrder) }
        }

        let pickle = try zip.data(for: pickleEntry)
        guard pickle.count <= 64 * 1_024 * 1_024 else {
            throw PyTorchStateDictError.pickleTooLarge(pickle.count)
        }
        var parser = RestrictedStateDictPickle(data: pickle)
        let parsed = try parser.parse()
        var descriptors: [PyTorchTensorDescriptor] = []
        var lookup: [String: PyTorchTensorDescriptor] = [:]
        var storages: [String: ParsedStorage] = [:]
        for item in parsed {
            guard lookup[item.name] == nil else { throw PyTorchStateDictError.duplicateTensorName(item.name) }
            let descriptor = try Self.validate(item)
            if let previous = storages[item.storage.key], previous != item.storage {
                throw PyTorchStateDictError.malformedPickle("storage '\(item.storage.key)' has conflicting metadata")
            }
            storages[item.storage.key] = item.storage
            lookup[item.name] = descriptor
            descriptors.append(descriptor)
        }

        let dataPrefix = "\(root)/data/"
        let storageEntries = zip.entries.values.filter { $0.name.hasPrefix(dataPrefix) }
        var entryByKey: [String: StoredZIPEntry] = [:]
        for entry in storageEntries {
            let key = String(entry.name.dropFirst(dataPrefix.count))
            guard Self.isASCIIDecimal(key) else {
                throw PyTorchStateDictError.unsafeArchivePath(entry.name)
            }
            entryByKey[key] = entry
        }
        for (key, storage) in storages {
            guard let entry = entryByKey[key] else { throw PyTorchStateDictError.missingStorage(key) }
            let expected = try Self.checkedProduct(storage.elementCount, storage.dataType.byteCount)
            guard entry.uncompressedSize == expected else {
                throw PyTorchStateDictError.storageByteCountMismatch(
                    key: key,
                    expected: expected,
                    actual: entry.uncompressedSize
                )
            }
        }
        for key in entryByKey.keys where storages[key] == nil {
            throw PyTorchStateDictError.unusedStorage(key)
        }

        var allowedNames = Set(
            [pickleEntry.name, "\(root)/version", "\(root)/byteorder"]
                + entryByKey.values.map(\.name)
        )
        // PyTorch 2.6+ may add this inert decimal identifier. Accept only the
        // exact root-relative name and a small ASCII-decimal payload; no other
        // hidden metadata or archive entry is admitted by the state-dict reader.
        let serializationIDName = "\(root)/.data/serialization_id"
        if zip.entries[serializationIDName] != nil {
            let identifier = try zip.utf8Entry(named: serializationIDName)
            guard !identifier.isEmpty,
                  identifier.utf8.count <= 128,
                  identifier.utf8.allSatisfy({ (0x30...0x39).contains($0) }) else {
                throw PyTorchStateDictError.malformedZIP("invalid serialization identifier")
            }
            allowedNames.insert(serializationIDName)
        }
        // PyTorch 2.8+ adds two inert root metadata files for its ZIP writer.
        // Admit only the exact filenames and the currently documented scalar
        // forms; they do not influence tensor decoding or storage addressing.
        let formatVersionName = "\(root)/.format_version"
        if zip.entries[formatVersionName] != nil {
            let formatVersion = try zip.utf8Entry(named: formatVersionName)
            guard formatVersion == "1" else {
                throw PyTorchStateDictError.malformedZIP("invalid ZIP format version")
            }
            allowedNames.insert(formatVersionName)
        }
        let storageAlignmentName = "\(root)/.storage_alignment"
        if zip.entries[storageAlignmentName] != nil {
            let rawAlignment = try zip.utf8Entry(named: storageAlignmentName)
            guard rawAlignment.utf8.count <= 4,
                  rawAlignment.utf8.allSatisfy({ (0x30...0x39).contains($0) }),
                  let alignment = Int(rawAlignment),
                  alignment > 0,
                  alignment <= 4_096,
                  alignment.nonzeroBitCount == 1 else {
                throw PyTorchStateDictError.malformedZIP("invalid storage alignment")
            }
            allowedNames.insert(storageAlignmentName)
        }
        if let unexpected = zip.entries.keys.first(where: { !allowedNames.contains($0) }) {
            throw PyTorchStateDictError.unsafeArchivePath(unexpected)
        }

        self.zip = zip
        self.storageEntryByKey = entryByKey
        self.tensorByName = lookup
        self.tensors = descriptors
    }

    public func descriptor(named name: String) -> PyTorchTensorDescriptor? {
        tensorByName[name]
    }

    public func rawData(for descriptor: PyTorchTensorDescriptor) throws -> Data {
        guard tensorByName[descriptor.name] == descriptor else {
            throw PyTorchStateDictError.unknownTensorDescriptor(descriptor.name)
        }
        guard let entry = storageEntryByKey[descriptor.storageKey] else {
            throw PyTorchStateDictError.missingStorage(descriptor.storageKey)
        }
        let storageRange = try zip.validatedDataRange(for: entry)
        let byteOffset = try Self.checkedProduct(descriptor.storageOffset, descriptor.dataType.byteCount)
        guard byteOffset <= storageRange.count, descriptor.byteCount <= storageRange.count - byteOffset else {
            throw PyTorchStateDictError.tensorOutOfBounds(descriptor.name)
        }
        let start = storageRange.lowerBound + byteOffset
        return zip.mappedData.subdata(in: start..<(start + descriptor.byteCount))
    }

    public func rawData(named name: String) throws -> Data {
        guard let descriptor = tensorByName[name] else {
            throw PyTorchStateDictError.unknownTensorDescriptor(name)
        }
        return try rawData(for: descriptor)
    }

    public func loadArray(for descriptor: PyTorchTensorDescriptor, dtype: DType? = nil) throws -> MLXArray {
        let source = MLXArray(try rawData(for: descriptor), descriptor.shape, dtype: descriptor.dataType.mlxDType)
        guard let dtype, dtype != source.dtype else { return source }
        return source.asType(dtype)
    }

    public func loadArray(named name: String, dtype: DType? = nil) throws -> MLXArray {
        guard let descriptor = tensorByName[name] else {
            throw PyTorchStateDictError.unknownTensorDescriptor(name)
        }
        return try loadArray(for: descriptor, dtype: dtype)
    }

    private static func validate(_ tensor: ParsedTensor) throws -> PyTorchTensorDescriptor {
        guard tensor.shape.count == tensor.stride.count, tensor.shape.count <= 32 else {
            throw PyTorchStateDictError.malformedPickle("tensor '\(tensor.name)' has invalid rank metadata")
        }
        guard tensor.shape.allSatisfy({ $0 > 0 }), tensor.storageOffset >= 0 else {
            throw PyTorchStateDictError.tensorOutOfBounds(tensor.name)
        }
        var expectedStride = 1
        for index in tensor.shape.indices.reversed() {
            guard tensor.stride[index] == expectedStride else {
                throw PyTorchStateDictError.nonContiguousTensor(
                    name: tensor.name,
                    shape: tensor.shape,
                    stride: tensor.stride
                )
            }
            expectedStride = try checkedProduct(expectedStride, tensor.shape[index])
        }
        guard tensor.storageOffset <= tensor.storage.elementCount,
              expectedStride <= tensor.storage.elementCount - tensor.storageOffset else {
            throw PyTorchStateDictError.tensorOutOfBounds(tensor.name)
        }
        return PyTorchTensorDescriptor(
            name: tensor.name,
            shape: tensor.shape,
            stride: tensor.stride,
            dataType: tensor.storage.dataType,
            storageKey: tensor.storage.key,
            storageOffset: tensor.storageOffset,
            storageElementCount: tensor.storage.elementCount,
            requiresGradient: tensor.requiresGradient
        )
    }

    private static func checkedProduct(_ lhs: Int, _ rhs: Int) throws -> Int {
        let result = lhs.multipliedReportingOverflow(by: rhs)
        guard !result.overflow else { throw PyTorchStateDictError.malformedPickle("integer overflow") }
        return result.partialValue
    }

    private static func isASCIIDecimal(_ value: String) -> Bool {
        !value.isEmpty && value.utf8.allSatisfy { (0x30...0x39).contains($0) }
    }
}
