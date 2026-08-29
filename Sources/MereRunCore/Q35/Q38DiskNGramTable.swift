import Dispatch
import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

/// Disk-backed PLE row gathering informed by ddalcu/mlx-serve's qwen4_exp.zig
/// at 09970f9bd3051fa2f39fc03e7df3d4a937c37e2e (MIT, David Dalcu).
/// See THIRD_PARTY_NOTICES.md. Unlike its merged-table CPU dequantizer, this
/// reads existing sharded packs and uses MLX dequantization for exact parity.
final class Q38DiskNGramTable: @unchecked Sendable {
    enum LoadError: LocalizedError {
        case invalidLayout(String)

        var errorDescription: String? {
            switch self {
            case .invalidLayout(let key): return "Unsupported Qwen PLE table layout: \(key)"
            }
        }
    }

    /// Immutable read-only mapping, never handed to an MLX array. The OS may
    /// evict its clean pages; only selected rows enter the MLX allocator.
    private final class MappedFile: @unchecked Sendable {
        let bytes: UnsafeMutableRawPointer
        let count: Int
        let metadata: [String: SafetensorsStreamingLoader.TensorMetadata]

        init(url: URL) throws {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            let count = Int(try handle.seekToEnd())
            guard count >= 8 else { throw SafetensorsStreamingLoader.LoaderError.fileTooSmall(url) }
            let mapped = mmap(nil, count, PROT_READ, MAP_PRIVATE, handle.fileDescriptor, 0)
            guard let mapped, mapped != MAP_FAILED else {
                throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
            }
            do {
                let data = Data(bytesNoCopy: mapped, count: count, deallocator: .none)
                self.metadata = try SafetensorsStreamingLoader.metadata(fileData: data, fileURL: url)
            } catch {
                munmap(mapped, count)
                throw error
            }
            self.bytes = mapped
            self.count = count
        }

        deinit { munmap(bytes, count) }
    }

    private struct Tensor: Sendable {
        let file: MappedFile
        let metadata: SafetensorsStreamingLoader.TensorMetadata
        let rowBytes: Int

        init(file: MappedFile, key: String) throws {
            guard let metadata = file.metadata[key], metadata.shape.count == 2,
                  metadata.shape.allSatisfy({ $0 > 0 }),
                  metadata.endOffset - metadata.startOffset
                    == metadata.shape[0] * metadata.shape[1] * metadata.dtype.size else {
                throw LoadError.invalidLayout(key)
            }
            self.file = file
            self.metadata = metadata
            self.rowBytes = metadata.shape[1] * metadata.dtype.size
        }

        func copy(row: Int, to destination: UnsafeMutableRawPointer) {
            destination.copyMemory(
                from: file.bytes.advanced(by: metadata.startOffset + row * rowBytes), byteCount: rowBytes
            )
        }
    }

    private struct Shard: Sendable {
        let rowOffset: Int
        let weight: Tensor
        let scales: Tensor
        let biases: Tensor
    }

    /// Each worker owns disjoint row ranges in these temporary output buffers.
    private struct RowBuffers: @unchecked Sendable {
        let weight: UnsafeMutableRawPointer
        let scales: UnsafeMutableRawPointer
        let biases: UnsafeMutableRawPointer
    }

    private let shards: [Shard]
    let rowCount: Int
    let dimensions: Int
    let bits: Int
    let groupSize: Int
    let dtype: DType
    let tableByteCount: Int

    init(indexURL: URL, base: String, shardCount: Int, dimensions: Int, minimumRowCount: Int) throws {
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: Data(contentsOf: indexURL))
        let root = indexURL.deletingLastPathComponent()
        var files: [String: MappedFile] = [:]
        func tensor(_ key: String) throws -> Tensor {
            guard let filename = index.weightMap[key] else { throw Q35Error.missingFiles([key]) }
            let file: MappedFile
            if let existing = files[filename] {
                file = existing
            } else {
                file = try MappedFile(url: root.appendingPathComponent(filename))
                files[filename] = file
            }
            return try Tensor(file: file, key: key)
        }
        var shards: [Shard] = []
        var rows = 0
        for part in 0..<shardCount {
            let key = "\(base).shard_\(part)"
            let weight = try tensor("\(key).weight")
            let scales = try tensor("\(key).scales")
            let biases = try tensor("\(key).biases")
            guard weight.metadata.dtype == .uint32,
                  [.bfloat16, .float16, .float32].contains(scales.metadata.dtype),
                  scales.metadata.dtype == biases.metadata.dtype,
                  scales.metadata.shape == biases.metadata.shape,
                  weight.metadata.shape[0] == scales.metadata.shape[0] else {
                throw LoadError.invalidLayout(key)
            }
            if let first = shards.first {
                guard weight.rowBytes == first.weight.rowBytes,
                      scales.rowBytes == first.scales.rowBytes,
                      scales.metadata.dtype == first.scales.metadata.dtype else {
                    throw LoadError.invalidLayout(key)
                }
            }
            shards.append(Shard(rowOffset: rows, weight: weight, scales: scales, biases: biases))
            rows += weight.metadata.shape[0]
        }
        guard let first = shards.first, dimensions > 0, rows >= minimumRowCount,
              (first.weight.rowBytes * 8).isMultiple(of: dimensions),
              dimensions.isMultiple(of: first.scales.metadata.shape[1]) else {
            throw LoadError.invalidLayout(base)
        }
        let bits = first.weight.rowBytes * 8 / dimensions
        let groupSize = dimensions / first.scales.metadata.shape[1]
        guard [2, 3, 4, 5, 6, 8].contains(bits), [32, 64, 128].contains(groupSize) else {
            throw LoadError.invalidLayout(base)
        }
        self.shards = shards
        self.rowCount = rows
        self.dimensions = dimensions
        self.bits = bits
        self.groupSize = groupSize
        self.dtype = first.scales.metadata.dtype
        self.tableByteCount = rows * (first.weight.rowBytes + first.scales.rowBytes + first.biases.rowBytes)
    }

    func lookup(_ ids: [Int32]) -> MLXArray {
        precondition(ids.allSatisfy { $0 >= 0 && Int($0) < rowCount })
        guard !ids.isEmpty else { return MLXArray.zeros([0, dimensions], dtype: dtype) }
        let first = shards[0]
        var weights = Data(count: ids.count * first.weight.rowBytes)
        var scales = Data(count: ids.count * first.scales.rowBytes)
        var biases = Data(count: ids.count * first.biases.rowBytes)
        weights.withUnsafeMutableBytes { weight in
            scales.withUnsafeMutableBytes { scale in
                biases.withUnsafeMutableBytes { bias in
                    let buffers = RowBuffers(weight: weight.baseAddress!, scales: scale.baseAddress!, biases: bias.baseAddress!)
                    let workers = min(16, ids.count)
                    DispatchQueue.concurrentPerform(iterations: workers) { worker in
                        for position in stride(from: worker, to: ids.count, by: workers) {
                            let id = Int(ids[position])
                            let shard = self.shard(containing: id)
                            let row = id - shard.rowOffset
                            shard.weight.copy(row: row, to: buffers.weight.advanced(by: position * shard.weight.rowBytes))
                            shard.scales.copy(row: row, to: buffers.scales.advanced(by: position * shard.scales.rowBytes))
                            shard.biases.copy(row: row, to: buffers.biases.advanced(by: position * shard.biases.rowBytes))
                        }
                    }
                }
            }
        }
        return MLX.dequantized(
            MLXArray(weights, [ids.count, first.weight.metadata.shape[1]], dtype: .uint32),
            scales: MLXArray(scales, [ids.count, dimensions / groupSize], dtype: dtype),
            biases: MLXArray(biases, [ids.count, dimensions / groupSize], dtype: dtype),
            groupSize: groupSize, bits: bits
        )
    }

    private func shard(containing id: Int) -> Shard {
        var low = 0
        var high = shards.count
        while low + 1 < high {
            let middle = (low + high) / 2
            if shards[middle].rowOffset <= id { low = middle } else { high = middle }
        }
        return shards[low]
    }
}
