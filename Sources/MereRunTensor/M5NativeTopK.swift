#if os(macOS)
import Metal

/// Exact BF16 top-16 for wide, contiguous vocabulary rows.
/// A caller owns command ordering and must read results before reusing this workspace.
@available(macOS 26.0, *)
package final class M5NativeTopK {
    package static let count = 16
    private static let spans = 16
    private static let lanes = 256
    package static let recordWords = 2 + count * 2

    private let span: any MTLComputePipelineState
    private let merge: any MTLComputePipelineState
    private let partials: any MTLBuffer
    package let output: any MTLBuffer

    enum Failure: Error { case allocation, missingFunction }

    package init(device: any MTLDevice) throws {
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source)
        guard let span = library.makeFunction(name: "tf_bf16_topk_span"),
              let merge = library.makeFunction(name: "tf_bf16_topk_merge") else { throw Failure.missingFunction }
        self.span = try device.makeComputePipelineState(function: span)
        self.merge = try device.makeComputePipelineState(function: merge)
        let rowBytes = Self.recordWords * MemoryLayout<UInt32>.stride
        guard let partials = device.makeBuffer(length: 16 * Self.spans * rowBytes, options: .storageModePrivate),
              let output = device.makeBuffer(length: 16 * rowBytes, options: .storageModeShared) else { throw Failure.allocation }
        self.partials = partials
        self.output = output
    }

    package func encode(into encoder: any MTLComputeCommandEncoder, logits: any MTLBuffer,
                        rows: Int, vocabulary: Int, offset: Int = 0, skippingTokenIDs: Range<Int>? = nil) {
        precondition((1...16).contains(rows) && vocabulary >= 4096)
        precondition(offset >= 0 && offset.isMultiple(of: 2) && offset + rows * vocabulary * 2 <= logits.length)
        let gap = skippingTokenIDs ?? vocabulary..<vocabulary
        precondition(gap.lowerBound >= 0 && gap.lowerBound <= vocabulary)
        let parameters = [rows, vocabulary, Self.count, vocabulary, gap.lowerBound, gap.count].map(UInt32.init)
        let lanes = MTLSize(width: Self.lanes, height: 1, depth: 1)
        parameters.withUnsafeBytes { bytes in
            encoder.setComputePipelineState(span)
            encoder.setBuffer(logits, offset: offset, index: 0)
            encoder.setBuffer(partials, offset: 0, index: 1)
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: Self.spans, height: rows, depth: 1), threadsPerThreadgroup: lanes)
            encoder.memoryBarrier(scope: .buffers)
            encoder.setComputePipelineState(merge)
            encoder.setBuffer(partials, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: lanes)
        }
    }

    package func read(rows: Int) -> (tokens: [Int32], scores: [Float])? {
        precondition((1...16).contains(rows))
        let records = output.contents().assumingMemoryBound(to: UInt32.self)
        var tokens: [Int32] = [], scores: [Float] = []
        tokens.reserveCapacity(rows * Self.count)
        scores.reserveCapacity(rows * Self.count)
        for row in 0..<rows {
            let record = records + row * Self.recordWords
            guard record[0] == 0 && record[1] == Self.count else { return nil }
            for rank in 0..<Self.count {
                tokens.append(Int32(record[2 + rank * 2]))
                scores.append(Float(bitPattern: record[3 + rank * 2]))
            }
        }
        return (tokens, scores)
    }
}
#endif
