import MLX

package struct ParakeetEncoderOutput {
    package let features: MLXArray
    package let lengths: [Int]

    package init(features: MLXArray, lengths: [Int]) {
        self.features = features
        self.lengths = lengths
    }
}

package protocol ParakeetExternalEncoder: AnyObject {
    func encode(_ mel: MLXArray) throws -> ParakeetEncoderOutput
}

package protocol ParakeetExternalTDTDecoder: AnyObject {
    var maximumBatchSize: Int { get }

    func decode(
        encoded: MLXArray,
        lengths: [Int]
    ) throws -> [ParakeetAlignedResult]
}
