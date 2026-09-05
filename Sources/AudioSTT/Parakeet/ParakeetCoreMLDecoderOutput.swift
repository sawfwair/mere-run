#if canImport(CoreML)
import CoreML

/// Reads the returned tensors even when Core ML declines the proposed output backings.
struct ParakeetCoreMLDecoderOutput {
    private let tokens: MLMultiArray
    private let durations: MLMultiArray
    let hidden: MLMultiArray
    let cell: MLMultiArray

    init(prediction: any MLFeatureProvider, manifest: ParakeetCoreMLManifest.CoreMLDecoder) throws {
        let decisions = [manifest.lanes, manifest.windowFrames]
        let state = [manifest.layers, manifest.lanes, manifest.hiddenSize]
        let tokenShape = decisions + (manifest.usesANESelection ? [2] : [])
        let decisionType: MLMultiArrayDataType = manifest.usesANESelection ? .float16 : .int32
        tokens = try Self.array(manifest.tokenOutputName, in: prediction, shape: tokenShape, type: decisionType)
        durations = try Self.array(manifest.durationOutputName, in: prediction, shape: decisions, type: decisionType)
        hidden = try Self.array(manifest.hiddenOutputName, in: prediction, shape: state, type: .float16)
        cell = try Self.array(manifest.cellOutputName, in: prediction, shape: state, type: .float16)
    }

    func token(lane: Int, frame: Int) -> Int {
        Self.decision(tokens, lane: lane, frame: frame)
    }

    func duration(lane: Int, frame: Int) -> Int {
        Self.decision(durations, lane: lane, frame: frame)
    }

    static func copyStateLane(_ lane: Int, from source: MLMultiArray, to destination: MLMultiArray) {
        let sourcePointer = source.dataPointer.assumingMemoryBound(to: UInt16.self)
        let destinationPointer = destination.dataPointer.assumingMemoryBound(to: UInt16.self)
        let sourceStrides = source.strides.map(\.intValue)
        let destinationStrides = destination.strides.map(\.intValue)
        for layer in 0..<source.shape[0].intValue {
            let sourceOffset = layer * sourceStrides[0] + lane * sourceStrides[1]
            let destinationOffset = layer * destinationStrides[0] + lane * destinationStrides[1]
            if sourceStrides[2] == 1, destinationStrides[2] == 1 {
                destinationPointer.advanced(by: destinationOffset).update(
                    from: sourcePointer.advanced(by: sourceOffset), count: source.shape[2].intValue
                )
            } else {
                for feature in 0..<source.shape[2].intValue {
                    destinationPointer[destinationOffset + feature * destinationStrides[2]] =
                        sourcePointer[sourceOffset + feature * sourceStrides[2]]
                }
            }
        }
    }

    private static func decision(_ array: MLMultiArray, lane: Int, frame: Int) -> Int {
        let offset = lane * array.strides[0].intValue + frame * array.strides[1].intValue
        if array.dataType == .float16 {
            let pointer = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            let first = Int(Float16(bitPattern: pointer[offset]))
            if array.shape.count == 3 {
                let second = Int(Float16(bitPattern: pointer[offset + array.strides[2].intValue]))
                return first * 128 + second
            }
            return first
        }
        return Int(array.dataPointer.assumingMemoryBound(to: Int32.self)[offset])
    }

    private static func array(
        _ name: String,
        in prediction: any MLFeatureProvider,
        shape: [Int],
        type: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        guard let array = prediction.featureValue(for: name)?.multiArrayValue else {
            throw ParakeetCoreMLError.missingOutput(name)
        }
        guard array.shape.map(\.intValue) == shape else {
            throw ParakeetCoreMLError.invalidOutputShape(array.shape.map(\.intValue))
        }
        guard array.dataType == type else {
            throw ParakeetCoreMLError.unsupportedMultiArrayType("\(array.dataType)")
        }
        return array
    }
}
#endif
