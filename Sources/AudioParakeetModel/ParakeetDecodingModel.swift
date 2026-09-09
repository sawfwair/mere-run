import Foundation
import MLX
import MLXFast
import MLXNN

public protocol ParakeetDecodingModel: AnyObject {
    var config: ParakeetModelConfig { get }
    var preferredWindowBatchSize: Int { get }
    func decode(_ mel: MLXArray) throws -> [ParakeetAlignedResult]
    func decode(
        _ mel: MLXArray,
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult]
    func decodeWindows(
        _ mels: [MLXArray],
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult]
}

public extension ParakeetDecodingModel {
    var preferredWindowBatchSize: Int { 1 }

    func decode(
        _ mel: MLXArray,
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult] {
        let started = ParakeetMonotonicClock.now()
        let result = try decode(mel)
        timings.decoderSeconds += ParakeetMonotonicClock.seconds(since: started)
        return result
    }

    func decodeWindows(
        _ mels: [MLXArray],
        timings: inout ParakeetModelTimings
    ) throws -> [ParakeetAlignedResult] {
        try mels.map { mel in
            try decode(mel, timings: &timings).first
                ?? ParakeetAlignedResult(text: "", sentences: [])
        }
    }
}
