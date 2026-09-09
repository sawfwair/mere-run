import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public actor LTXDistilledLatentGenerator {
    var textEncoder: LTXGemmaTextEncoder?
    var transformer: LTXDistilledTransformer?
    var decoder: LTXVideoDecoder?
    var encoder: LTXVideoEncoder?
    var upsampler: LTXLatentUpsampler?
    var modelWeightsURL: URL?
    var loadedDType: DType = .bfloat16
    var loadedRoot: URL?

    public init() {}

    public func unload() async {
        if let textEncoder {
            await textEncoder.unload()
        }
        textEncoder = nil
        transformer = nil
        decoder = nil
        encoder = nil
        upsampler = nil
        modelWeightsURL = nil
        loadedRoot = nil
        Memory.clearCache()
    }
}
