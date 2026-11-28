import Foundation
import MereRunCore
import MLX
import MLXNN
import MereRunAudioModels

extension MMAudioBigVGAN {
    public static func load(resources: MMAudioModelResources) throws -> MMAudioBigVGAN {
        let model = MMAudioBigVGAN()
        let weightsURL = resources.bigVGANWeightsURL()
        if weightsURL.pathExtension == "safetensors" {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: weightsURL,
                to: model,
                dtype: .float16,
                verify: .none,
                mapper: mapSafetensorsWeights
            )
        } else {
            try MMAudioBigVGANCheckpointLoader.load(url: weightsURL, into: model)
        }
        return model
    }
}
