import Foundation
import MereRunCore

extension InstantMeshResources {
    public static func loadModel(from checkpoint: InstantMeshCheckpoint) throws -> InstantMeshModel {
        let verified = try inspectConvertedDirectory(checkpoint.rootURL)
        guard verified == checkpoint else {
            throw InstantMeshResourceError.checkpointIdentityChanged
        }
        let model = InstantMeshModel()
        try InstantMeshWeights.load(
            model: model,
            safetensorsURL: verified.weightsURL,
            dtype: .float32
        )
        return model
    }
}
