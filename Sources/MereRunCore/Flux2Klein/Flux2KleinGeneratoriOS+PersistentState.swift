import Foundation
import MLX
import MLXRandom
import MLXNN
import ImageIO

extension Flux2KleinGeneratoriOS {

    // MARK: - Persistent State Loading

    /// Load tokenizer and BatchNorm stats - kept in memory (tiny ~2MB)
    func loadPersistentState(from path: String) async throws {
        let modelRootURL = URL(fileURLWithPath: path).standardizedFileURL
        let manifest = try MereRunModelManifest.loadRequired(from: modelRootURL)
        let componentResolver = ModelComponentResolver(modelRootURL: modelRootURL, manifest: manifest)
        let tokenizerComponent = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        let vaeComponent = try componentResolver.resolveDirectory(for: .vae, fallbackLocalPath: "vae")

        // Load tokenizer
        tokenizer = try QwenTokenizer.load(from: tokenizerComponent.directoryURL, maxLengthOverride: 512)

        // Load BN stats from VAE weights (only extract the two small arrays we need)
        let vaeWeightsURL = vaeComponent.directoryURL.appendingPathComponent("diffusion_pytorch_model.safetensors")

        // Load only the BN stats we need
        let weights = try MLX.loadArrays(url: vaeWeightsURL)

        guard let mean = weights["bn.running_mean"],
              let variance = weights["bn.running_var"] else {
            throw Flux2Error.missingBatchNormStats
        }

        // Evaluate to materialize values before weights dict goes out of scope
        bnRunningMean = mean
        bnRunningVar = variance
        MLX.eval(bnRunningMean!, bnRunningVar!)

        loadedModelPath = path
        loadedManifest = manifest
        // Note: Don't clear GPU memory here - BN stats need to stay valid
    }


}
