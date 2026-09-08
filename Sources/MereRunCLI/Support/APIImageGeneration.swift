import Foundation
import MereRunCore

extension APIServerContract.ImageGenerationPlan {
    /// Preserve the v1 API's four-step Klein default and input conditioning.
    /// Equivalent explicit text-to-image settings otherwise use the CLI's plan.
    func operationPlan(
        modelRoot: URL,
        outputURL: URL,
        manifest: MereRunModelManifest,
        qwenEditDefaults: Bool = false
    ) throws -> MereRunCore.ImageGenerationPlan {
        var effectiveManifest = manifest
        if qwenEditDefaults {
            // A recognized managed Qwen installation supplied the backend before
            // manifests carried family/engine fields. Preserve that legacy input.
            effectiveManifest.family = manifest.family ?? .qwen
            effectiveManifest.engine = manifest.engine ?? .qwenImageEdit
        }
        let options = ImageGenerationOptions(
            prompt: prompt, negativePrompt: negativePrompt, outputURL: outputURL,
            width: width, height: height, steps: steps, guidanceScale: guidanceScale,
            seed: seed, inputImage: inputImage, referenceImages: additionalInputImages, strength: strength
        )
        // The v1 endpoint accepts mask parts for compatibility but conditions on
        // the whole image, as documented. Strict masking remains a CLI feature.
        return try MereRunCore.ImageGenerationPlan.resolve(
            options, modelRoot: modelRoot, manifest: effectiveManifest,
            policy: ImageGenerationPolicy(
                kleinUsesManifestDefaults: false, kleinInputAsReference: false,
                fallbackSteps: qwenEditDefaults ? 20 : 4,
                fallbackGuidanceScale: qwenEditDefaults ? 4 : 1
            )
        )
    }
}
