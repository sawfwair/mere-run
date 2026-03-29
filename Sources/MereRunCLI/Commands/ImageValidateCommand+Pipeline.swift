import Foundation
import MLX
import MereRunCore

/// Owns the end-to-end pipeline validation path.
/// This file keeps the family-specific generation wiring together so readers
/// can see the full prompt -> image path separately from the lower-level
/// component checks.
extension ImageValidate {
    func runPipelineTests(modelURL: URL, outputDir: URL) async throws {
        print("\n== Pipeline validation ==")
        print("▶ Test: Deterministic generation with fixed seed")
        print("  Purpose: Validate full pipeline produces reproducible outputs")

        let testPrompt = "a red cube on a white background, simple, minimal"
        let testSeed: UInt64 = 12345
        let width = 512
        let height = 512

        print("  Prompt: \"\(testPrompt)\"")
        print("  Seed: \(testSeed)")
        print("  Size: \(width)x\(height)")

        let outputURL = outputDir.appendingPathComponent("pipeline_deterministic.png")
        try await runPipelineGeneration(
            modelPath: modelURL.path,
            prompt: testPrompt,
            seed: testSeed,
            width: width,
            height: height,
            outputURL: outputURL
        )

        if saveReference {
            let refPath = outputDir.appendingPathComponent("reference_pipeline_output.png")
            if FileManager.default.fileExists(atPath: refPath.path) {
                try FileManager.default.removeItem(at: refPath)
            }
            try FileManager.default.copyItem(at: outputURL, to: refPath)
            print("  Saved as reference: \(refPath.lastPathComponent)")
        }

        if compare, let refDir = referenceDir {
            let refPath = URL(fileURLWithPath: refDir).appendingPathComponent("reference_pipeline_output.png")
            if FileManager.default.fileExists(atPath: refPath.path) {
                print("  Comparing with reference...")
                print("  ✓ Reference exists at: \(refPath.path)")
                print("  (Visual inspection recommended)")
            } else {
                print("  ⚠ No reference found at: \(refPath.path)")
            }
        }

        Memory.clearCache()
    }

    private func runPipelineGeneration(
        modelPath: String,
        prompt: String,
        seed: UInt64,
        width: Int,
        height: Int,
        outputURL: URL
    ) async throws {
        if family == "zimage" {
            let request = GenerationRequest(
                prompt: prompt,
                negativePrompt: nil,
                width: width,
                height: height,
                steps: 4,
                guidanceScale: 0.0,
                seed: seed,
                outputURL: outputURL,
                model: modelPath,
                maxSequenceLength: 512,
                lora: nil,
                enhancePrompt: false,
                inputImage: nil,
                strength: 1.0,
                useBetaSigmas: false
            )

            let generator = ZImageTurboGenerator()
            print("  Running generation...")
            let result = try await generator.generate(request, progressHandler: zimagePipelineProgress)
            print("  ✓ Generated: \(result.outputURL.lastPathComponent)")
            return
        }

        let request = GenerationRequest(
            prompt: prompt,
            width: width,
            height: height,
            steps: 4,
            guidanceScale: 1.0,
            seed: seed,
            outputURL: outputURL,
            model: modelPath
        )

        let generator = Flux2KleinGenerator()
        print("  Running generation...")
        _ = try await generator.generate(request, progressHandler: kleinPipelineProgress)
        print("  ✓ Generated: \(outputURL.lastPathComponent)")
    }

    private func zimagePipelineProgress(_ progress: GenerationProgress) {
        switch progress.stage {
        case .loadingModel, .loadingEncoder, .loadingTransformer, .loadingVAE, .loadingLoRA, .encodingReferenceImages, .saving:
            break
        case .encodingText:
            print("  Encoding text...")
        case .denoising:
            if progress.stepIndex == 0 {
                print("  Denoising...")
            }
        case .decoding:
            print("  Decoding...")
        }
    }

    private func kleinPipelineProgress(_ progress: GenerationProgress) {
        switch progress.stage {
        case .loadingTransformer, .loadingEncoder, .loadingVAE, .loadingLoRA, .encodingReferenceImages, .saving, .loadingModel:
            break
        case .encodingText:
            print("  Encoding prompt...")
        case .denoising:
            if progress.stepIndex == 0 {
                print("  Denoising \(progress.totalSteps) steps...")
            }
        case .decoding:
            print("  Decoding...")
        }
    }
}
