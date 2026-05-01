import Foundation
import MLX
import MLXRandom
import MereRunCore

/// Owns VAE-only validation for the image families.
/// This file keeps the decode and roundtrip checks together so the reader can
/// follow the latent -> image -> latent path without jumping through the rest
/// of the validation suite.
extension ImageValidate {
    func runVAETests(modelURL: URL, outputDir: URL) async throws {
        print("\n== VAE validation (\(family.uppercased())) ==")

        let vaeWeightsPath = modelURL.appendingPathComponent("vae/diffusion_pytorch_model.safetensors")
        let (vaeConfig, latentChannels) = try loadVAEConfig(from: modelURL)
        let vae = AutoencoderKL(configuration: vaeConfig)

        try loadVAEWeights(
            into: vae,
            from: vaeWeightsPath,
            family: family.lowercased(),
            config: vaeConfig
        )
        print("  VAE loaded")

        try runVAEDecodeTest(vae: vae, latentChannels: latentChannels, outputDir: outputDir)
        try runVAERoundtripTest(vae: vae, outputDir: outputDir)

        Memory.clearCache()
    }

    private func loadVAEConfig(from modelURL: URL) throws -> (VAEConfig, Int) {
        if family.lowercased() == "klein" {
            let configURL = modelURL.appendingPathComponent("vae/config.json")
            let data = try Data(contentsOf: configURL)
            let flux2Config = try JSONDecoder().decode(Flux2VAEConfig.self, from: data)
            let vaeConfig = VAEConfig(
                inChannels: flux2Config.inChannels,
                outChannels: flux2Config.outChannels,
                latentChannels: flux2Config.latentChannels,
                scalingFactor: flux2Config.scalingFactor ?? 1.0,
                shiftFactor: flux2Config.shiftFactor ?? 0.0,
                blockOutChannels: flux2Config.blockOutChannels,
                layersPerBlock: flux2Config.layersPerBlock,
                normNumGroups: flux2Config.normNumGroups,
                sampleSize: flux2Config.sampleSize ?? 1024,
                midBlockAddAttention: flux2Config.midBlockAddAttention,
                useQuantConv: flux2Config.useQuantConv ?? false,
                usePostQuantConv: flux2Config.usePostQuantConv ?? false
            )
            print("  Config: Klein VAE (latentChannels=\(flux2Config.latentChannels), useQuantConv=\(vaeConfig.useQuantConv))")
            return (vaeConfig, flux2Config.latentChannels)
        }

        let vaeConfig = VAEConfig()
        print("  Config: Z-Image Turbo VAE (latentChannels=\(vaeConfig.latentChannels))")
        return (vaeConfig, vaeConfig.latentChannels)
    }

    private func loadVAEWeights(
        into vae: AutoencoderKL,
        from weightsURL: URL,
        family: String,
        config: VAEConfig
    ) throws {
        if family == "klein" {
            let hasQuantConv = config.useQuantConv
            let hasPostQuantConv = config.usePostQuantConv
            try HFSafetensorsWeightsLoader.applyWeights(
                url: weightsURL,
                to: vae,
                dtype: .bfloat16,
                verify: [],
                mapper: { key, value in
                    if key.hasPrefix("bn.") {
                        return []
                    }
                    if !hasQuantConv && key.hasPrefix("quant_conv") {
                        return []
                    }
                    if !hasPostQuantConv && key.hasPrefix("post_quant_conv") {
                        return []
                    }
                    if value.ndim == 4 && key.contains("conv") {
                        return [(key, HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value))]
                    }
                    return [(key, value)]
                }
            )
            return
        }

        try HFSafetensorsWeightsLoader.applyWeights(
            url: weightsURL,
            to: vae,
            dtype: .bfloat16,
            verify: []
        )
    }

    private func runVAEDecodeTest(
        vae: AutoencoderKL,
        latentChannels: Int,
        outputDir: URL
    ) throws {
        print("\n▶ Test 1: VAE decode with random latents")
        print("  Purpose: Validate VAE produces plausible images from random noise")

        MLXRandom.seed(42)

        let latents = MLXRandom.normal([1, latentChannels, 96, 96]).asType(.bfloat16)
        MLX.eval(latents)
        print("  Latents shape: \(latents.shape)")

        if saveReference {
            let latentsPath = outputDir.appendingPathComponent("reference_latents.safetensors")
            try MLX.save(arrays: ["latents": latents], url: latentsPath)
            print("  Saved reference latents to: \(latentsPath.lastPathComponent)")
        }

        print("  Decoding latents...")
        let (decodedNCHW, _) = vae.decode(latents)
        MLX.eval(decodedNCHW)
        print("  Decoded shape: \(decodedNCHW.shape) (NCHW)")

        let decoded = decodedNCHW.transposed(0, 2, 3, 1)
        let imagePath = outputDir.appendingPathComponent("vae_decode_random.png")
        try saveImage(decoded, to: imagePath)
        print("  ✓ Saved decoded image: \(imagePath.lastPathComponent)")
    }

    private func runVAERoundtripTest(vae: AutoencoderKL, outputDir: URL) throws {
        print("\n▶ Test 2: VAE encode-decode roundtrip")
        print("  Purpose: Validate reconstruction quality")

        let testImage = buildVAETestImage(size: 768)
        print("  Test image shape: \(testImage.shape) (NCHW)")

        let originalPath = outputDir.appendingPathComponent("vae_roundtrip_original.png")
        let testImageNHWC = testImage.transposed(0, 2, 3, 1)
        try saveImage(testImageNHWC, to: originalPath)
        print("  Saved original: \(originalPath.lastPathComponent)")

        print("  Encoding to latents...")
        let encoded = vae.encodeToLatents(testImage)
        MLX.eval(encoded)
        print("  Encoded latents shape: \(encoded.shape)")

        if saveReference {
            let encodedPath = outputDir.appendingPathComponent("reference_encoded_latents.safetensors")
            try MLX.save(arrays: ["latents": encoded], url: encodedPath)
            print("  Saved encoded latents: \(encodedPath.lastPathComponent)")
        }

        print("  Decoding latents...")
        let (reconstructedNCHW, _) = vae.decode(encoded)
        MLX.eval(reconstructedNCHW)
        print("  Reconstructed shape: \(reconstructedNCHW.shape) (NCHW)")

        let reconstructed = reconstructedNCHW.transposed(0, 2, 3, 1)
        let reconstructedPath = outputDir.appendingPathComponent("vae_roundtrip_reconstructed.png")
        try saveImage(reconstructed, to: reconstructedPath)
        print("  ✓ Saved reconstructed: \(reconstructedPath.lastPathComponent)")

        let mse = MLX.mean(MLX.pow(testImageNHWC - reconstructed, 2)).item(Float.self)
        print("  Reconstruction MSE: \(String(format: "%.6f", mse))")
        if mse < 0.01 {
            print("  ✓ Reconstruction quality: GOOD")
        } else if mse < 0.05 {
            print("  ⚠ Reconstruction quality: ACCEPTABLE")
        } else {
            print("  ✗ Reconstruction quality: POOR - check VAE implementation")
        }
    }

    private func buildVAETestImage(size: Int) -> MLXArray {
        var testPixels = [Float](repeating: 0, count: 3 * size * size)
        for c in 0..<3 {
            for y in 0..<size {
                for x in 0..<size {
                    let idx = c * size * size + y * size + x
                    switch c {
                    case 0:
                        testPixels[idx] = Float(x) / Float(size)
                    case 1:
                        testPixels[idx] = Float(y) / Float(size)
                    case 2:
                        testPixels[idx] = 0.5
                    default:
                        break
                    }
                }
            }
        }
        return MLXArray(testPixels)
            .reshaped([1, 3, size, size])
            .asType(.bfloat16)
    }
}
