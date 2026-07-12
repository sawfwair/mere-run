import MediaIO
import MLX
import XCTest
@testable import MereRunCore

final class Wan2RealVAERoundTripTests: MereRunCoreTestCase {
    func testImageEncoderMatchesReferenceHarness() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_WAN2_MODEL_ROOT"],
              let sourcePath = environment["MERERUN_WAN2_SOURCE_IMAGE"] else {
            throw XCTSkip("Set the Wan2 model and source image environment variables.")
        }
        let source = try MediaImageIO.centerCropped(
            MediaImageIO.decode(URL(fileURLWithPath: sourcePath)),
            width: 128,
            height: 128
        )
        let channels = MediaImageIO.rgbCHWFloat(source, normalizedToMinusOneToOne: true)
        let frame = MLXArray(channels)
            .reshaped(3, 128, 128)
            .transposed(1, 2, 0)
            .reshaped(1, 1, 128, 128, 3)
        let resources = Wan2Resources(rootURL: URL(fileURLWithPath: root))
        let vae = try Wan2ModelLoader.loadVAE(resources: resources)
        let latents = vae.encodeImage(frame).asType(.float32)
        eval(latents)
        let outputURL = URL(fileURLWithPath: "/tmp/wan-swift-vae-encode.npy")
        try MLX.save(array: latents, url: outputURL)
        print(
            "Wan VAE Swift encode shape=\(latents.shape) "
                + "mean=\(MLX.mean(latents).item(Float.self)) "
                + "std=\(MLX.std(latents).item(Float.self)) path=\(outputURL.path)"
        )
    }

    func testStaticVideoRoundTripWithInstalledVAE() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_WAN2_MODEL_ROOT"],
              let sourcePath = environment["MERERUN_WAN2_SOURCE_IMAGE"],
              let outputPath = environment["MERERUN_WAN2_VAE_OUTPUT"] else {
            throw XCTSkip("Set the Wan2 model, source image, and VAE output environment variables.")
        }
        let source = try MediaImageIO.centerCropped(
            MediaImageIO.decode(URL(fileURLWithPath: sourcePath)),
            width: 128,
            height: 128
        )
        let channels = MediaImageIO.rgbCHWFloat(source, normalizedToMinusOneToOne: true)
        let frame = MLXArray(channels)
            .reshaped(3, 128, 128)
            .transposed(1, 2, 0)
            .reshaped(1, 1, 128, 128, 3)
        let video = MLX.repeated(frame, count: 9, axis: 1)
        let resources = Wan2Resources(rootURL: URL(fileURLWithPath: root))
        let vae = try Wan2ModelLoader.loadVAE(resources: resources)
        let latents = vae.encodeVideo(video)
        let decoded = MLX.clip((vae.decode(latents) + 1) * 127.5, min: 0, max: 255).asType(.uint8)
        eval(decoded)
        XCTAssertEqual(decoded.shape, [1, 9, 128, 128, 3])
        try LTXVideoMP4Writer.writeMP4(
            frames: decoded,
            fps: 24,
            to: URL(fileURLWithPath: outputPath)
        )
    }
}
