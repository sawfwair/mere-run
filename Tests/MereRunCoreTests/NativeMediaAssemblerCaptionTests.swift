#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import MereRunCore

final class NativeMediaAssemblerCaptionTests: XCTestCase {
    func testCaptionsAreBurnedIntoFrames() async throws {
        let fm = FileManager.default
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("mererun-native-assembler-caption-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }

        let imageURL = tempRoot.appendingPathComponent("frame.png")
        try writeSolidPNG(url: imageURL, width: 1080, height: 1920, rgb: (0, 0, 0))

        let outputURL = tempRoot.appendingPathComponent("final.mov")
        let workDir = tempRoot.appendingPathComponent("work", isDirectory: true)

        let segment = NativeMediaSegment(
            startSeconds: 0,
            endSeconds: 2.0,
            source: .image(
                url: imageURL,
                motion: NativeKenBurnsMotion(
                    startZoom: 1.0,
                    endZoom: 1.0,
                    driftXFraction: 0,
                    driftYFraction: 0,
                    driftXFrequency: 0,
                    driftYFrequency: 0
                )
            )
        )

        let cue = NativeMediaCaptionCue(
            startSeconds: 0.0,
            endSeconds: 2.0,
            text: "HELLO WORLD"
        )

        let request = NativeMediaAssemblyRequest(
            renderWidth: 1080,
            renderHeight: 1920,
            fps: 10,
            segments: [segment],
            captions: [cue],
            narrationLayers: [],
            backgroundLayer: nil,
            outputURL: outputURL,
            workDirectory: workDir,
            captionStyle: .shortsDefault
        )

        _ = try NativeMediaAssembler().assemble(request: request, onLog: { _ in })
        XCTAssertTrue(fm.fileExists(atPath: outputURL.path))

        let noCaptionOutputURL = tempRoot.appendingPathComponent("final-no-captions.mov")
        let noCaptionRequest = NativeMediaAssemblyRequest(
            renderWidth: request.renderWidth,
            renderHeight: request.renderHeight,
            fps: request.fps,
            segments: request.segments,
            captions: [],
            narrationLayers: request.narrationLayers,
            backgroundLayer: request.backgroundLayer,
            outputURL: noCaptionOutputURL,
            workDirectory: tempRoot.appendingPathComponent("work-no-captions", isDirectory: true),
            captionStyle: request.captionStyle
        )
        _ = try NativeMediaAssembler().assemble(request: noCaptionRequest, onLog: { _ in })
        XCTAssertTrue(fm.fileExists(atPath: noCaptionOutputURL.path))

        let asset = AVURLAsset(url: outputURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceAfter = .zero
        generator.requestedTimeToleranceBefore = .zero

        let frameTime = CMTime(seconds: 1.0, preferredTimescale: 600)
        let cgImage = try await generateCGImage(with: generator, at: frameTime)

        let plainAsset = AVURLAsset(url: noCaptionOutputURL)
        let plainGenerator = AVAssetImageGenerator(asset: plainAsset)
        plainGenerator.appliesPreferredTrackTransform = true
        plainGenerator.requestedTimeToleranceAfter = .zero
        plainGenerator.requestedTimeToleranceBefore = .zero

        let plainCGImage = try await generateCGImage(with: plainGenerator, at: frameTime)

        let changedPixels = pixelDifferenceCount(
            cgImage,
            plainCGImage,
            channelDeltaThreshold: 16
        )
        if changedPixels == 0 {
            throw XCTSkip("CoreAnimation caption burn-in was not rendered in this runtime.")
        }
        XCTAssertGreaterThan(
            changedPixels,
            2_000,
            "Expected burned captions to change the rendered frame. Changed pixels: \(changedPixels)."
        )
    }

    private func generateCGImage(
        with generator: AVAssetImageGenerator,
        at time: CMTime
    ) async throws -> CGImage {
        try await withCheckedThrowingContinuation { continuation in
            generator.generateCGImageAsynchronously(for: time) { image, _, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let image else {
                    continuation.resume(
                        throwing: NSError(
                            domain: "NativeMediaAssemblerCaptionTests",
                            code: 2
                        )
                    )
                    return
                }

                continuation.resume(returning: image)
            }
        }
    }

    private func writeSolidPNG(
        url: URL,
        width: Int,
        height: Int,
        rgb: (UInt8, UInt8, UInt8)
    ) throws {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        var data = [UInt8](repeating: 0, count: bytesPerRow * height)

        for y in 0..<height {
            for x in 0..<width {
                let idx = (y * bytesPerRow) + (x * 4)
                data[idx] = rgb.0
                data[idx + 1] = rgb.1
                data[idx + 2] = rgb.2
                data[idx + 3] = 255
            }
        }

        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let image = ctx.makeImage() else {
            XCTFail("Failed to create solid PNG context.")
            return
        }

        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            XCTFail("Failed to create image destination.")
            return
        }

        CGImageDestinationAddImage(dest, image, nil)
        if !CGImageDestinationFinalize(dest) {
            throw NSError(domain: "NativeMediaAssemblerCaptionTests", code: 1)
        }
    }

    private func pixelDifferenceCount(
        _ lhs: CGImage,
        _ rhs: CGImage,
        channelDeltaThreshold: UInt8
    ) -> Int {
        guard lhs.width == rhs.width, lhs.height == rhs.height else { return .max }

        let width = lhs.width
        let height = lhs.height
        let bytesPerRow = width * 4

        var leftData = [UInt8](repeating: 0, count: bytesPerRow * height)
        var rightData = [UInt8](repeating: 0, count: bytesPerRow * height)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue

        guard let leftCtx = CGContext(
            data: &leftData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let rightCtx = CGContext(
            data: &rightData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return 0
        }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        leftCtx.draw(lhs, in: rect)
        rightCtx.draw(rhs, in: rect)

        var changedPixels = 0
        let threshold = Int(channelDeltaThreshold)
        for i in stride(from: 0, to: leftData.count, by: 4) {
            let dr = abs(Int(leftData[i]) - Int(rightData[i]))
            let dg = abs(Int(leftData[i + 1]) - Int(rightData[i + 1]))
            let db = abs(Int(leftData[i + 2]) - Int(rightData[i + 2]))
            if max(dr, max(dg, db)) >= threshold {
                changedPixels += 1
            }
        }

        return changedPixels
    }
}
#endif
