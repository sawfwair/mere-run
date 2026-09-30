#if os(macOS)
import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest
import UniformTypeIdentifiers
@testable import MereRunCore

final class OrnithVisionPrefixCheckpointTests: MereRunCoreTestCase {
    private struct Measurement: Encodable {
        let label: String
        let cacheEnabled: Bool
        let response: String
        let promptTokens: Int
        let outputTokens: Int
        let prefillSeconds: Double
        let decodeSeconds: Double
        let hits: Int
        let reusedTokens: Int
        let storedTokens: Int
        let peakMLXBytes: Int
    }

    func testInstalledImagePrefixReuseMatchesUncachedImageAnswers() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_TEST_ORNITH_PREFIX_MODEL_ROOT"],
              let output = environment["MERERUN_TEST_ORNITH_PREFIX_OUTPUT"] else {
            throw XCTSkip("Set MERERUN_TEST_ORNITH_PREFIX_MODEL_ROOT and MERERUN_TEST_ORNITH_PREFIX_OUTPUT.")
        }
        let directory = URL(fileURLWithPath: output, isDirectory: true)
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            return XCTFail("Use a fresh output directory")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let first = try Self.fixture(directory: directory, swapped: false)
        let second = try Self.fixture(directory: directory, swapped: true)
        let reference = (0..<1_500).map { "Reference row \($0): alpha beta gamma delta." }.joined(separator: "\n")
        let question = "Name only the colors of the two shapes in the last image, from left to right."
        let system = "The reference rows are irrelevant to the image question. Reply with only two color names.\n" + reference
        let cases: [(String, [ChatMessage], String)] = [
            ("first", [.init(role: .system, content: system), .init(role: .user, content: question, imageUrl: first.path)], "red, blue"),
            ("repeat", [.init(role: .system, content: system), .init(role: .user, content: question, imageUrl: first.path)], "red, blue"),
            ("changed-pixels", [.init(role: .system, content: system), .init(role: .user, content: question, imageUrl: second.path)], "blue, red"),
            ("original-again", [.init(role: .system, content: system), .init(role: .user, content: question, imageUrl: first.path)], "red, blue"),
            ("changed-text", [.init(role: .system, content: "Updated reference.\n" + system),
                              .init(role: .user, content: question, imageUrl: first.path)], "red, blue"),
            ("two-images", [.init(role: .system, content: system),
                            .init(role: .user, content: "Earlier image.", imageUrl: second.path),
                            .init(role: .assistant, content: "I will use the last image for the next question."),
                            .init(role: .user, content: question, imageUrl: first.path)], "red, blue"),
            ("retained-last-image", [.init(role: .system, content: system),
                                     .init(role: .user, content: question, imageUrl: first.path)], "red, blue"),
        ]
        var baseline: [String] = []
        var measurements: [Measurement] = []
        for enabled in [false, true] {
            let generator = Q35Generator(
                modelId: Q35Resources.ornith35BMLX8BitModelId,
                prefixKVCacheEnabled: enabled, continuousBatchingEnabled: false
            )
            do {
                for (index, entry) in cases.enumerated() {
                    let before = await generator.prefixKVCacheStats()
                    let response = try await generator.chat(
                        ChatRequest(messages: entry.1, maxTokens: 32, temperature: 0, topP: 1,
                                    showThinking: false, maxContextTokens: 32_768),
                        modelPath: root,
                        progressHandler: { progress in
                            if progress.message?.contains("Reusing") == true {
                                FileHandle.standardError.write(Data("[vision-prefix] \(progress.message ?? "")\n".utf8))
                            }
                        }
                    )
                    let after = await generator.prefixKVCacheStats()
                    let normalized = response.response.lowercased().split(whereSeparator: { !$0.isLetter }).joined(separator: ", ")
                    XCTAssertEqual(normalized, entry.2, "\(entry.0), cache=\(enabled)")
                    if enabled {
                        XCTAssertEqual(response.response, baseline[index], "Cached and uncached image output differs")
                        if ["repeat", "changed-pixels", "original-again", "retained-last-image"].contains(entry.0) {
                            XCTAssertGreaterThan(after.reusedTokens - before.reusedTokens, 10_000)
                        }
                    } else {
                        baseline.append(response.response)
                    }
                    let measurement = Measurement(
                        label: entry.0, cacheEnabled: enabled, response: response.response,
                        promptTokens: response.promptTokens ?? 0, outputTokens: response.tokensGenerated,
                        prefillSeconds: response.timing?.prefillSeconds ?? 0,
                        decodeSeconds: response.timing?.decodeSeconds ?? 0,
                        hits: after.hits - before.hits, reusedTokens: after.reusedTokens - before.reusedTokens,
                        storedTokens: after.storedTokens, peakMLXBytes: Memory.peakMemory
                    )
                    measurements.append(measurement)
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    try encoder.encode(measurements).write(to: directory.appendingPathComponent("measurements.json"))
                    print("[vision-prefix] \(entry.0) cache=\(enabled) prefill=\(measurement.prefillSeconds) reused=\(measurement.reusedTokens) answer=\(String(reflecting: response.response))")
                }
            } catch {
                await generator.unload()
                throw error
            }
            await generator.unload()
            Memory.clearCache()
        }
    }

    private static func fixture(directory: URL, swapped: Bool) throws -> URL {
        let context = try XCTUnwrap(CGContext(
            data: nil, width: 512, height: 384, bitsPerComponent: 8, bytesPerRow: 512 * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        let red = CGColor(red: 0.9, green: 0.05, blue: 0.05, alpha: 1)
        let blue = CGColor(red: 0.05, green: 0.15, blue: 0.9, alpha: 1)
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 512, height: 384))
        context.setFillColor(swapped ? blue : red)
        context.fillEllipse(in: CGRect(x: 55, y: 117, width: 150, height: 150))
        context.setFillColor(swapped ? red : blue)
        context.fill(CGRect(x: 307, y: 117, width: 150, height: 150))
        let url = directory.appendingPathComponent(swapped ? "blue-red.png" : "red-blue.png")
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return url
    }
}
#endif
