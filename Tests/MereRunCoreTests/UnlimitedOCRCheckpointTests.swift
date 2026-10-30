#if canImport(Metal)
import Foundation
import Metal
import MLX
import XCTest
@testable import MereRunCore

/// Released assets are opt-in. Every request preserves its raw text and measured Metal memory.
final class UnlimitedOCRCheckpointTests: MereRunCoreTestCase {
    func testFullCheckpointDocumentsOnMetal() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_UNLIMITED_OCR"] == "1", "Full-checkpoint OCR qualification is opt-in.")
        try XCTSkipUnless(environment["MERERUN_TEST_MLX_DEVICE"] == "gpu", "Qualification requires Metal.")
        XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
        let root = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_UNLIMITED_OCR_ROOT"]))
        let input = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_UNLIMITED_OCR_INPUTS"]))
        let output = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_UNLIMITED_OCR_OUTPUT"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: input.appendingPathComponent("manifest.json")))
        let selected = environment["MERERUN_TEST_UNLIMITED_OCR_CASES"]?.split(separator: ",").map(String.init)
        let cases = manifest.cases.filter { selected?.contains($0.id) ?? true }
        XCTAssertFalse(cases.isEmpty)
        let generator = UnlimitedOCRGenerator()
        for item in cases {
            Memory.clearCache()
            Memory.peakMemory = 0
            let start = Date()
            let result = try await generator.ocr(
                imageURLs: item.pages.map { input.appendingPathComponent($0.image) }, modelPath: root.path,
                maxTokens: item.maxTokens, temperature: 0, crop: item.crop == "auto" ? nil : item.crop == "tiles",
                noRepeatNgramSize: 35, ngramWindow: item.pages.count > 1 ? 1024 : 128,
                progress: { message in
                    FileHandle.standardError.write(Data("[unlimited-ocr] \(item.id): \(message)\n".utf8))
                }
            )
            let segments = result.text.components(separatedBy: "<PAGE>")
            let pages = Array(segments.dropFirst())
            var missing: [[String]] = []
            var unexpected: [[String]] = []
            for (index, page) in item.pages.enumerated() {
                let text = index < pages.count ? Self.normalized(pages[index]) : ""
                missing.append(page.anchors.filter { !text.contains(Self.normalized($0)) })
                unexpected.append((page.forbidden ?? []).filter { text.contains(Self.normalized($0)) })
            }
            let record = Record(
                caseID: item.id, checkpointRevision: UnlimitedOCRResources.revision,
                device: MTLCreateSystemDefaultDevice()?.name ?? "unknown", mlxDevice: "gpu",
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                seconds: Date().timeIntervalSince(start), tokensGenerated: result.tokensGenerated,
                pagesGenerated: result.pagesGenerated, expectedPages: item.pages.count,
                reachedTokenLimit: result.reachedTokenLimit, activeMemoryBytes: Memory.activeMemory,
                peakMemoryBytes: Memory.peakMemory, loadSeconds: result.loadSeconds,
                visionSeconds: result.visionSeconds, prefillSeconds: result.prefillSeconds,
                decodeSeconds: result.decodeSeconds, missingAnchorsByPage: missing,
                unexpectedAnchorsByPage: unexpected, noRepeatNgramSize: 35,
                ngramWindow: item.pages.count > 1 ? 1024 : 128, text: result.text
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(record).write(to: output.appendingPathComponent(item.id + ".json"))
            try (result.text + "\n").write(to: output.appendingPathComponent(item.id + ".txt"), atomically: true, encoding: .utf8)
            FileHandle.standardError.write(Data("[unlimited-ocr] \(item.id): \(record.seconds)s, \(result.tokensGenerated) tokens, pages=\(result.pagesGenerated), missing=\(missing), peak=\(record.peakMemoryBytes)\n".utf8))
            XCTAssertFalse(result.reachedTokenLimit, "\(item.id): output hit its token cap")
            XCTAssertEqual(result.pagesGenerated, item.pages.count, "\(item.id): page count mismatch")
            XCTAssertEqual(pages.count, item.pages.count)
            XCTAssertTrue(Self.normalized(segments[0]).isEmpty, "\(item.id): content precedes its first page marker")
            XCTAssertTrue(missing.allSatisfy(\.isEmpty), "\(item.id): missing page anchors \(missing)")
            XCTAssertTrue(unexpected.allSatisfy(\.isEmpty), "\(item.id): content leaked across pages \(unexpected)")
        }
    }

    func testReceiptMatchesPublicCLIOnMetal() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MERERUN_TEST_UNLIMITED_OCR"] == "1" && env["MERERUN_TEST_MLX_DEVICE"] == "gpu",
                          "Full-checkpoint OCR qualification on Metal is opt-in.")
        guard let executable = env["MERERUN_TEST_UNLIMITED_OCR_CLI"] else {
            throw XCTSkip("Set MERERUN_TEST_UNLIMITED_OCR_CLI to compare the public executable.")
        }
        let root = try XCTUnwrap(env["MERERUN_TEST_UNLIMITED_OCR_ROOT"])
        let inputs = URL(fileURLWithPath: try XCTUnwrap(env["MERERUN_TEST_UNLIMITED_OCR_INPUTS"]))
        let output = URL(fileURLWithPath: try XCTUnwrap(env["MERERUN_TEST_UNLIMITED_OCR_OUTPUT"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let image = inputs.appendingPathComponent("receipt.jpeg")
        let result = try await UnlimitedOCRGenerator().ocr(
            imageURLs: [image], modelPath: root, maxTokens: 1024, temperature: 0,
            crop: false, noRepeatNgramSize: 35, ngramWindow: 128
        )
        XCTAssertFalse(result.reachedTokenLimit)
        XCTAssertEqual(result.pagesGenerated, 1)
        let nativeText = result.text + "\n"
        try nativeText.write(to: output.appendingPathComponent("cross-process.native.txt"), atomically: true, encoding: .utf8)
        let stdoutURL = output.appendingPathComponent("cross-process.cli.stdout.txt")
        let stderrURL = output.appendingPathComponent("cross-process.cli.stderr.txt")
        try Data().write(to: stdoutURL)
        try Data().write(to: stderrURL)
        let stdout = try FileHandle(forWritingTo: stdoutURL)
        let stderr = try FileHandle(forWritingTo: stderrURL)
        defer { try? stdout.close(); try? stderr.close() }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = ["vision", "ocr", image.path, "--backend", "unlimited", "--unlimited-model", root,
                             "--unlimited-crop", "base", "--temperature", "0", "--max-tokens", "1024",
                             "--unlimited-no-repeat-ngram-size", "35", "--unlimited-ngram-window", "128", "--quiet"]
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let diagnostics = try String(contentsOf: stderrURL, encoding: .utf8)
        XCTAssertEqual(process.terminationStatus, 0, diagnostics)
        XCTAssertEqual(try String(contentsOf: stdoutURL, encoding: .utf8), nativeText)
        let testLibrary = Bundle(for: Self.self).bundleURL.appendingPathComponent("Contents/MacOS/mlx.metallib")
        let cliLibrary = URL(fileURLWithPath: executable).resolvingSymlinksInPath().deletingLastPathComponent().appendingPathComponent("mlx.metallib")
        XCTAssertEqual(try Data(contentsOf: testLibrary), try Data(contentsOf: cliLibrary))
    }

    private static func normalized(_ text: String) -> String {
        let folded = text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
        return String(folded.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) })
    }

    private struct Manifest: Decodable { let cases: [Case] }
    private struct Case: Decodable, Sendable {
        let id: String
        let crop: String
        let maxTokens: Int
        let pages: [Page]
    }
    private struct Page: Decodable, Sendable { let image: String; let anchors: [String]; let forbidden: [String]? }
    private struct Record: Encodable {
        let caseID: String
        let checkpointRevision: String
        let device: String
        let mlxDevice: String
        let physicalMemoryBytes: UInt64
        let seconds: Double
        let tokensGenerated: Int
        let pagesGenerated: Int
        let expectedPages: Int
        let reachedTokenLimit: Bool
        let activeMemoryBytes: Int
        let peakMemoryBytes: Int
        let loadSeconds: Double
        let visionSeconds: Double
        let prefillSeconds: Double
        let decodeSeconds: Double
        let missingAnchorsByPage: [[String]]
        let unexpectedAnchorsByPage: [[String]]
        let noRepeatNgramSize: Int
        let ngramWindow: Int
        let text: String
    }
}
#endif
