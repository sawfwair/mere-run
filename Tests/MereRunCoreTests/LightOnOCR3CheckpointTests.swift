import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// Opt-in released-checkpoint qualification. Inputs and weights are external;
/// records preserve the raw output so content and layout can be reviewed.
final class LightOnOCR3CheckpointTests: MereRunCoreTestCase {
    func testReleasedCheckpointDocuments() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_LIGHTONOCR3"] == "1", "Released OCR checkpoint qualification is opt-in.")
        try XCTSkipUnless(environment["MERERUN_TEST_MLX_DEVICE"] == "gpu", "Checkpoint qualification requires the Metal GPU device.")
        let root = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_LIGHTONOCR3_ROOT"]))
        let input = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_LIGHTONOCR3_INPUTS"]))
        let output = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_LIGHTONOCR3_OUTPUT"]))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: input.appendingPathComponent("manifest.json")))
        let sizes = (environment["MERERUN_TEST_LIGHTONOCR3_SIZES"] ?? "1B,0.8B,4B").split(separator: ",").map(String.init)
        let caseIDs = environment["MERERUN_TEST_LIGHTONOCR3_CASES"]?.split(separator: ",").map(String.init)
        let generator = LightOnOCRGenerator()
        for size in sizes {
            for item in manifest.cases where caseIDs?.contains(item.id) ?? true {
                for mode in [LightOnOCRMode.plain, .grounding] {
                    Memory.peakMemory = 0
                    let start = Date()
                    let result: LightOnOCRGenerator.Result
                    do {
                        result = try await generator.ocr(
                            imageURL: input.appendingPathComponent(item.image),
                            modelPath: root.appendingPathComponent(size).path,
                            config: .init(maxNewTokens: item.id == "blank" ? 64 : 1024, temperature: 0, mode: mode)
                        )
                    } catch {
                        await generator.unload()
                        throw error
                    }
                    let normalized = Self.normalized(result.text)
                    let missing = item.anchors.filter { !normalized.contains(Self.normalized($0)) }
                    let boxes = try Self.boxes(in: result.text)
                    let invalid = boxes.filter { !($0.x1 >= 0 && $0.y1 >= 0 && $0.x2 <= 1000 && $0.y2 <= 1000
                        && $0.x2 > $0.x1 && $0.y2 > $0.y1) }
                    let record = Record(
                        size: size, caseID: item.id, mode: mode.rawValue,
                        seconds: Date().timeIntervalSince(start), tokensGenerated: result.tokensGenerated,
                        activeMemoryBytes: Memory.activeMemory, peakMemoryBytes: Memory.peakMemory,
                        missingAnchors: missing, boxes: boxes, invalidBoxCount: invalid.count, text: result.text
                    )
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    try encoder.encode(record).write(to: output.appendingPathComponent("\(size)-\(item.id)-\(mode.rawValue).json"))
                    FileHandle.standardError.write(Data("[lightonocr3] \(size) \(item.id) \(mode.rawValue): \(record.seconds)s, \(result.tokensGenerated) tokens, missing=\(missing), boxes=\(boxes.count), invalid=\(invalid.count)\n".utf8))
                    XCTAssertTrue(missing.isEmpty, "\(size) \(item.id) \(mode.rawValue): missing \(missing)")
                    XCTAssertEqual(invalid.count, 0, "\(size) \(item.id): invalid normalized boxes")
                    if item.id == "blank" {
                        XCTAssertTrue(result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                                      "\(size) \(mode.rawValue): blank page produced content")
                    } else if mode == .grounding {
                        XCTAssertFalse(boxes.isEmpty, "\(size) \(item.id): no grounding boxes")
                    }
                    XCTAssertLessThan(result.tokensGenerated, item.id == "blank" ? 64 : 1024, "Output hit its token cap")
                }
            }
            await generator.unload()
        }
    }

    private static func normalized(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    private static func boxes(in text: String) throws -> [Box] {
        let regex = try NSRegularExpression(pattern: #"!\[([^\]]+)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"#)
        let source = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: source.length)).map { match in
            Box(label: source.substring(with: match.range(at: 1)),
                x1: Int(source.substring(with: match.range(at: 2)))!,
                y1: Int(source.substring(with: match.range(at: 3)))!,
                x2: Int(source.substring(with: match.range(at: 4)))!,
                y2: Int(source.substring(with: match.range(at: 5)))!)
        }
    }

    private struct Manifest: Decodable { let cases: [Case] }
    private struct Case: Decodable { let id: String; let image: String; let anchors: [String] }
    private struct Box: Encodable { let label: String; let x1: Int; let y1: Int; let x2: Int; let y2: Int }
    private struct Record: Encodable {
        let size: String
        let caseID: String
        let mode: String
        let seconds: Double
        let tokensGenerated: Int
        let activeMemoryBytes: Int
        let peakMemoryBytes: Int
        let missingAnchors: [String]
        let boxes: [Box]
        let invalidBoxCount: Int
        let text: String
    }
}
