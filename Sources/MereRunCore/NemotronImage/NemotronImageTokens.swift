import Foundation

/// Unitron's virtual image IDs are outside the text tokenizer's vocabulary.
struct NemotronImageTokens: Decodable {
    struct Ranges: Decodable {
        struct Framing: Decodable {
            let start, end, row: Int
            enum CodingKeys: String, CodingKey {
                case start = "<|img_start|>", end = "<|img_end|>", row = "<|vis_row|>"
            }
        }
        struct Vision: Decodable {
            let start, end: Int
            enum CodingKeys: String, CodingKey { case start = "start_id", end = "end_id" }
        }
        let framing: Framing
        let vision: Vision
    }
    let ranges: Ranges
    enum CodingKeys: String, CodingKey { case ranges = "unitron_token_ranges" }

    static let budget = 4_352

    static func supports(width: Int, height: Int) -> Bool {
        guard width > 0, height > 0, width.isMultiple(of: 16), height.isMultiple(of: 16) else { return false }
        // Room for the row markers, the size header, and the opening and closing tokens.
        return height / 16 <= (budget - 12) / (width / 16 + 1)
    }

    func validate() throws {
        guard ranges.framing.start == 131_072, ranges.framing.end == 131_073, ranges.framing.row == 131_074,
              ranges.vision.start == 131_075, ranges.vision.end == 327_635 else {
            throw NemotronImageResources.invalid("Unsupported Unitron framing tokens or visual vocabulary.")
        }
    }

    func codes(from tokens: [Int], width: Int, height: Int, decodeHeader: ([Int]) -> String) throws -> [Int32] {
        let framing = ranges.framing
        func malformed(_ detail: String) -> ImageGenerationIssue { .init("nemotron_image_block_invalid", detail) }
        guard let start = tokens.firstIndex(of: framing.start),
              let end = tokens[(start + 1)...].firstIndex(of: framing.end),
              let row = tokens[(start + 1)..<end].firstIndex(of: framing.row) else {
            throw malformed("Generation ended without a complete image block.")
        }
        let prefix = tokens[(start + 1)..<row]
        guard prefix.allSatisfy({ (0..<framing.start).contains($0) }) else {
            throw malformed("Unexpected visual token in the size header.")
        }
        let header = decodeHeader(Array(prefix))
        let pattern = #"\bheight=([0-9]+)\s+width=([0-9]+)\b"#
        let expression = try NSRegularExpression(pattern: pattern)
        guard let match = expression.firstMatch(in: header, range: NSRange(header.startIndex..., in: header)),
              let h = Range(match.range(at: 1), in: header), let w = Range(match.range(at: 2), in: header),
              Int(header[h]) == height, Int(header[w]) == width else {
            throw malformed("Generated size header does not match \(width)x\(height).")
        }
        let columns = width / 16
        guard Self.supports(width: width, height: height), end - row == height / 16 * (columns + 1) else {
            throw malformed("Generated image grid has the wrong dimensions.")
        }
        var codes: [Int32] = []
        codes.reserveCapacity(height / 16 * columns)
        for (index, token) in tokens[row..<end].enumerated() {
            if index.isMultiple(of: columns + 1) {
                guard token == framing.row else { throw malformed("Missing visual row marker.") }
            } else {
                guard (ranges.vision.start..<ranges.vision.end).contains(token) else {
                    throw malformed("Expected a visual code at grid offset \(index).")
                }
                codes.append(Int32(token - ranges.vision.start))
            }
        }
        return codes
    }
}
