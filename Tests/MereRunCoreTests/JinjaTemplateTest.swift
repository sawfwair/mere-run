import XCTest
import Jinja

final class Gemma4TemplateParseTest: XCTestCase {
    func testFullGemma4Template() throws {
        let path = "/Users/nerd/Library/Application Support/MereRun/hub/models/google/gemma-4-31B-it/chat_template.jinja"
        let full = try String(contentsOfFile: path, encoding: .utf8)
        let lines = full.components(separatedBy: "\n")

        for end in 1...lines.count {
            let sub = lines[0..<end].joined(separator: "\n")
            do {
                let _ = try Template(sub)
            } catch {
                print("FAIL at line \(end): |\(lines[end-1])|")
                XCTFail("Parse fails at line \(end): \(lines[end-1]) — \(error)")
                return
            }
        }

        XCTAssertNoThrow(try Template(full))
    }
}
