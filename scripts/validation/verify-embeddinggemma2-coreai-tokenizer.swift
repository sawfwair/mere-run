// Compile against MereRunCoreAIModel; see docs/runtime/embeddinggemma2.md.
import Foundation
import MereRunCoreAIModel

enum VerificationError: Error { case arguments, invalidCase(Int) }
struct Receipt: Encodable {
    let sourceRevision: String
    let cases: Int
    let referenceTokenIDs: Int
    let mismatches: Int
}

guard CommandLine.arguments.count == 3 else { throw VerificationError.arguments }
let tokenizer = try GemmaTokenizer(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
guard let cases = try ExactJSON.parse(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])).array else {
    throw VerificationError.invalidCase(-1)
}
let prefixes = ["d": "title: none | text: ", "q": "task: search result | query: ", "raw": ""]
var mismatches = 0, tokens = 0
for (index, test) in cases.enumerated() {
    guard let text = test["text"]?.string, let kind = test["kind"]?.string,
          let prefix = prefixes[kind], let ids = test["ids"]?.array else { throw VerificationError.invalidCase(index) }
    let expected = try ids.map { value -> Int32 in
        guard let id = value.int, let exact = Int32(exactly: id) else { throw VerificationError.invalidCase(index) }
        return exact
    }
    let actual = tokenizer.encodeWithSpecials(prefix + text)
    tokens += expected.count
    if actual != expected {
        mismatches += 1
        if mismatches <= 3 {
            FileHandle.standardError.write(Data("Tokenizer mismatch at case \(index)\n".utf8))
        }
    }
}
let receipt = Receipt(sourceRevision: "305fc2a96790d9343f38038aeca211dfca56d9a5",
                      cases: cases.count, referenceTokenIDs: tokens, mismatches: mismatches)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
FileHandle.standardOutput.write(try encoder.encode(receipt))
FileHandle.standardOutput.write(Data("\n".utf8))
if mismatches > 0 { exit(1) }
