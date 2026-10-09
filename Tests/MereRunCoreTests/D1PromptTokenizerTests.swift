import Foundation
import XCTest
@testable import MereRunCore

final class D1PromptTokenizerTests: XCTestCase {
    struct Group: Decodable {
        let family: String
        let cases: [Case]
        struct Case: Decodable { let request: String; let ids: [Int]; let markers: [Int] }
    }
    func testPinnedNativeTokenizersAgainstOriginalPrompts() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let causalRoot = environment["MERERUN_TEST_D1_TOKENIZER"], let omniRoot = environment["MERERUN_TEST_D1_OMNI_TOKENIZER"] else {
            throw XCTSkip("Set MERERUN_TEST_D1_TOKENIZER and MERERUN_TEST_D1_OMNI_TOKENIZER to pinned tokenizer/config directories.")
        }
        let file = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil)).appending(path: "D1/prompts.json")
        let groups = try JSONDecoder().decode([Group].self, from: Data(contentsOf: file))
        for group in groups {
            let root = URL(fileURLWithPath: group.family == "causal" ? causalRoot : omniRoot)
            let tokenizer = try D1Tokenizer.load(root: root)
            let config = try D1Catalog.configuration(root: root)
            for item in group.cases {
                let request = try D1DecisionRequest.decode(Data(item.request.utf8))
                let question = try XCTUnwrap(request.questions.first)
                let sequence = try config.isOmni ? tokenizer.omni(request, question: question, config: config, mediaTokens: 0)
                    : tokenizer.causal(request, question: question, mediaText: "", maxLength: config.maxLength)
                XCTAssertEqual(sequence.ids, item.ids, "\(group.family): \(question.id)")
                XCTAssertEqual(sequence.markers, item.markers, "\(group.family): \(question.id)")
            }
        }
    }
}
