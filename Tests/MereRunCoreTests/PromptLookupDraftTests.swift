import XCTest
@testable import MereRunCore

/// The n-gram matcher behind Gemma4 prompt-lookup speculation. Pure host
/// function: no models involved.
final class PromptLookupDraftTests: XCTestCase {
    func testDraftsContinuationOfEarlierTrigram() {
        // ... 1 2 3 9 9 ... 1 2 3 -> should draft [9, 9, ...]
        let context = [5, 1, 2, 3, 9, 9, 8, 7, 6, 1, 2, 3]
        let draft = Gemma4Generator.promptLookupDraft(context: context, blockSize: 4)
        XCTAssertEqual(draft, [9, 9, 8, 7])
    }

    func testPrefersMostRecentOccurrence() {
        // 1 2 3 appears twice earlier; the LATER one (followed by 42) wins.
        let context = [1, 2, 3, 7, 0, 1, 2, 3, 42, 5, 1, 2, 3]
        let draft = Gemma4Generator.promptLookupDraft(context: context, blockSize: 2)
        XCTAssertEqual(draft, [42, 5])
    }

    func testFallsBackToBigramWhenNoTrigramMatch() {
        // trailing trigram [9, 2, 3] never occurred; bigram [2, 3] did.
        let context = [1, 2, 3, 8, 8, 8, 9, 2, 3]
        let draft = Gemma4Generator.promptLookupDraft(context: context, blockSize: 3)
        XCTAssertEqual(draft, [8, 8, 8])
    }

    func testClipsAtContextEnd() {
        let context = [1, 2, 3, 4, 1, 2, 3]
        // Match at start; only [4] follows before running into the tail —
        // the draft can extend across the tail region.
        let draft = Gemma4Generator.promptLookupDraft(context: context, blockSize: 8)
        XCTAssertEqual(draft, [4, 1, 2, 3])
    }

    func testNoMatchDraftsNothing() {
        XCTAssertEqual(Gemma4Generator.promptLookupDraft(context: [1, 2, 3, 4, 5, 6, 7], blockSize: 4), [])
    }

    func testDegenerateContextsDraftNothing() {
        XCTAssertEqual(Gemma4Generator.promptLookupDraft(context: [], blockSize: 4), [])
        XCTAssertEqual(Gemma4Generator.promptLookupDraft(context: [1, 2], blockSize: 4), [])
        XCTAssertEqual(Gemma4Generator.promptLookupDraft(context: [1, 2, 3, 1, 2, 3], blockSize: 0), [])
    }
}
