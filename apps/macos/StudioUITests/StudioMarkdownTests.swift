import XCTest
@testable import StudioKit
@testable import StudioUI

final class StudioMarkdownTests: XCTestCase {
    func testParagraphsSplitOnBlankLines() {
        let blocks = StudioMarkdownParser.parse("First paragraph.\n\nSecond paragraph.")
        XCTAssertEqual(blocks, [
            .paragraph("First paragraph."),
            .paragraph("Second paragraph.")
        ])
    }

    func testAdjacentLinesStayInOneParagraph() {
        let blocks = StudioMarkdownParser.parse("line one\nline two")
        XCTAssertEqual(blocks, [.paragraph("line one\nline two")])
    }

    func testHeadingLevels() {
        let blocks = StudioMarkdownParser.parse("# Title\n\n### Sub")
        XCTAssertEqual(blocks, [
            .heading(level: 1, text: "Title"),
            .heading(level: 3, text: "Sub")
        ])
    }

    func testHashWithoutSpaceIsNotAHeading() {
        let blocks = StudioMarkdownParser.parse("#hashtag")
        XCTAssertEqual(blocks, [.paragraph("#hashtag")])
    }

    func testFencedCodeWithLanguage() {
        let blocks = StudioMarkdownParser.parse("""
        Before.

        ```swift
        let x = 1
        ```

        After.
        """)
        XCTAssertEqual(blocks, [
            .paragraph("Before."),
            .code(language: "swift", content: "let x = 1"),
            .paragraph("After.")
        ])
    }

    func testUnclosedFenceSwallowsRemainderForStreaming() {
        let blocks = StudioMarkdownParser.parse("Intro.\n\n```python\nprint(1)\nprint(2)")
        XCTAssertEqual(blocks, [
            .paragraph("Intro."),
            .code(language: "python", content: "print(1)\nprint(2)")
        ])
    }

    func testFenceWithoutLanguageHasNilLanguage() {
        let blocks = StudioMarkdownParser.parse("```\nplain\n```")
        XCTAssertEqual(blocks, [.code(language: nil, content: "plain")])
    }

    func testBulletsMergeIntoOneBlock() {
        let blocks = StudioMarkdownParser.parse("- one\n- two\n* three")
        XCTAssertEqual(blocks, [.bullets(["one", "two", "three"])])
    }

    func testNumberedList() {
        let blocks = StudioMarkdownParser.parse("1. first\n2) second")
        XCTAssertEqual(blocks, [.numbered(["first", "second"])])
    }

    func testIndentedContinuationJoinsPreviousBulletItem() {
        let blocks = StudioMarkdownParser.parse("- a long item\n  that continues")
        XCTAssertEqual(blocks, [.bullets(["a long item that continues"])])
    }

    func testQuoteLinesMerge() {
        let blocks = StudioMarkdownParser.parse("> quoted\n> more")
        XCTAssertEqual(blocks, [.quote("quoted\nmore")])
    }

    func testHorizontalRule() {
        let blocks = StudioMarkdownParser.parse("above\n\n---\n\nbelow")
        XCTAssertEqual(blocks, [
            .paragraph("above"),
            .rule,
            .paragraph("below")
        ])
    }

    func testListFollowedByParagraphFlushesCleanly() {
        let blocks = StudioMarkdownParser.parse("- item\nplain text after")
        XCTAssertEqual(blocks, [
            .bullets(["item"]),
            .paragraph("plain text after")
        ])
    }

    func testCodeFenceInterruptsParagraphWithoutBlankLine() {
        let blocks = StudioMarkdownParser.parse("text\n```\ncode\n```")
        XCTAssertEqual(blocks, [
            .paragraph("text"),
            .code(language: nil, content: "code")
        ])
    }

    func testInlineFallsBackToPlainText() {
        // Inline parsing never throws for plain input; the result preserves the characters.
        let attributed = StudioMarkdownParser.inline("plain **bold** text")
        XCTAssertTrue(String(attributed.characters).contains("bold"))
    }

    func testEmptyInputParsesToNoBlocks() {
        XCTAssertEqual(StudioMarkdownParser.parse(""), [])
    }
}
