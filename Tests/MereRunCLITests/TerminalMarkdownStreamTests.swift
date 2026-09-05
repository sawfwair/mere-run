import XCTest
@testable import MereRunCLI

final class TerminalMarkdownStreamTests: XCTestCase {
    func testAutomaticPresentationOnlyRendersInteractiveText() {
        XCTAssertEqual(
            TerminalMarkdownPresentation.resolve(
                mode: .auto,
                stdoutIsTTY: true,
                isTextResponse: true,
                environment: ["TERM": "xterm-256color"]
            ),
            TerminalMarkdownPresentation(
                rendersMarkdown: true,
                usesANSIStyles: true,
                usesColor: true
            )
        )
        XCTAssertEqual(
            TerminalMarkdownPresentation.resolve(
                mode: .auto,
                stdoutIsTTY: false,
                isTextResponse: true,
                environment: ["TERM": "xterm-256color"]
            ),
            .raw
        )
        XCTAssertEqual(
            TerminalMarkdownPresentation.resolve(
                mode: .auto,
                stdoutIsTTY: true,
                isTextResponse: false,
                environment: ["TERM": "xterm-256color"]
            ),
            .raw
        )
        XCTAssertEqual(
            TerminalMarkdownPresentation.resolve(
                mode: .auto,
                stdoutIsTTY: true,
                isTextResponse: true,
                environment: ["TERM": "dumb"]
            ),
            .raw
        )
    }

    func testAlwaysRendersStructureWithoutInjectingANSIIntoPipes() {
        let presentation = TerminalMarkdownPresentation.resolve(
            mode: .always,
            stdoutIsTTY: false,
            isTextResponse: true,
            environment: ["TERM": "xterm-256color"]
        )

        XCTAssertTrue(presentation.rendersMarkdown)
        XCTAssertFalse(presentation.usesANSIStyles)
        XCTAssertFalse(presentation.usesColor)
    }

    func testNoColorKeepsTypographyButDisablesColor() {
        let presentation = TerminalMarkdownPresentation.resolve(
            mode: .always,
            stdoutIsTTY: true,
            isTextResponse: true,
            environment: ["TERM": "xterm-256color", "NO_COLOR": "1"]
        )

        XCTAssertTrue(presentation.rendersMarkdown)
        XCTAssertTrue(presentation.usesANSIStyles)
        XCTAssertFalse(presentation.usesColor)
    }

    func testChunkBoundariesDoNotChangeRenderedMarkdown() {
        let output = MarkdownOutputRecorder()
        let renderer = TerminalMarkdownStream(
            presentation: .init(
                rendersMarkdown: true,
                usesANSIStyles: false,
                usesColor: false
            ),
            writer: output.write
        )

        renderer.append("# Hel")
        renderer.append("lo\n\n- o")
        renderer.append("ne\n**bo")
        renderer.append("ld** and `co")
        renderer.append("de`\n```swi")
        renderer.append("ft\nlet x = 1\n``")
        renderer.append("`\n")
        renderer.finish()

        XCTAssertEqual(
            output.value,
            """
            Hello

            • one
            bold and code
            ┌─ swift
            │ let x = 1
            └─

            """
        )
    }

    func testFinishClosesUnterminatedCodeFence() {
        let output = MarkdownOutputRecorder()
        let renderer = TerminalMarkdownStream(
            presentation: .init(
                rendersMarkdown: true,
                usesANSIStyles: false,
                usesColor: false
            ),
            writer: output.write
        )

        renderer.append("```swift\nlet value = 1")
        renderer.finish()

        XCTAssertEqual(output.value, "┌─ swift\n│ let value = 1\n└─\n")
    }

    func testFinishResolvesPendingBlockMarkers() {
        let ruleOutput = MarkdownOutputRecorder()
        let ruleRenderer = makePlainRenderer(writer: ruleOutput.write)
        ruleRenderer.append("---")
        ruleRenderer.finish()

        let fenceOutput = MarkdownOutputRecorder()
        let fenceRenderer = makePlainRenderer(writer: fenceOutput.write)
        fenceRenderer.append("```")
        fenceRenderer.finish()

        XCTAssertEqual(ruleOutput.value, String(repeating: "─", count: 32) + "\n")
        XCTAssertEqual(fenceOutput.value, "┌─\n└─\n")
    }

    func testClosingFenceNeedsNoTrailingNewlineAndMayBeIndented() {
        for closingFence in ["```", "   ```"] {
            let output = MarkdownOutputRecorder()
            let renderer = makePlainRenderer(writer: output.write)

            renderer.append("```swift\nlet ready = true\n\(closingFence)")
            renderer.finish()

            XCTAssertEqual(output.value, "┌─ swift\n│ let ready = true\n└─\n")
        }
    }

    func testInlineContextRestartsAfterClosingFence() {
        let output = MarkdownOutputRecorder()
        let renderer = makePlainRenderer(writer: output.write)

        renderer.append("```\nvalue\n```\n*after*")
        renderer.finish()

        XCTAssertEqual(output.value, "┌─\n│ value\n└─\nafter\n")
    }

    func testBackslashesOnlyEscapeMarkdownPunctuation() {
        let output = MarkdownOutputRecorder()
        let renderer = makePlainRenderer(writer: output.write)

        renderer.append(#"C:\Users and \*literal star*"#)
        renderer.finish()

        XCTAssertEqual(output.value, "C:\\Users and *literal star*\n")
    }

    func testLiteralOperatorsStayVisibleWhileDelimitedEmphasisRenders() {
        let output = MarkdownOutputRecorder()
        let renderer = makePlainRenderer(writer: output.write)

        renderer.append("2 * 3\nfile**name\n*italic* and ~~done~~")
        renderer.finish()

        XCTAssertEqual(output.value, "2 * 3\nfile**name\nitalic and done\n")
    }

    func testEachTokenChunkIsBatchedIntoOneTerminalWrite() {
        let output = MarkdownOutputRecorder()
        let renderer = makePlainRenderer(writer: output.write)

        renderer.append("ordinary text")
        XCTAssertEqual(output.writeCount, 1)

        renderer.finish()
        XCTAssertEqual(output.writeCount, 2)
    }

    func testRenderedModeNeutralizesTerminalControlCharacters() {
        let output = MarkdownOutputRecorder()
        let renderer = TerminalMarkdownStream(
            presentation: .init(
                rendersMarkdown: true,
                usesANSIStyles: false,
                usesColor: false
            ),
            writer: output.write
        )

        renderer.append("safe \u{001B}[31mred\u{0007}")
        renderer.finish()

        XCTAssertEqual(output.value, "safe ␛[31mred␇\n")
        XCTAssertFalse(output.value.contains("\u{001B}"))
    }

    private func makePlainRenderer(
        writer: @escaping @Sendable (String) -> Void
    ) -> TerminalMarkdownStream {
        TerminalMarkdownStream(
            presentation: .init(
                rendersMarkdown: true,
                usesANSIStyles: false,
                usesColor: false
            ),
            writer: writer
        )
    }
}

private final class MarkdownOutputRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""
    private var writes = 0

    var value: String {
        lock.lock()
        defer { lock.unlock() }
        return text
    }

    var writeCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return writes
    }

    func write(_ value: String) {
        lock.lock()
        defer { lock.unlock() }
        text += value
        writes += 1
    }
}
