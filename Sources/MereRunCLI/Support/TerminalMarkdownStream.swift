import Foundation

/// An append-only Markdown presenter for token streams. It deliberately avoids
/// cursor movement so partially generated output never flickers or rewrites.
final class TerminalMarkdownStream {
    private enum ANSI {
        static let reset = "\u{001B}[0m"
        static let bold = "\u{001B}[1m"
        static let boldOff = "\u{001B}[22m"
        static let italic = "\u{001B}[3m"
        static let italicOff = "\u{001B}[23m"
        static let strike = "\u{001B}[9m"
        static let strikeOff = "\u{001B}[29m"
        static let accent = "\u{001B}[38;5;75m"
        static let code = "\u{001B}[38;5;215m"
        static let colorOff = "\u{001B}[39m"
    }

    private let presentation: TerminalMarkdownPresentation
    private let writer: @Sendable (String) -> Void
    private var writeBuffer = ""

    private var atLineStart = true
    private var linePrefix = ""
    private var readingFenceHeader = false
    private var fenceHeader = ""
    private var inCodeFence = false
    private var fenceLength = 3
    private var codeLinePrefix = ""
    private var codeLineStarted = false

    private var pendingAsterisks = 0
    private var pendingTildes = 0
    private var previousInlineCharacter: Character?
    private var escaped = false
    private var bold = false
    private var italic = false
    private var strike = false
    private var inlineCode = false
    private var headingStyle = false
    private var endedWithNewline = true

    init(
        presentation: TerminalMarkdownPresentation,
        writer: @escaping @Sendable (String) -> Void
    ) {
        self.presentation = presentation
        self.writer = writer
    }

    func append(_ text: String) {
        for character in text {
            consume(character)
        }
        flushWrites()
    }

    func finish() {
        if readingFenceHeader {
            openFence()
        }

        if !inCodeFence, let marker = linePrefix.first, linePrefix.count >= 3 {
            if "-*_".contains(marker), linePrefix.allSatisfy({ $0 == marker }) {
                linePrefix = ""
                emitMarker(String(repeating: "─", count: 32))
            } else if marker == "`", linePrefix.allSatisfy({ $0 == "`" }) {
                fenceLength = linePrefix.count
                linePrefix = ""
                openFence()
            }
        }

        if inCodeFence {
            if !codeLineStarted, isClosingFence(codeLinePrefix) {
                codeLinePrefix = ""
            } else {
                flushPendingCodeLine()
            }
            emitCodeBorder("└─")
            emit("\n")
            inCodeFence = false
        } else {
            flushLinePrefixAsInline()
            finishInlineLine()
        }

        if presentation.usesANSIStyles {
            queue(ANSI.reset)
        }
        if !endedWithNewline {
            emit("\n")
        }
        flushWrites()
    }

    private func consume(_ character: Character) {
        if readingFenceHeader {
            if character == "\n" {
                openFence()
            } else {
                fenceHeader.append(character)
            }
            return
        }

        if inCodeFence {
            consumeCode(character)
            return
        }

        if atLineStart {
            consumeLineStart(character)
            return
        }

        consumeInline(character)
    }

    private func consumeLineStart(_ character: Character) {
        if linePrefix.isEmpty {
            if character == "\n" {
                emit("\n")
            } else if isPotentialBlockPrefix(character) {
                linePrefix.append(character)
            } else {
                atLineStart = false
                consumeInline(character)
            }
            return
        }

        let first = linePrefix.first
        switch first {
        case "#":
            consumeHeadingPrefix(character)
        case ">":
            if character == " " {
                emitMarker("│ ")
                linePrefix = ""
                atLineStart = false
            } else {
                resolvePrefixAsInline(with: character)
            }
        case "+":
            if character == " " {
                emitMarker("• ")
                linePrefix = ""
                atLineStart = false
            } else {
                resolvePrefixAsInline(with: character)
            }
        case "-", "*", "_":
            consumeListOrRulePrefix(character)
        case "`":
            consumeFencePrefix(character)
        default:
            consumeNumberedPrefix(character)
        }
    }

    private func consumeHeadingPrefix(_ character: Character) {
        if character == "#", linePrefix.count < 6 {
            linePrefix.append(character)
        } else if character == " ", (1...6).contains(linePrefix.count) {
            linePrefix = ""
            atLineStart = false
            headingStyle = true
            if presentation.usesANSIStyles { queue(ANSI.bold) }
            if presentation.usesColor { queue(ANSI.accent) }
        } else {
            resolvePrefixAsInline(with: character)
        }
    }

    private func consumeListOrRulePrefix(_ character: Character) {
        guard let marker = linePrefix.first else { return }
        if character == marker {
            linePrefix.append(character)
            return
        }
        if character == " ", linePrefix.count == 1, marker != "_" {
            emitMarker("• ")
            linePrefix = ""
            atLineStart = false
            return
        }
        if character == "\n", linePrefix.count >= 3 {
            emitMarker(String(repeating: "─", count: 32))
            emit("\n")
            linePrefix = ""
            atLineStart = true
            return
        }
        resolvePrefixAsInline(with: character)
    }

    private func consumeFencePrefix(_ character: Character) {
        if character == "`" {
            linePrefix.append(character)
            return
        }
        guard linePrefix.count >= 3 else {
            resolvePrefixAsInline(with: character)
            return
        }

        fenceLength = linePrefix.count
        linePrefix = ""
        if character == "\n" {
            openFence()
        } else {
            readingFenceHeader = true
            fenceHeader.append(character)
        }
    }

    private func consumeNumberedPrefix(_ character: Character) {
        if character.isNumber, !linePrefix.contains(".") {
            linePrefix.append(character)
        } else if character == ".", !linePrefix.contains(".") {
            linePrefix.append(character)
        } else if character == " ", linePrefix.last == "." {
            emitMarker(linePrefix + " ")
            linePrefix = ""
            atLineStart = false
        } else {
            resolvePrefixAsInline(with: character)
        }
    }

    private func resolvePrefixAsInline(with character: Character) {
        let prefix = linePrefix
        linePrefix = ""
        atLineStart = false
        for buffered in prefix {
            consumeInline(buffered)
        }
        consumeInline(character)
    }

    private func flushLinePrefixAsInline() {
        guard !linePrefix.isEmpty else { return }
        let prefix = linePrefix
        linePrefix = ""
        for buffered in prefix {
            consumeInline(buffered)
        }
    }

    private func consumeInline(_ character: Character) {
        if character == "\n" {
            resolvePendingMarkers(next: nil)
            finishInlineLine()
            emit("\n")
            atLineStart = true
            previousInlineCharacter = nil
            return
        }

        if escaped {
            resolvePendingMarkers(next: character)
            escaped = false
            if "\\`*{}[]()#+-.!_>~".contains(character) {
                emitSanitized(character)
            } else {
                emit("\\")
                emitSanitized(character)
            }
            return
        }

        if character == "*" {
            resolvePendingTildes(next: character)
            pendingAsterisks += 1
            return
        }
        if character == "~" {
            resolvePendingAsterisks(next: character)
            pendingTildes += 1
            return
        }

        resolvePendingMarkers(next: character)

        if character == "\\" {
            escaped = true
            return
        }

        switch character {
        case "`":
            inlineCode.toggle()
            if presentation.usesColor {
                queue(inlineCode ? ANSI.code : ANSI.colorOff)
            }
        default:
            emitSanitized(character)
        }
    }

    private func finishInlineLine() {
        resolvePendingMarkers(next: nil)
        if escaped {
            escaped = false
            emit("\\")
        }

        bold = false
        italic = false
        strike = false
        inlineCode = false
        if headingStyle || presentation.usesANSIStyles {
            emitANSI(ANSI.reset)
        }
        headingStyle = false
    }

    private func resolvePendingMarkers(next: Character?) {
        resolvePendingAsterisks(next: next)
        resolvePendingTildes(next: next)
    }

    private func resolvePendingAsterisks(next: Character?) {
        guard pendingAsterisks > 0 else { return }
        var count = pendingAsterisks
        pendingAsterisks = 0

        while count >= 2 {
            if bold, canCloseEmphasis(next: next) {
                bold = false
                emitANSI(ANSI.boldOff)
            } else if !bold, canOpenEmphasis(next: next) {
                bold = true
                emitANSI(ANSI.bold)
            } else {
                emitInlineLiteral("**")
            }
            count -= 2
        }

        if count == 1 {
            if italic, canCloseEmphasis(next: next) {
                italic = false
                emitANSI(ANSI.italicOff)
            } else if !italic, canOpenEmphasis(next: next) {
                italic = true
                emitANSI(ANSI.italic)
            } else {
                emitInlineLiteral("*")
            }
        }
    }

    private func resolvePendingTildes(next: Character?) {
        guard pendingTildes > 0 else { return }
        var count = pendingTildes
        pendingTildes = 0

        while count >= 2 {
            if strike, canCloseEmphasis(next: next) {
                strike = false
                emitANSI(ANSI.strikeOff)
            } else if !strike, canOpenEmphasis(next: next) {
                strike = true
                emitANSI(ANSI.strike)
            } else {
                emitInlineLiteral("~~")
            }
            count -= 2
        }

        if count == 1 {
            emitInlineLiteral("~")
        }
    }

    private func canOpenEmphasis(next: Character?) -> Bool {
        guard let next, !next.isMarkdownWhitespace else { return false }
        guard let previousInlineCharacter else { return true }
        return previousInlineCharacter.isMarkdownWhitespace
            || previousInlineCharacter.isMarkdownPunctuation
    }

    private func canCloseEmphasis(next: Character?) -> Bool {
        guard let previousInlineCharacter,
              !previousInlineCharacter.isMarkdownWhitespace else {
            return false
        }
        guard let next else { return true }
        return next.isMarkdownWhitespace || next.isMarkdownPunctuation
    }

    private func emitInlineLiteral(_ text: String) {
        emit(text)
        previousInlineCharacter = text.last
    }

    private func openFence() {
        let language = fenceHeader.trimmingCharacters(in: .whitespacesAndNewlines)
        emitCodeBorder(language.isEmpty ? "┌─" : "┌─ \(language)")
        emit("\n")
        fenceHeader = ""
        readingFenceHeader = false
        inCodeFence = true
        atLineStart = true
        codeLineStarted = false
        previousInlineCharacter = nil
    }

    private func consumeCode(_ character: Character) {
        if !codeLineStarted {
            if character == "`" || character == " " {
                codeLinePrefix.append(character)
                return
            }
            if character == "\n" {
                if isClosingFence(codeLinePrefix) {
                    codeLinePrefix = ""
                    emitCodeBorder("└─")
                    emit("\n")
                    inCodeFence = false
                    atLineStart = true
                    previousInlineCharacter = nil
                } else {
                    emitCodeLinePrefix()
                    for buffered in codeLinePrefix { emitSanitized(buffered) }
                    codeLinePrefix = ""
                    emit("\n")
                    codeLineStarted = false
                    previousInlineCharacter = nil
                }
                return
            }

            emitCodeLinePrefix()
            for buffered in codeLinePrefix { emitSanitized(buffered) }
            codeLinePrefix = ""
            codeLineStarted = true
        }

        if character == "\n" {
            emit("\n")
            codeLineStarted = false
            previousInlineCharacter = nil
        } else {
            emitSanitized(character)
        }
    }

    private func flushPendingCodeLine() {
        if !codeLinePrefix.isEmpty || codeLineStarted {
            if !codeLineStarted { emitCodeLinePrefix() }
            for buffered in codeLinePrefix { emitSanitized(buffered) }
            codeLinePrefix = ""
            if !endedWithNewline { emit("\n") }
        }
        codeLineStarted = false
    }

    private func isClosingFence(_ candidate: String) -> Bool {
        let trimmed = candidate.trimmingCharacters(in: .whitespaces)
        return trimmed.count >= fenceLength && trimmed.allSatisfy { $0 == "`" }
    }

    private func emitMarker(_ marker: String) {
        if presentation.usesColor { queue(ANSI.accent) }
        emit(marker)
        if presentation.usesColor { queue(ANSI.colorOff) }
    }

    private func emitCodeBorder(_ border: String) {
        if presentation.usesColor { queue(ANSI.code) }
        emit(border)
        if presentation.usesColor { queue(ANSI.colorOff) }
    }

    private func emitCodeLinePrefix() {
        emitCodeBorder("│ ")
    }

    private func emitANSI(_ sequence: String) {
        guard presentation.usesANSIStyles else { return }
        queue(sequence)
    }

    private func emitSanitized(_ character: Character) {
        for scalar in character.unicodeScalars {
            switch scalar.value {
            case 0x00...0x1F:
                if scalar.value == 0x09 {
                    emit("\t")
                } else {
                    emit(String(UnicodeScalar(0x2400 + scalar.value)!))
                }
            case 0x7F:
                emit("␡")
            case 0x80...0x9F:
                emit("�")
            default:
                emit(String(scalar))
            }
        }
        previousInlineCharacter = character
    }

    private func emit(_ text: String) {
        guard !text.isEmpty else { return }
        queue(text)
        endedWithNewline = text.last == "\n"
    }

    private func queue(_ text: String) {
        writeBuffer += text
    }

    private func flushWrites() {
        guard !writeBuffer.isEmpty else { return }
        writer(writeBuffer)
        writeBuffer = ""
    }

    private func isPotentialBlockPrefix(_ character: Character) -> Bool {
        character.isNumber || "#>*+-_`".contains(character)
    }
}

private extension Character {
    var isMarkdownWhitespace: Bool {
        unicodeScalars.allSatisfy(CharacterSet.whitespacesAndNewlines.contains)
    }

    var isMarkdownPunctuation: Bool {
        unicodeScalars.allSatisfy(CharacterSet.punctuationCharacters.contains)
    }
}
