import AppKit
import SwiftUI

/// A deliberately small Markdown block model for assistant replies. Fenced code gets a real
/// panel (mono, language chip, copy); prose gets headings, lists, and quotes with inline
/// styling. Not a CommonMark engine — just the shapes local models actually emit.
enum StudioMarkdownBlock: Equatable {
    case paragraph(String)
    case heading(level: Int, text: String)
    case bullets([String])
    case numbered([String])
    case quote(String)
    case code(language: String?, content: String)
    case rule
}

enum StudioMarkdownParser {
    /// Parses text into renderable blocks. An unclosed fence swallows the rest of the input as
    /// code, so a streaming reply renders correctly while the closing fence is still in flight.
    static func parse(_ text: String) -> [StudioMarkdownBlock] {
        var blocks: [StudioMarkdownBlock] = []
        var paragraphLines: [String] = []
        var bulletItems: [String] = []
        var numberedItems: [String] = []
        var quoteLines: [String] = []
        var codeLines: [String] = []
        var codeLanguage: String?
        var inCode = false

        func flushProse() {
            if !paragraphLines.isEmpty {
                blocks.append(.paragraph(paragraphLines.joined(separator: "\n")))
                paragraphLines = []
            }
            if !bulletItems.isEmpty {
                blocks.append(.bullets(bulletItems))
                bulletItems = []
            }
            if !numberedItems.isEmpty {
                blocks.append(.numbered(numberedItems))
                numberedItems = []
            }
            if !quoteLines.isEmpty {
                blocks.append(.quote(quoteLines.joined(separator: "\n")))
                quoteLines = []
            }
        }

        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if inCode {
                if trimmed == "```" {
                    blocks.append(.code(language: codeLanguage, content: codeLines.joined(separator: "\n")))
                    codeLines = []
                    codeLanguage = nil
                    inCode = false
                } else {
                    codeLines.append(line)
                }
                continue
            }

            if trimmed.hasPrefix("```") {
                flushProse()
                let language = trimmed.dropFirst(3).trimmingCharacters(in: .whitespaces)
                codeLanguage = language.isEmpty ? nil : language
                inCode = true
                continue
            }

            if trimmed.isEmpty {
                flushProse()
                continue
            }

            if let heading = headingBlock(from: trimmed) {
                flushProse()
                blocks.append(heading)
                continue
            }

            if trimmed == "---" || trimmed == "***" {
                flushProse()
                blocks.append(.rule)
                continue
            }

            if let item = bulletItem(from: trimmed) {
                if !paragraphLines.isEmpty || !numberedItems.isEmpty || !quoteLines.isEmpty { flushProse() }
                bulletItems.append(item)
                continue
            }

            if let item = numberedItem(from: trimmed) {
                if !paragraphLines.isEmpty || !bulletItems.isEmpty || !quoteLines.isEmpty { flushProse() }
                numberedItems.append(item)
                continue
            }

            if trimmed.hasPrefix(">") {
                if !paragraphLines.isEmpty || !bulletItems.isEmpty || !numberedItems.isEmpty { flushProse() }
                quoteLines.append(String(trimmed.dropFirst()).trimmingCharacters(in: .whitespaces))
                continue
            }

            // Indented continuation of the previous list item, if any.
            if line.hasPrefix("  "), !bulletItems.isEmpty {
                bulletItems[bulletItems.count - 1] += " " + trimmed
                continue
            }
            if line.hasPrefix("  "), !numberedItems.isEmpty {
                numberedItems[numberedItems.count - 1] += " " + trimmed
                continue
            }

            if !bulletItems.isEmpty || !numberedItems.isEmpty || !quoteLines.isEmpty { flushProse() }
            paragraphLines.append(line)
        }

        if inCode {
            blocks.append(.code(language: codeLanguage, content: codeLines.joined(separator: "\n")))
        }
        flushProse()
        return blocks
    }

    /// Inline styling (bold/italic/`code`/links) for one prose run.
    static func inline(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace
        )
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }

    private static func headingBlock(from line: String) -> StudioMarkdownBlock? {
        guard line.hasPrefix("#") else { return nil }
        let hashes = line.prefix(while: { $0 == "#" })
        guard hashes.count <= 6 else { return nil }
        let rest = line.dropFirst(hashes.count)
        guard rest.first == " " else { return nil }
        return .heading(
            level: hashes.count,
            text: rest.trimmingCharacters(in: .whitespaces)
        )
    }

    private static func bulletItem(from line: String) -> String? {
        for marker in ["- ", "* ", "• "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count)).trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    private static func numberedItem(from line: String) -> String? {
        let digits = line.prefix(while: \.isNumber)
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let rest = line.dropFirst(digits.count)
        guard rest.hasPrefix(". ") || rest.hasPrefix(") ") else { return nil }
        return String(rest.dropFirst(2)).trimmingCharacters(in: .whitespaces)
    }
}

/// Renders parsed Markdown blocks in the app's voice. Used by assistant chat turns and the
/// in-app guide so long-form text reads as designed content, not dumped text.
struct StudioMarkdownText: View {
    let content: String
    var bodyFont: Font = MereRunTheme.bodyFont

    var body: some View {
        let blocks = StudioMarkdownParser.parse(content)
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                blockView(block)
            }
        }
    }

    @ViewBuilder
    private func blockView(_ block: StudioMarkdownBlock) -> some View {
        switch block {
        case .paragraph(let text):
            Text(StudioMarkdownParser.inline(text))
                .font(bodyFont)
                .foregroundStyle(MereRunTheme.textPrimary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        case .heading(let level, let text):
            Text(StudioMarkdownParser.inline(text))
                .font(headingFont(level))
                .foregroundStyle(MereRunTheme.textPrimary)
                .textSelection(.enabled)
                .padding(.top, level <= 2 ? 4 : 2)
        case .bullets(let items):
            listView(items) { _ in Text("•") }
        case .numbered(let items):
            listView(items) { index in Text("\(index + 1).").monospacedDigit() }
        case .quote(let text):
            Text(StudioMarkdownParser.inline(text))
                .font(bodyFont.italic())
                .foregroundStyle(MereRunTheme.textSecondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(MereRunTheme.Spacing.sm)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                        .fill(MereRunTheme.accentSoft.opacity(0.6))
                }
        case .code(let language, let content):
            StudioCodeBlock(language: language, content: content)
        case .rule:
            Divider().overlay(MereRunTheme.border.opacity(0.5))
        }
    }

    private func listView(_ items: [String], marker: @escaping (Int) -> Text) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    marker(index)
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.accent)
                    Text(StudioMarkdownParser.inline(item))
                        .font(bodyFont)
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(.leading, 2)
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .system(.title3, weight: .semibold)
        case 2: return .system(.headline, weight: .semibold)
        default: return .system(.subheadline, weight: .semibold)
        }
    }
}

/// A fenced code panel: language chip, hover-revealed copy, wrapped monospaced text.
private struct StudioCodeBlock: View {
    let language: String?
    let content: String

    @State private var hovering = false
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Text(language ?? "code")
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .textCase(.uppercase)
                Spacer(minLength: 0)
                Button {
                    copy()
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 10.5, weight: .medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                }
                .buttonStyle(.mereIcon)
                .opacity(hovering || copied ? 1 : 0)
                .accessibilityLabel("Copy code")
            }
            .padding(.horizontal, MereRunTheme.Spacing.sm)
            .padding(.top, MereRunTheme.Spacing.xs)
            .padding(.bottom, 4)

            Text(content)
                .font(.system(size: 12.5, design: .monospaced))
                .foregroundStyle(MereRunTheme.textPrimary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, MereRunTheme.Spacing.sm)
                .padding(.bottom, MereRunTheme.Spacing.sm)
        }
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface)
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                }
        }
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
    }

    private func copy() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(content, forType: .string)
        copied = true
        Task {
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            copied = false
        }
    }
}
