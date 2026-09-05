import ArgumentParser
import Foundation

enum TerminalMarkdownMode: String, CaseIterable, ExpressibleByArgument {
    case auto
    case always
    case never
}

struct TerminalMarkdownPresentation: Equatable {
    let rendersMarkdown: Bool
    let usesANSIStyles: Bool
    let usesColor: Bool

    static let raw = TerminalMarkdownPresentation(
        rendersMarkdown: false,
        usesANSIStyles: false,
        usesColor: false
    )

    static func resolve(
        mode: TerminalMarkdownMode,
        stdoutIsTTY: Bool,
        isTextResponse: Bool,
        environment: [String: String]
    ) -> TerminalMarkdownPresentation {
        guard isTextResponse, mode != .never else {
            return .raw
        }

        let terminalSupportsStyles = environment["TERM"]?.lowercased() != "dumb"
        let rendersMarkdown = mode == .always || (stdoutIsTTY && terminalSupportsStyles)
        guard rendersMarkdown else {
            return .raw
        }

        let usesANSIStyles = stdoutIsTTY && terminalSupportsStyles
        return TerminalMarkdownPresentation(
            rendersMarkdown: true,
            usesANSIStyles: usesANSIStyles,
            usesColor: usesANSIStyles && environment["NO_COLOR"] == nil
        )
    }
}
