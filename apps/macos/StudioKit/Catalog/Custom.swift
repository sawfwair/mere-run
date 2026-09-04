import Foundation

// MARK: - Custom templates

extension CommandCatalog {
    package static let customTemplates: [CommandTemplate] = [
        CommandTemplate(
            id: .custom,
            category: .custom,
            title: "Raw arguments",
            subtitle: "Run any mere.run command",
            systemImage: "terminal",
            defaultExtraArguments: "--help"
        )
    ]
}

// MARK: - Custom arguments

extension CommandArguments {
    package static func custom(_ draft: CommandDraft) -> [String] {
        ShellWords.split(draft.extraArguments)
    }
}
