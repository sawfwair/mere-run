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

// MARK: - Custom validation

extension CommandCatalog {
    /// The reason the custom template's draft cannot run, beyond the prompt and input checks
    /// every template shares; nil for a draft that can, and for every other template.
    package static func customValidationMessage(for id: CommandTemplateID, draft: CommandDraft) -> String? {
        switch id {
        case .custom:
            if ShellWords.split(draft.extraArguments).isEmpty {
                return "Enter mere.run arguments."
            }
        default:
            break
        }
        return nil
    }
}
