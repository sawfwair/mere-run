import Foundation
import MereRunContract

/// The command a user approved, before the launcher adds global configuration and transport flags.
/// History and replay retain this vector rather than reconstructing it from a simplified form.
package struct StudioExecution: Codable, Equatable {
    package let templateID: CommandTemplateID
    package let arguments: [String]

    package init(templateID: CommandTemplateID, arguments: [String]) {
        self.templateID = templateID
        self.arguments = arguments
    }

    package var form: StudioConsoleDraft? {
        templateID.capability.map { StudioConsoleCommand.seed(capability: $0, arguments: arguments) }
    }

    package var validationMessage: String? {
        guard let capability = templateID.capability, let form else { return nil }
        return StudioConsoleCommand.validationMessage(for: capability, draft: form)
    }

    /// Projects fields used by history and artifact discovery. The vector remains authoritative:
    /// options without a simplified control are retained and replayed too.
    package func project(onto seed: CommandDraft) -> CommandDraft {
        guard let capability = templateID.capability, let form,
              let template = CommandCatalog.template(id: templateID) else { return seed }
        return StudioConsoleCommand.commandDraft(seed: seed, template: template, capability: capability, draft: form)
    }

    /// Changes an explicitly selected option without reordering or rebuilding any other argument.
    package func replacing(_ flag: String, with value: String?) -> StudioExecution {
        var result: [String] = []
        var index = 0
        var inserted = false
        while index < arguments.count {
            let token = arguments[index]
            if token == flag || token.hasPrefix(flag + "=") {
                if !inserted, let value { result += [flag, value] }
                inserted = true
                if token == flag, index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") {
                    index += 1
                }
            } else {
                result.append(token)
            }
            index += 1
        }
        if !inserted, let value { result += [flag, value] }
        return StudioExecution(templateID: templateID, arguments: result)
    }

    /// Every replay owns fresh output paths, including explicitly named sidecars.
    package func replay(outputPath: String, seed: String? = nil) -> StudioExecution {
        guard let capability = templateID.capability, let form else { return self }
        var replay = self
        let sidecarStem = outputPath.isBlank ? "rerun-" + UUID().uuidString
            : URL(fileURLWithPath: outputPath).deletingPathExtension().lastPathComponent
        let outputFlags = Set([capability.output.flag].compactMap { $0 })
            .union(["--json-output", "--mask-output-dir", "--structured-prompt-output", "--lrc-output",
                    "--recipe-output", "--daw-bundle", "--timings-output"])
        for option in capability.options where outputFlags.contains(option.flag) {
            let old = form.text(option.flag)
            guard !old.isEmpty else { continue }
            let path: String
            if option.flag == capability.output.flag {
                path = outputPath
            } else {
                let url = URL(fileURLWithPath: old)
                path = url.deletingLastPathComponent()
                    .appendingPathComponent(sidecarStem + "-" + url.lastPathComponent).path
            }
            replay = replay.replacing(option.flag, with: path)
        }
        if let seed { replay = replay.replacing("--seed", with: seed) }
        return replay
    }
}


extension CommandDraft {
    /// Persistable settings never contain launch credentials.
    package var withoutSecrets: CommandDraft {
        var saved = self
        saved.apiKey = ""
        saved.visionInfinityAPIKey = ""
        saved.openWebUIAdminPassword = ""
        saved.extraArguments = ShellWords.split(extraArguments).maskingSecrets().shellQuoted()
        return saved
    }
}

package enum StudioLibraryReplay {
    package static func request(for item: StudioLibraryItem, variationSeed: String? = nil) -> StudioRunRequest? {
        guard let templateID = item.templateID, let template = CommandCatalog.template(id: templateID),
              let stored = item.commandDraft else { return nil }
        let original = StudioExecution(templateID: templateID,
                                       arguments: item.commandArguments ?? template.arguments(from: stored))
        let draft = original.project(onto: stored)
        let namedOutput = draft.outputPath.isBlank ? "" : StudioOutputLocation.namedOutputPath(
            templateID: templateID, outputKind: template.outputKind,
            prompt: draft.prompt, seed: variationSeed ?? draft.seed,
            fingerprint: UUID().uuidString, fallbackStem: template.title, existing: draft.outputPath
        )
        // Reserve identity at submission, before either of two simultaneous runs creates a file.
        let namedURL = URL(fileURLWithPath: namedOutput)
        let output = namedOutput.isEmpty ? "" : namedURL.deletingPathExtension().path + "-"
            + UUID().uuidString.prefix(8) + (namedURL.pathExtension.isEmpty ? "" : "." + namedURL.pathExtension)
        let execution = original.replay(outputPath: output, seed: variationSeed)
        return StudioRunRequest(mode: item.mode, templateID: templateID, template: template,
                                draft: stored.withoutSecrets, execution: execution, parentID: item.id)
    }
}


extension String {
    package func maskingAPIKeyValue() -> String {
        ShellWords.split(self).maskingSecrets().shellQuoted()
    }
}
