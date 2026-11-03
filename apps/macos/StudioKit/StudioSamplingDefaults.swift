import Foundation
import MereRunContract

/// Untouched sampling controls belong to the CLI's selected model, not the app's
/// generic draft. Keep this policy shared by prompt tasks, task forms, and Console.
package enum StudioSamplingDefaults {
    private static let flags: Set<String> = [
        "--steps", "--cfg", "--cfg-scale", "--guidance", "--guidance-scale", "--sigma-shift", "--shift",
        "--video-cfg-guidance-scale", "--audio-cfg-guidance-scale", "--a2v-guidance-scale",
        "--v2a-guidance-scale", "--a2v-steps", "--ode-steps", "--temperature", "--top-p",
        "--top-k", "--min-p"
    ]

    package static func applies(to flag: String, in capability: MereRunCommandCapability) -> Bool {
        !capability.command.contains(where: { $0.hasPrefix("train") || $0.hasPrefix("benchmark") })
            && capability.options.contains { $0.flag == flag && isSamplingOption($0) }
    }

    package static func isSamplingOption(_ option: MereRunCapabilityOption) -> Bool {
        flags.contains(option.flag) && (option.kind == .integer || option.kind == .number)
    }

    /// Keep selectors and attachments identical while comparing sampling fields.
    /// Some builders emit steps only for a particular model or quality profile.
    package static func baseline(for draft: CommandDraft, template: CommandTemplate) -> CommandDraft {
        let defaults = template.defaultDraft()
        var baseline = draft
        let integers: [WritableKeyPath<CommandDraft, Int>] = [\.steps, \.topK, \.a2vSteps, \.musicTopK]
        let numbers: [WritableKeyPath<CommandDraft, Double>] = [
            \.cfgScale, \.sigmaShift, \.scheduleShift, \.temperature, \.topP, \.minP,
            \.a2vGuidanceScale, \.v2aGuidanceScale, \.videoCFGGuidanceScale, \.audioCFGGuidanceScale,
            \.musicTemperature
        ]
        for key in integers { baseline[keyPath: key] = defaults[keyPath: key] }
        for key in numbers { baseline[keyPath: key] = defaults[keyPath: key] }
        baseline.audioGuidanceScale = defaults.audioGuidanceScale
        baseline.audioODESteps = defaults.audioODESteps
        baseline.speechCFGScale = defaults.speechCFGScale
        baseline.h3Steps = defaults.h3Steps
        baseline.musicGuidanceScale = defaults.musicGuidanceScale
        baseline.musicShift = defaults.musicShift
        baseline.operationsRuntimeTemperature = defaults.operationsRuntimeTemperature
        baseline.operationsRuntimeTopP = defaults.operationsRuntimeTopP
        baseline.operationsRuntimeMinP = defaults.operationsRuntimeMinP
        return baseline
    }

    /// A value inherited from the template is automatic. A different value, or a
    /// deliberate override equal to that value, remains an explicit argument.
    package static func filtering(
        _ argv: [String], baseline: [String], capability: MereRunCommandCapability, overrides: Set<String>
    ) -> [String] {
        let count = capability.command.count
        let baselineValues = Dictionary(
            StudioArgvToken.read(Array(baseline.dropFirst(count)), capability: capability).compactMap { token -> (String, [String])? in
                guard case let .option(flag, _, values) = token else { return nil }
                return (flag, values)
            }, uniquingKeysWith: { first, _ in first }
        )
        return capability.command + StudioArgvToken.read(Array(argv.dropFirst(count)), capability: capability).flatMap { token in
            guard case let .option(flag, _, values) = token,
                  applies(to: flag, in: capability), !overrides.contains(flag),
                  let inherited = baselineValues[flag], values == inherited else { return token.tokens }
            return []
        }
    }

    package static func binding(
        _ raw: StudioContractBinding<StudioDraft>, flag: String, mode: StudioMode, baseline: StudioDraft
    ) -> StudioContractBinding<StudioDraft> {
        guard let capability = mode.defaultTemplateID.capability, applies(to: flag, in: capability) else { return raw }
        let kind = capability.options.first { $0.flag == flag }?.kind
        return StudioContractBinding(
            fieldID: raw.fieldID,
            read: { draft in
                if draft.samplingOverrides?.contains(flag) == true || raw.read(draft) != raw.read(baseline) {
                    return raw.read(draft)
                }
                return .unset
            },
            write: { draft, value in
                var value = value
                if case .text(let text) = value {
                    if text.isEmpty {
                        value = .unset
                    } else if kind == .integer, let integer = Int(text) {
                        value = .integer(integer)
                    } else if kind == .number, let number = Double(text) {
                        value = .number(number)
                    } else { return }
                }
                if value == .unset {
                    raw.reset(&draft, to: baseline)
                    draft.samplingOverrides?.remove(flag)
                } else {
                    raw.write(&draft, value)
                    if draft.samplingOverrides == nil { draft.samplingOverrides = [] }
                    draft.samplingOverrides?.insert(flag)
                }
            }, storage: raw.storage
        )
    }

    /// Older forms saved every template value, including its generic sampling
    /// defaults. Migrate those inherited values once; recorded argv is seeded as
    /// an explicit form and never takes this path.
    package static func migrating(_ form: StudioConsoleDraft, capability: MereRunCommandCapability) -> StudioConsoleDraft {
        guard form.modelSamplingDefaults == nil,
              let template = CommandCatalog.templates.first(where: { $0.id.capability?.id == capability.id }) else { return form }
        let baseline = StudioConsoleCommand.seed(
            capability: capability,
            arguments: CommandArguments.build(for: template.id, draft: template.defaultDraft(), source: .contract)
        )
        var next = form
        for option in capability.options where applies(to: option.flag, in: capability) {
            let sameNumber = next[option.flag].numericValue.map { $0 == baseline[option.flag].numericValue } ?? false
            if next[option.flag] == baseline[option.flag] || sameNumber { next.values[option.flag] = nil }
        }
        next.modelSamplingDefaults = true
        return next
    }
}
