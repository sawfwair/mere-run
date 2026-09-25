import Foundation
import MereRunContract

// Which options a surface shows, validates, and sends is one decision: the contract resolves the
// runtime family from the argv the surface would launch (`MereRunCommandCapability.resolveFamily`)
// and `options(forFamily:)` says what that family takes. `StudioOptionScope` is that answer, and
// every surface reads it — the composer's fields, chips, and wells, the task inspector, the
// Command panel, the console, validation, readiness, the model pickers, and the argv builders.
//
// Values the scope hides stay in the draft, so switching back to a model brings them back; they
// are reset in a scoped copy before validation and launch, and the surface's note lists them.

/// Whether a surface is waiting on `catalog resolve` for its model.
package enum StudioIdentityState: Equatable, Sendable {
    /// The contract answered on its own (a managed id, the default, a selector), or the command
    /// has no routing.
    case notNeeded
    /// The CLI is identifying `model`; every option shows meanwhile.
    case pending(model: String)
    /// The CLI could not identify `model`; every option shows and the CLI checks them at run time.
    case failed(model: String)
}

/// Where scopes come from: the contract the app ships, and what the CLI has said about local
/// models. Tests and offscreen renders substitute either.
package struct StudioScopeSource: Sendable {
    package var identities: any StudioModelIdentifying
    /// The capability for a capability id.
    package var capability: @Sendable (_ id: String) -> MereRunCommandCapability?

    package init(
        identities: any StudioModelIdentifying,
        capability: @escaping @Sendable (_ id: String) -> MereRunCommandCapability? = { MereRunCapabilityCatalog.command(id: $0) }
    ) {
        self.identities = identities
        self.capability = capability
    }

    /// The shipped contract and the app's `catalog resolve` answers.
    package static let live = StudioScopeSource(identities: StudioModelIdentityStore.shared)

    package func capability(for templateID: CommandTemplateID) -> MereRunCommandCapability? {
        templateID.capabilityID.flatMap(capability)
    }

    /// The scope of a full command line for `capability`: the command path, then its arguments.
    package func scope(capability: MereRunCommandCapability, commandLine: [String]) -> StudioOptionScope {
        let arguments = commandLine.starts(with: capability.command)
            ? Array(commandLine.dropFirst(capability.command.count)) : commandLine
        return StudioOptionScope(capability: capability, arguments: arguments, identities: identities)
    }
}

/// The options one command line's runtime family takes, read from the contract.
package struct StudioOptionScope: Equatable {
    package let capability: MereRunCommandCapability
    /// The argv after the command path this scope was read from.
    package let invocation: MereRunCommandInvocation
    package let resolution: MereRunFamilyResolution
    package let identity: StudioIdentityState
    /// The family's options with its rules applied (narrowed choices, the family's default and
    /// range); every option, unchanged, when the family is not known.
    package let options: [MereRunCapabilityOption]
    /// Every way the command line leaves the family's scope, errors and warnings, in the gate's
    /// own words.
    package let violations: [MereRunOptionViolation]
    /// Why the CLI would refuse this command line before loading anything, or nil.
    package let refusal: String?

    /// Reads `arguments` (the argv after the command path) through the contract resolver.
    /// `identities` answers only for a model the contract does not list.
    package init(capability: MereRunCommandCapability, arguments: [String], identities: any StudioModelIdentifying) {
        let invocation = MereRunCommandInvocation(capability: capability, arguments: arguments)
        var asked: StudioIdentityState = .notNeeded
        let identify = { (model: String) -> MereRunModelIdentification? in
            guard let routing = capability.routing, !Self.lists(model, routing) else { return nil }
            switch identities.identity(of: model, flag: Self.flag(carrying: model, invocation, routing), for: capability) {
            case .identified(let identification):
                return identification
            case .pending:
                asked = .pending(model: model)
                return nil
            case .unidentified:
                asked = .failed(model: model)
                return nil
            }
        }
        let resolution = capability.resolveFamily(invocation, identify: identify)
        let report = capability.resolutionReport(invocation, identify: identify)
        self.capability = capability
        self.invocation = invocation
        self.resolution = resolution
        // Selectors can name the family of a folder nobody has identified; only a model that
        // left the family open is waiting on the CLI.
        if case .unidentified = resolution { identity = asked } else { identity = .notNeeded }
        let family: String?
        if case .family(let id, _, _) = resolution { family = id } else { family = nil }
        options = capability.options(forFamily: family)
        violations = family.map { capability.violations(invocation, family: $0) } ?? []
        refusal = report.violations.first
    }

    /// The runtime family the command line runs, when the contract (or the CLI) knows it.
    package var family: MereRunRuntimeFamily? {
        guard case .family(let id, _, _) = resolution else { return nil }
        return capability.routing?.family(id: id)
    }

    /// The managed model the command line runs, when it names or defaults to one.
    package var managedModel: String? {
        guard case .family(_, let model?, _) = resolution, let routing = capability.routing,
              Self.lists(model, routing) else { return nil }
        return model
    }

    /// Whether the family takes `flag`. A flag the capability does not declare is not the
    /// scope's to hide, and every flag is allowed while the family is unknown.
    package func allows(_ flag: String) -> Bool {
        guard family != nil, capability.options.contains(where: { $0.flag == flag }) else { return true }
        return options.contains { $0.flag == flag }
    }

    /// `flag`'s option narrowed to the family; nil when the family does not take it.
    package func option(_ flag: String) -> MereRunCapabilityOption? {
        options.first { $0.flag == flag }
    }

    /// The one value the family runs `flag` with, when its rule fixes it.
    package func fixedValue(_ flag: String) -> String? {
        guard family != nil, let values = option(flag)?.familyRules.first?.values, values.count == 1 else { return nil }
        return values[0]
    }

    /// Flags the command line passes that the family does not use, or with a value its rule
    /// turns away. A missing required option is not one: the run needs it, not less of it.
    package var unusedFlags: Set<String> {
        Set(violations.filter { $0.kind != .missingRequired }.map(\.flag))
    }

    /// Whether the family takes `flag` with every value in `values`.
    package func accepts(_ flag: String, values: [String]) -> Bool {
        guard allows(flag) else { return false }
        guard let family else { return true }
        let occurrence = MereRunCommandInvocation(capability: capability, arguments: values.flatMap { [flag, $0] })
        return !capability.violations(occurrence, family: family.id)
            .contains { $0.flag == flag && $0.kind != .missingRequired }
    }

    private static func lists(_ model: String, _ routing: MereRunCapabilityRouting) -> Bool {
        routing.excludedModel(id: model) != nil || routing.families.contains { $0.models.contains(model) }
    }

    private static func flag(
        carrying model: String,
        _ invocation: MereRunCommandInvocation,
        _ routing: MereRunCapabilityRouting
    ) -> String {
        let flags = routing.modelFlags + routing.families.compactMap(\.modelFlag)
        return flags.first { invocation.value($0) == model } ?? flags.first ?? "--model"
    }
}

// MARK: - Reading argv

/// One unit of the argv after a command path.
package enum StudioArgvToken: Equatable {
    /// A declared option under its canonical flag, the tokens it spans, and its values.
    case option(flag: String, tokens: [String], values: [String])
    case positional(String)
    /// An option token the contract does not declare, kept verbatim.
    case undeclared(String)

    package var tokens: [String] {
        switch self {
        case .option(_, let tokens, _): return tokens
        case .positional(let token), .undeclared(let token): return [token]
        }
    }

    /// `arguments` read the way ArgumentParser reads them for `capability`: every spelling folds
    /// into its canonical flag through `MereRunCommandInvocation`, and an option takes the next
    /// token exactly when its kind takes a value (a Boolean never does), whatever that token
    /// looks like — so `--target-peak-db -1` is one option, not two.
    package static func read(_ arguments: [String], capability: MereRunCommandCapability) -> [StudioArgvToken] {
        var result: [StudioArgvToken] = []
        var index = 0
        while index < arguments.count {
            let token = arguments[index]
            if token == "--" {
                result.append(.undeclared(token))
                result += arguments[(index + 1)...].map(StudioArgvToken.positional)
                break
            }
            guard token.hasPrefix("-"), token.count > 1 else {
                result.append(.positional(token))
                index += 1
                continue
            }
            let spelled = MereRunCommandInvocation(capability: capability, arguments: [token])
            guard let flag = spelled.values.keys.first,
                  let option = capability.options.first(where: { $0.flag == flag }) else {
                result.append(.undeclared(token))
                index += 1
                continue
            }
            let takesNext = option.kind != .boolean && !token.contains("=") && index + 1 < arguments.count
            let tokens = Array(arguments[index..<(index + (takesNext ? 2 : 1))])
            let values = MereRunCommandInvocation(capability: capability, arguments: tokens).values[flag] ?? []
            result.append(.option(flag: flag, tokens: tokens, values: values))
            index += tokens.count
        }
        return result
    }
}

// MARK: - Filtering argv

package enum StudioOptionScopes {
    /// `argv` (a full command line) without the declared options `scope`'s family does not use,
    /// whose value its rule turns away, past its most occurrences, or that repeat the value the
    /// family runs when the option is left off. Positionals and undeclared tokens stay, in order.
    ///
    /// The builders call this before appending Extra arguments, which stay a raw escape hatch
    /// that the CLI's gate answers. A replay calls it on the whole recorded command.
    package static func filtered(_ argv: [String], scope: StudioOptionScope) -> [String] {
        guard let family = scope.family, argv.starts(with: scope.capability.command) else { return argv }
        let path = scope.capability.command.count
        var kept = Array(argv.prefix(path))
        var counts: [String: Int] = [:]
        for token in StudioArgvToken.read(Array(argv.dropFirst(path)), capability: scope.capability) {
            guard case let .option(flag, tokens, values) = token else {
                kept += token.tokens
                continue
            }
            guard let option = scope.option(flag) else {
                // A flag the capability declares is dropped when the family does not use it; one
                // it does not declare is not the scope's to judge.
                if !scope.capability.options.contains(where: { $0.flag == flag }) { kept += tokens }
                continue
            }
            counts[flag, default: 0] += 1
            guard scope.accepts(flag, values: values) else { continue }
            guard let rule = option.familyRules.first(where: { $0.family == family.id }) else {
                kept += tokens
                continue
            }
            if let maximum = rule.maxCount, counts[flag, default: 0] > maximum { continue }
            // The family runs its own default when the option is left off, so saying it again
            // adds nothing — and on a folder not identified yet it would be a flag another
            // family refuses.
            if let familyDefault = rule.defaultValue, values.count == 1, sameValue(values[0], familyDefault) { continue }
            kept += tokens
        }
        return kept
    }

    private static func sameValue(_ lhs: String, _ rhs: String) -> Bool {
        if let left = Double(lhs), let right = Double(rhs) { return left == right }
        return lhs == rhs
    }
}
