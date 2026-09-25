import Combine
import Foundation
import MereRunContract

// The contract resolves a managed id, a blank model, and selector flags on its own. Some command
// lines only the CLI can settle: a model the contract does not list (a local folder, an upstream
// alias), a managed id whose family depends on what is installed, a default the machine chooses,
// and a command whose own router has the last word. For those Studio asks
// `mere.run catalog resolve --json -- <command line>`, which runs the gate's resolver with Core's
// identifier, choosers, and routers, and remembers the answer until a flag that picks the family,
// or the folder it names, changes. Every surface reads that answer through
// `StudioModelIdentifying`, so tests can hand in a fixed one.

/// What Studio knows about a command line the contract could not settle alone.
package enum StudioModelIdentity: Equatable, Sendable {
    /// `catalog resolve` answered for the whole command line.
    case resolved(MereRunFamilyResolution)
    /// What the model value is, on its own: a family, or the managed model it names. Fixed
    /// answers in tests and offscreen renders; the contract resolves the command line with it.
    case identified(MereRunModelIdentification)
    /// The CLI could not say, or could not be asked.
    case unidentified
    /// The question is out; the answer re-renders every surface that asked.
    case pending
}

/// Answers for command lines the contract cannot settle alone. Never blocks: a command line
/// nobody has asked about yet reads as `.pending` while the lookup runs.
package protocol StudioModelIdentifying: Sendable {
    /// `arguments` is the argv after `capability`'s command path; `model` is the model value it
    /// names, or nil when it names none (a default, a selector-routed command).
    func identity(of arguments: [String], model: String?, for capability: MereRunCommandCapability) -> StudioModelIdentity
}

/// The same answer every time, per model value: what tests and offscreen renders use in place of
/// the CLI. A command line that names no model is looked up under "". Anything the map does not
/// name is unidentified.
package struct StudioFixedModelIdentities: StudioModelIdentifying {
    package var answers: [String: StudioModelIdentity]

    package init(_ answers: [String: StudioModelIdentity] = [:]) {
        self.answers = answers
    }

    package func identity(of arguments: [String], model: String?, for capability: MereRunCommandCapability) -> StudioModelIdentity {
        answers[model ?? ""] ?? .unidentified
    }
}

/// The app's answers from `catalog resolve`, keyed by capability and the tokens that can change
/// the family: the routing flags (`MereRunCapabilityRouting.routingFlags`), a folder by its path
/// and modification date, and every choice and switch, which the CLI's identifier may read (video
/// generate's `--output-mode` picks the folder `video-ltx-av` runs). Typing a prompt or stepping a
/// number never asks again; editing a checkpoint folder does. While a new question about the same
/// model is out, its last answer stands, so a surface does not flash back to every option.
/// `MereRunController` supplies the resolver (a utility-lane CLI run) and republishes each
/// answer, so every surface that waited re-renders scoped.
///
/// Reads are synchronous and safe from any thread: argv builders consult the store from wherever
/// they run. Answers are recorded on the main actor.
package final class StudioModelIdentityStore: ObservableObject, StudioModelIdentifying, @unchecked Sendable {
    /// Runs `mere.run catalog resolve --json -- <commandLine>` and decodes its report; nil when
    /// the command failed or printed something else.
    package typealias Resolver = @MainActor @Sendable (_ commandLine: [String]) async -> MereRunFamilyResolutionReport?

    package static let shared = StudioModelIdentityStore()

    private struct Key: Hashable {
        /// The capability and its model flags' tokens: one model, whatever else the line says.
        let model: [String]
        /// The other tokens that can change the family.
        let routing: [String]
        /// Modification dates of the folders among them, in order.
        let modified: [Date?]
    }

    private let lock = NSLock()
    private var answers: [Key: StudioModelIdentity] = [:]
    /// The latest answer for each model, standing in while a new question about it is out.
    private var latest: [[String]: StudioModelIdentity] = [:]
    private var resolver: Resolver?
    /// Bumped whenever the answers stop applying (another CLI, another model location), so a
    /// lookup that started before cannot record into the new set.
    private var generation = 0

    package init() {}

    /// Starts asking `resolver`, forgetting every earlier answer.
    @MainActor
    package func use(_ resolver: Resolver?) {
        lock.withLock {
            self.resolver = resolver
            answers = [:]
            latest = [:]
            generation += 1
        }
        objectWillChange.send()
    }

    /// Forgets every answer, so the next read asks again: the CLI or the model location changed.
    @MainActor
    package func forget() {
        lock.withLock {
            answers = [:]
            latest = [:]
            generation += 1
        }
        objectWillChange.send()
    }

    package func identity(of arguments: [String], model: String?, for capability: MereRunCommandCapability) -> StudioModelIdentity {
        let key = Self.key(arguments, capability: capability)
        let lookup: (resolver: Resolver, generation: Int)? = lock.withLock {
            guard answers[key] == nil, let resolver else { return nil }
            answers[key] = .pending
            return (resolver, generation)
        }
        guard let lookup else {
            return lock.withLock {
                guard let answer = answers[key] else { return .unidentified }
                return answer == .pending ? latest[key.model] ?? .pending : answer
            }
        }
        let commandLine = capability.command + arguments
        Task { @MainActor [weak self] in
            let report = await lookup.resolver(commandLine)
            self?.record(report.map { Self.identity(from: $0, in: capability) } ?? .unidentified,
                         for: key, generation: lookup.generation)
        }
        return lock.withLock { latest[key.model] } ?? .pending
    }

    @MainActor
    private func record(_ identity: StudioModelIdentity, for key: Key, generation: Int) {
        let recorded = lock.withLock {
            guard generation == self.generation else { return false }
            answers[key] = identity
            latest[key.model] = identity
            return true
        }
        if recorded { objectWillChange.send() }
    }

    /// What a `catalog resolve` report says about the command line it was asked about.
    package static func identity(
        from report: MereRunFamilyResolutionReport,
        in capability: MereRunCommandCapability
    ) -> StudioModelIdentity {
        switch report.resolution(in: capability) {
        case .unidentified, .unrouted: return .unidentified
        case let resolution: return .resolved(resolution)
        }
    }

    /// The command line's model tokens and the other tokens that can change its family, in order,
    /// each folder by its standardized path, and the folders' modification dates.
    private static func key(_ arguments: [String], capability: MereRunCommandCapability) -> Key {
        let routing = capability.routing
        let modelFlags = Set((routing?.modelFlags ?? []) + (routing?.families.compactMap(\.modelFlag) ?? []))
        let routingFlags = routing?.routingFlags ?? []
        var model = [capability.id]
        var other: [String] = []
        var modified: [Date?] = []
        for token in StudioArgvToken.read(arguments, capability: capability) {
            guard case let .option(flag, _, values) = token,
                  let option = capability.options.first(where: { $0.flag == flag }),
                  routingFlags.contains(flag) || option.kind == .choice || option.kind == .boolean else { continue }
            var tokens = [flag]
            for value in values {
                guard value.hasPrefix("/") || value.hasPrefix("~") else {
                    tokens.append(value)
                    continue
                }
                let url = URL(fileURLWithPath: NSString(string: value).expandingTildeInPath).standardizedFileURL
                tokens.append(url.path)
                modified.append(try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
            }
            if modelFlags.contains(flag) { model += tokens } else { other += tokens }
        }
        return Key(model: model, routing: other, modified: modified)
    }
}
