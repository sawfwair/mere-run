import Combine
import Foundation
import MereRunContract

// The contract resolves a managed id, a blank model, and selector flags on its own. A model it
// does not list (a local folder, an upstream alias) is a question only the CLI can answer: it
// runs the same Core identifier as the gate, behind `mere.run catalog resolve --json`. Studio
// asks once per folder and remembers the answer until the folder changes, and every surface
// reads that answer through `StudioModelIdentifying` so tests can hand in a fixed one.

/// What Studio knows about a model the contract does not list.
package enum StudioModelIdentity: Equatable, Sendable {
    /// `catalog resolve` named its family, or the managed model it is an alias of.
    case identified(MereRunModelIdentification)
    /// The CLI could not say, or could not be asked.
    case unidentified
    /// The question is out; the answer re-renders every surface that asked.
    case pending
}

/// Answers the contract resolver's `identify` hook for one capability. Never blocks: a model
/// nobody has asked about yet reads as `.pending` while the lookup runs.
package protocol StudioModelIdentifying: Sendable {
    /// `flag` is the option that carries `model` on the command line (`--model`,
    /// `--model-root`), which the CLI's identifier may read.
    func identity(of model: String, flag: String, for capability: MereRunCommandCapability) -> StudioModelIdentity
}

/// The same answer every time, per model value: what tests and offscreen renders use in place of
/// the CLI. A model the map does not name is unidentified.
package struct StudioFixedModelIdentities: StudioModelIdentifying {
    package var answers: [String: StudioModelIdentity]

    package init(_ answers: [String: StudioModelIdentity] = [:]) {
        self.answers = answers
    }

    package func identity(of model: String, flag: String, for capability: MereRunCommandCapability) -> StudioModelIdentity {
        answers[model] ?? .unidentified
    }
}

/// The app's answers from `catalog resolve`, keyed by capability, standardized path, and the
/// folder's modification date, so editing a checkpoint folder asks again. `MereRunController`
/// supplies the resolver (a utility-lane CLI run) and republishes each answer, so every surface
/// that showed the full option list while it waited re-renders scoped.
///
/// Reads are synchronous and safe from any thread: argv builders consult the store from wherever
/// they run. Answers are recorded on the main actor.
package final class StudioModelIdentityStore: ObservableObject, StudioModelIdentifying, @unchecked Sendable {
    /// Runs `mere.run catalog resolve --json -- <commandLine>` and decodes its report; nil when
    /// the command failed or printed something else.
    package typealias Resolver = @MainActor @Sendable (_ commandLine: [String]) async -> MereRunFamilyResolutionReport?

    package static let shared = StudioModelIdentityStore()

    private struct Key: Hashable {
        let capability: String
        let model: String
        let modified: Date?
    }

    private let lock = NSLock()
    private var answers: [Key: StudioModelIdentity] = [:]
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
            generation += 1
        }
        objectWillChange.send()
    }

    /// Forgets every answer, so the next read asks again: the CLI or the model location changed.
    @MainActor
    package func forget() {
        lock.withLock {
            answers = [:]
            generation += 1
        }
        objectWillChange.send()
    }

    package func identity(of model: String, flag: String, for capability: MereRunCommandCapability) -> StudioModelIdentity {
        let key = Self.key(model: model, capability: capability)
        let lookup: (resolver: Resolver, generation: Int)? = lock.withLock {
            guard answers[key] == nil, let resolver else { return nil }
            answers[key] = .pending
            return (resolver, generation)
        }
        guard let lookup else {
            return lock.withLock { answers[key] } ?? .unidentified
        }
        let commandLine = capability.command + [flag, model]
        Task { @MainActor [weak self] in
            let report = await lookup.resolver(commandLine)
            self?.record(report.map { Self.identity(from: $0, in: capability) } ?? .unidentified,
                         for: key, generation: lookup.generation)
        }
        return .pending
    }

    @MainActor
    private func record(_ identity: StudioModelIdentity, for key: Key, generation: Int) {
        let recorded = lock.withLock {
            guard generation == self.generation else { return false }
            answers[key] = identity
            return true
        }
        if recorded { objectWillChange.send() }
    }

    /// What a `catalog resolve` report says about the model it was asked about: the managed model
    /// an alias names (listed or excluded), else the family a folder belongs to.
    package static func identity(
        from report: MereRunFamilyResolutionReport,
        in capability: MereRunCommandCapability
    ) -> StudioModelIdentity {
        if let model = report.model, let routing = capability.routing,
           routing.excludedModel(id: model) != nil || routing.families.contains(where: { $0.models.contains(model) }) {
            return .identified(.managedModel(model))
        }
        if let family = report.family { return .identified(.family(family)) }
        return .unidentified
    }

    /// A folder is keyed by its absolute path and modification date; anything else by its text.
    private static func key(model: String, capability: MereRunCommandCapability) -> Key {
        guard model.hasPrefix("/") || model.hasPrefix("~") else {
            return Key(capability: capability.id, model: model, modified: nil)
        }
        let url = URL(fileURLWithPath: NSString(string: model).expandingTildeInPath).standardizedFileURL
        let modified = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        return Key(capability: capability.id, model: url.path, modified: modified)
    }
}
