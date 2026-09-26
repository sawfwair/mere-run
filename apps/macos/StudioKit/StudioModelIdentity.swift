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
    /// The question is out, and this is what the CLI said about a command line with the same
    /// model and routing flags, differing only in choices and switches the family resolver does
    /// not read (a hidden value reset for launch, another output mode). It stands in, so a
    /// folder's surface does not flash back to every option; a changed model or routing flag has
    /// no stand-in, and the contract's own family shows until the answer lands.
    case pendingAfter(MereRunFamilyResolution)
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

/// One controller's answers from `catalog resolve`, keyed by capability and the tokens that can
/// change the family: the routing flags (`MereRunCapabilityRouting.routingFlags`), a folder by its
/// standardized path, and every choice and switch, which the CLI's identifier may read (video
/// generate's `--output-mode` picks the folder `video-ltx-av` runs). Typing a prompt or stepping a
/// number never asks again. `MereRunController` supplies the resolver (a utility-lane CLI run),
/// republishes each answer so every surface that waited re-renders scoped, and forgets every
/// answer when what the CLI would say changes: another CLI, model location, or hub cache, a
/// model pulled or removed, the inventory refreshed.
///
/// - A question waits `debounce` for the command line to stop changing, per capability, so a
///   language or a model path typed a letter at a time asks once (the superseded questions are
///   dropped and asked again only if a surface still reads them).
/// - A read never touches the file system: a folder's modification date, which can block on a
///   macOS volume-access prompt for a removable or network volume, is read off the main thread
///   when the CLI is asked, and again once an answer is `recheckFoldersAfter` old, when a changed
///   date drops the answer so the next read asks again.
/// - A failed question (the CLI exited nonzero or printed something else) reads as unidentified
///   and is asked again after `retryDelays`, the last delay repeating; an answer, including the
///   CLI's own "unidentified", stands until it is forgotten.
///
/// Reads are synchronous and safe from any thread: argv builders consult the store from wherever
/// they run. Answers are recorded on the main actor.
package final class StudioModelIdentityStore: ObservableObject, StudioModelIdentifying, @unchecked Sendable {
    /// Runs `mere.run catalog resolve --json -- <commandLine>` and decodes its report; nil when
    /// the command failed or printed something else.
    package typealias Resolver = @MainActor @Sendable (_ commandLine: [String]) async -> MereRunFamilyResolutionReport?

    private struct Key: Hashable {
        /// The capability, then its model flags' and routing flags' tokens: the command line as
        /// the family resolver reads it.
        let resolver: [String]
        /// The choices and switches outside the routing flags, which only the CLI's identifier
        /// may read (the folder `video-ltx-av` runs follows `--output-mode`).
        let choices: [String]
        /// The folders among the tokens, whose modification dates an answer is kept against.
        let folders: [URL]
    }

    private enum Entry {
        /// Asked, or about to be. `attempt` counts the failures before it.
        case asking(ticket: Int, attempt: Int)
        case answered(StudioModelIdentity, modified: [Date?], checked: ContinuousClock.Instant)
        case failed(attempt: Int, retryAt: ContinuousClock.Instant)
    }

    private let debounce: Duration
    private let recheckFoldersAfter: Duration
    private let retryDelays: [Duration]
    private let modificationDate: @Sendable (URL) -> Date?
    private let clock = ContinuousClock()

    private let lock = NSLock()
    private var entries: [Key: Entry] = [:]
    /// The latest answer for each command line as the family resolver reads it, standing in while
    /// a question that differs from it only in other choices and switches is out: a draft with a
    /// hidden choice reset, or another output mode. A new model or routing flag has no stand-in.
    private var latest: [[String]: MereRunFamilyResolution] = [:]
    /// The newest question per capability: an older one still waiting out `debounce` is dropped.
    private var newest: [String: Int] = [:]
    private var tickets = 0
    private var rechecking: Set<Key> = []
    private var resolver: Resolver?
    /// Bumped whenever the answers stop applying, so a lookup that started before cannot record
    /// into the new set.
    private var generation = 0

    package init(
        debounce: Duration = .milliseconds(300),
        recheckFoldersAfter: Duration = .seconds(5),
        retryDelays: [Duration] = [.seconds(1), .seconds(5), .seconds(30)],
        modificationDate: @escaping @Sendable (URL) -> Date? = StudioModelIdentityStore.contentModificationDate
    ) {
        precondition(!retryDelays.isEmpty, "a failed question needs a delay before it is asked again")
        self.debounce = debounce
        self.recheckFoldersAfter = recheckFoldersAfter
        self.retryDelays = retryDelays
        self.modificationDate = modificationDate
    }

    /// A folder's modification date as the file system reports it; nil when there is none.
    package static let contentModificationDate: @Sendable (URL) -> Date? = { url in
        try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
    }

    /// Starts asking `resolver`, forgetting every earlier answer.
    @MainActor
    package func use(_ resolver: Resolver?) {
        lock.withLock {
            self.resolver = resolver
            clear()
        }
        objectWillChange.send()
    }

    /// Forgets every answer, so the next read asks again: the CLI, the model location, or the
    /// installed models changed.
    @MainActor
    package func forget() {
        lock.withLock { clear() }
        objectWillChange.send()
    }

    private func clear() {
        entries = [:]
        latest = [:]
        newest = [:]
        rechecking = []
        generation += 1
    }

    package func identity(of arguments: [String], model: String?, for capability: MereRunCommandCapability) -> StudioModelIdentity {
        let key = Self.key(arguments, capability: capability)
        let commandLine = capability.command + arguments.removingSecrets()
        let now = clock.now
        var ask: (ticket: Int, attempt: Int, waits: Bool)?
        var recheck: [Date?]?
        let answer: StudioModelIdentity = lock.withLock {
            let pending = latest[key.resolver].map(StudioModelIdentity.pendingAfter) ?? .pending
            switch entries[key] {
            case nil:
                guard resolver != nil else { return .unidentified }
                tickets += 1
                entries[key] = .asking(ticket: tickets, attempt: 0)
                newest[capability.id] = tickets
                ask = (tickets, 0, true)
                return pending
            case .asking:
                return pending
            case let .answered(identity, modified, checked):
                if !key.folders.isEmpty, now - checked >= recheckFoldersAfter, rechecking.insert(key).inserted {
                    recheck = modified
                }
                return identity
            case let .failed(attempt, retryAt):
                if now >= retryAt, resolver != nil {
                    tickets += 1
                    entries[key] = .asking(ticket: tickets, attempt: attempt)
                    ask = (tickets, attempt, false)
                }
                return .unidentified
            }
        }
        if let ask {
            let generation = lock.withLock { self.generation }
            Task { @MainActor [weak self] in
                await self?.ask(key, commandLine: commandLine, capability: capability, ticket: ask.ticket,
                                attempt: ask.attempt, waits: ask.waits, generation: generation)
            }
        }
        if let recheck {
            Task { @MainActor [weak self] in await self?.recheck(key, against: recheck) }
        }
        return answer
    }

    /// Waits out `debounce` (a first question only), reads the folders' dates off the main
    /// thread, asks the CLI, and records what it says.
    @MainActor
    private func ask(
        _ key: Key,
        commandLine: [String],
        capability: MereRunCommandCapability,
        ticket: Int,
        attempt: Int,
        waits: Bool,
        generation: Int
    ) async {
        if waits, debounce > .zero { try? await Task.sleep(for: debounce) }
        let resolver: Resolver? = lock.withLock {
            guard self.generation == generation, case .asking(ticket, _)? = entries[key] else { return nil }
            if waits, newest[capability.id] != ticket {
                // A newer question about this command superseded it while it waited; a surface
                // that still reads this command line asks again.
                entries[key] = nil
                return nil
            }
            return self.resolver
        }
        guard let resolver else { return }
        let modified = await dates(of: key.folders)
        let report = await resolver(commandLine)
        let now = clock.now
        let recorded: Bool = lock.withLock {
            guard self.generation == generation, case .asking(ticket, _)? = entries[key] else { return false }
            guard let report else {
                let delay = retryDelays[min(attempt, retryDelays.count - 1)]
                entries[key] = .failed(attempt: attempt + 1, retryAt: now + delay)
                return true
            }
            let identity = Self.identity(from: report, in: capability)
            entries[key] = .answered(identity, modified: modified, checked: now)
            if case .resolved(let resolution) = identity { latest[key.resolver] = resolution }
            return true
        }
        guard recorded else { return }
        objectWillChange.send()
        if report == nil {
            // Surfaces re-read once the retry is due, which asks again.
            try? await Task.sleep(for: retryDelays[min(attempt, retryDelays.count - 1)])
            if lock.withLock({ self.generation == generation }) { objectWillChange.send() }
        }
    }

    /// Reads the folders' dates again, off the main thread; a changed one drops the answer.
    @MainActor
    private func recheck(_ key: Key, against modified: [Date?]) async {
        let generation = lock.withLock { self.generation }
        let current = await dates(of: key.folders)
        let now = clock.now
        let changed: Bool = lock.withLock {
            rechecking.remove(key)
            guard self.generation == generation, case let .answered(identity, recorded, _)? = entries[key],
                  recorded == modified else { return false }
            guard current == modified else {
                entries[key] = nil
                return true
            }
            entries[key] = .answered(identity, modified: recorded, checked: now)
            return false
        }
        if changed { objectWillChange.send() }
    }

    private func dates(of folders: [URL]) async -> [Date?] {
        guard !folders.isEmpty else { return [] }
        let modificationDate = modificationDate
        return await Task.detached(priority: .utility) { folders.map(modificationDate) }.value
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

    /// The command line's tokens that can change its family, in order: the model flags', then the
    /// other routing flags', then the other choices and switches, each folder by its standardized
    /// path. Reads no file.
    private static func key(_ arguments: [String], capability: MereRunCommandCapability) -> Key {
        let routing = capability.routing
        let modelFlags = Set((routing?.modelFlags ?? []) + (routing?.families.compactMap(\.modelFlag) ?? []))
        let routingFlags = routing?.routingFlags ?? []
        var model = [capability.id]
        var routed: [String] = []
        var other: [String] = []
        var folders: [URL] = []
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
                folders.append(url)
            }
            if modelFlags.contains(flag) {
                model += tokens
            } else if routingFlags.contains(flag) {
                routed += tokens
            } else {
                other += tokens
            }
        }
        return Key(resolver: model + ["--"] + routed, choices: other, folders: folders)
    }
}
