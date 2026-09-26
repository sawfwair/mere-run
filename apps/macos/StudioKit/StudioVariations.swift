import Foundation

/// How many runs "Run variations" submits at once.
package enum StudioVariationCount: Int, CaseIterable, Identifiable, Sendable {
    case two = 2
    case four = 4
    case eight = 8

    package var id: Int { rawValue }

    /// "2 variations", for the menus that offer the count.
    package var title: String { "\(rawValue) variations" }
}

/// Where one row sits in the variation group it was submitted with: "2 of 4".
package struct StudioVariationPosition: Equatable, Sendable {
    package let group: UUID
    /// 1-based, in submission order.
    package let index: Int
    package let count: Int

    package init(group: UUID, index: Int, count: Int) {
        self.group = group
        self.index = index
        self.count = count
    }

    package var title: String { "Variation \(index) of \(count)" }
}

/// "Run variations": the same command N times, each with its own recorded random seed, filed in
/// the Library as one group (`StudioLibraryItem.variationGroup`) so Compare can open them together.
/// Only a command that takes `--seed` for the model it runs can vary; everywhere else the action
/// is left out rather than offered to run the same thing N times.
package enum StudioVariations {
    package static let seedFlag = "--seed"

    /// The seeds a run draws from: positive and within a 32-bit signed integer, which every
    /// runtime family's `--seed` accepts.
    package static let seedRange = 1...Int(Int32.max)

    /// Whether the command `arguments` launch takes a seed for the model it runs.
    package static func takesSeed(templateID: CommandTemplateID, arguments: [String], source: StudioScopeSource) -> Bool {
        guard let capability = source.capability(for: templateID) else { return false }
        return source.scope(capability: capability, commandLine: arguments).option(seedFlag) != nil
    }

    /// Whether a Library row can run as variations: a finished or failed run whose recorded
    /// command takes a seed. Threads never do.
    package static func applies(to item: StudioLibraryItem, source: StudioScopeSource) -> Bool {
        guard !item.isConversation, item.commandDraft != nil, let templateID = item.templateID,
              let template = CommandCatalog.template(id: templateID), let draft = item.commandDraft else { return false }
        let arguments = item.commandArguments ?? template.arguments(from: draft, source: source)
        return takesSeed(templateID: templateID, arguments: arguments, source: source)
    }

    /// Whether a prompt mode's composer can run its draft as variations.
    package static func applies(mode: StudioMode, draft: StudioDraft, source: StudioScopeSource) -> Bool {
        guard !mode.isConversational, let scope = source.scope(mode: mode, draft: draft) else { return false }
        return scope.option(seedFlag) != nil
    }

    /// Whether a shared task workspace's composer can run its draft as variations.
    package static func applies(to draft: StudioTaskDraft, source: StudioScopeSource) -> Bool {
        source.scope(for: draft)?.option(seedFlag) != nil
    }

    /// `count` distinct seeds, as the argv carries them.
    package static func seeds<Generator: RandomNumberGenerator>(count: Int, using generator: inout Generator) -> [String] {
        var drawn: [Int] = []
        while drawn.count < count {
            let seed = Int.random(in: seedRange, using: &generator)
            if !drawn.contains(seed) { drawn.append(seed) }
        }
        return drawn.map(String.init)
    }

    package static func seeds(count: Int) -> [String] {
        var generator = SystemRandomNumberGenerator()
        return seeds(count: count, using: &generator)
    }

    /// `request` running with `seed`: the draft's field and, when the request carries its exact
    /// argv (a replay, or a task's Command edits), the `--seed` in that argv too, so a seed the
    /// Command view pinned never makes every variation the same picture.
    package static func seeded(_ request: StudioRunRequest, seed: String) -> StudioRunRequest {
        var draft = request.draft
        draft.seed = seed
        return StudioRunRequest(
            id: request.id, mode: request.mode, templateID: request.templateID, template: request.template,
            draft: draft, createdAt: request.createdAt, conversationID: request.conversationID,
            execution: request.execution?.replacing(seedFlag, with: seed), parentID: request.parentID
        )
    }

    /// A task draft running with `seed`: the form's argv with `--seed` replaced, read back
    /// through the contract so the value lands in the form the way the Command view writes it.
    package static func seeded(_ draft: StudioTaskDraft, seed: String) -> StudioTaskDraft {
        guard let capability = draft.capability else { return draft }
        let arguments = StudioConsoleCommand.arguments(for: capability, draft: draft.form)
        let replaced = StudioExecution(templateID: draft.templateID, arguments: arguments).replacing(seedFlag, with: seed)
        var seeded = draft
        seeded.form = StudioConsoleCommand.seed(capability: capability, arguments: replaced.arguments)
        return seeded
    }

    /// One replay of `item` per seed — the Library's Run again, each with its seed in the
    /// recorded argv and its own fresh destinations. Every request is validated before any is
    /// returned, so a command the CLI would refuse submits none of them.
    package static func replayRequests(
        for item: StudioLibraryItem,
        seeds: [String],
        source: StudioScopeSource
    ) throws -> [StudioRunRequest] {
        try seeds.map { seed in
            guard let request = StudioLibraryReplay.request(for: item, variationSeed: seed, source: source) else {
                throw StudioValidationError(message: "This older Library item does not include a replayable command.")
            }
            if let message = request.template.validationMessage(for: request.draft, execution: request.execution, source: source) {
                throw StudioValidationError(message: message)
            }
            return request
        }
    }

    /// The task a Library row's replays are submitted from: a shared-workspace task by its
    /// command, a prompt task by its mode.
    package static func submittingTask(for item: StudioLibraryItem) -> StudioTask {
        if let task = item.templateID?.studioTask, task.usesTaskDraft { return task }
        return item.mode.task
    }

    /// The rows of `group`, in the order they were submitted.
    package static func members(of group: UUID, in items: [StudioLibraryItem]) -> [StudioLibraryItem] {
        items.filter { $0.variationGroup == group }.sorted { $0.createdAt < $1.createdAt }
    }

    /// Every grouped row's place in its group, for the Library and feed labels.
    package static func positions(in items: [StudioLibraryItem]) -> [UUID: StudioVariationPosition] {
        var grouped: [UUID: [StudioLibraryItem]] = [:]
        for item in items {
            if let group = item.variationGroup { grouped[group, default: []].append(item) }
        }
        var positions: [UUID: StudioVariationPosition] = [:]
        for (group, members) in grouped {
            let ordered = members.sorted { $0.createdAt < $1.createdAt }
            for (offset, member) in ordered.enumerated() {
                positions[member.id] = StudioVariationPosition(group: group, index: offset + 1, count: ordered.count)
            }
        }
        return positions
    }
}
