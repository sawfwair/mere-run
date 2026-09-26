import Foundation
import UniformTypeIdentifiers

// "Send to…" on an output: every page and slot that takes the file, read from the same slot
// schema the wells draw (the prompt modes' `attachmentSlots` and every task template's
// `StudioTaskSchema.slots`), never from a per-page table. Choosing one fills the slot through its
// own `attach`, so a sent file lands exactly where a drop or a disk pick would put it.

/// One place an output can be sent: a page, and the slot on it that takes the file.
package struct StudioSendDestination: Identifiable, Equatable {
    package let task: StudioTask
    /// The variant whose well declares the slot, for a task on the shared task workspace; nil
    /// for a prompt mode, whose well does not change with the command.
    package let templateID: CommandTemplateID?
    package let slot: StudioAttachmentSlot

    package var id: String { "\(task.rawValue)|\(templateID?.rawValue ?? "")|\(slot.id)" }

    /// Whether the slot takes `url` for what it is. A slot's catch-all types (`.data`, the
    /// default for a file option the schema has no table entry for) take every file, so they
    /// never make a destination on their own; a folder slot never takes an output file.
    package func takes(_ url: URL) -> Bool {
        let specific = slot.acceptedTypes.filter { !StudioSendDestinations.catchAllTypes.contains($0) }
        guard !specific.isEmpty, !specific.contains(.folder) else { return false }
        return StudioAttachmentSlot.accepts(url, acceptedTypes: specific)
    }

    /// Puts `url` in the slot of a prompt mode's draft, through the slot's own `attach`.
    package func attach(_ url: URL, to draft: inout StudioDraft) {
        slot.attach([url], to: &draft)
    }

    /// Puts `url` in the slot of a task draft: on the variant the draft already runs when that
    /// variant's well has a slot of the same name that takes the file, else on the variant that
    /// declares this slot, which the draft switches to first (its other forms stay parked).
    package func attach(_ url: URL, to draft: inout StudioTaskDraft) {
        let current = StudioTaskSchema.slots(for: draft.templateID).first {
            $0.label == slot.label && $0.accepts(url)
        }
        if let current {
            current.attach([url], to: &draft)
            return
        }
        if let templateID { draft.switchTemplate(to: templateID) }
        slot.attach([url], to: &draft)
    }
}

/// One domain's destinations, as a section of the Send to menu.
package struct StudioSendSection: Identifiable, Equatable {
    package let domain: StudioDomain
    package let items: [Item]

    package var id: StudioDomain { domain }
    /// The section header ("Video", "3D").
    package var title: String { domain.title }

    /// A menu item: the destination and its label.
    package struct Item: Identifiable, Equatable {
        package let destination: StudioSendDestination
        /// The task's name ("Transcribe"), with the slot's when the task takes the file in more
        /// than one ("Generate · Start frame").
        package let title: String

        package var id: String { destination.id }
    }
}

package enum StudioSendDestinations {
    /// Types that take any file, so a slot never becomes a destination through them alone.
    package static let catchAllTypes: Set<UTType> = [.data, .item, .content]

    /// Every slot of every page with a well, in sidebar order: each prompt mode's slots, then
    /// each variant of each task-draft task. A Session task's draft drives a live process, not a
    /// run with inputs, so it has no destination; a Manage or Project page takes what its
    /// variants' wells declare. Built once, on the main actor that draws the menus.
    @MainActor package static let all: [StudioSendDestination] = StudioTask.allCases.flatMap { task -> [StudioSendDestination] in
        if let mode = task.mode {
            return mode.attachmentSlots.map { StudioSendDestination(task: task, templateID: nil, slot: $0) }
        }
        guard task.usesTaskDraft, task.archetype != .session else { return [] }
        return task.variantTemplates.flatMap { template in
            StudioTaskSchema.slots(for: template.id).map {
                StudioSendDestination(task: task, templateID: template.id, slot: $0)
            }
        }
    }

    /// The destinations that take `url`, leaving out `excluding` (the page the output is shown
    /// on, whose own well is "Use as input"). One per task and slot name: a slot several
    /// variants declare alike (3D's image) is offered once, on the first variant. What a slot
    /// takes depends only on the file's extension, so the answer is kept per extension.
    @MainActor package static func destinations(for url: URL, excluding: StudioTask? = nil) -> [StudioSendDestination] {
        let ext = url.pathExtension.lowercased()
        let taking: [StudioSendDestination]
        if let known = byExtension[ext] {
            taking = known
        } else {
            var seen: Set<String> = []
            taking = all.filter { destination in
                destination.takes(url) && seen.insert("\(destination.task.rawValue)|\(destination.slot.label)").inserted
            }
            byExtension[ext] = taking
        }
        return taking.filter { $0.task != excluding }
    }

    @MainActor private static var byExtension: [String: [StudioSendDestination]] = [:]

    /// `destinations` grouped by domain in sidebar order, each labeled.
    package static func sections(_ destinations: [StudioSendDestination]) -> [StudioSendSection] {
        StudioDomain.allCases.compactMap { domain in
            let inDomain = destinations.filter { $0.task.domain == domain }
            guard !inDomain.isEmpty else { return nil }
            let items = inDomain.map { destination in
                let siblings = inDomain.filter { $0.task == destination.task }.count
                let title = siblings > 1 ? "\(destination.task.title) · \(destination.slot.label)" : destination.task.title
                return StudioSendSection.Item(destination: destination, title: title)
            }
            return StudioSendSection(domain: domain, items: items)
        }
    }
}
