import AppKit
import StudioKit
import SwiftUI

// What an output offers wherever it is shown — a feed card, an output tile, a result row, a stem,
// a Library row: "Use as input" on the page showing it, "Send to" any other page and slot that
// takes the file (`StudioSendDestinations`), and "Share…" through the system share picker. One
// set of views serves every page; the window root supplies where "Use as input" and "Send to"
// land through `studioOutputRouting`.

// MARK: - Routing

/// Where an output's "Use as input" and "Send to" land, for whatever page the window shows.
struct StudioOutputRouting {
    /// The page the output is shown on. Its own well is "Use as input", so it is never also a
    /// "Send to" destination.
    let currentTask: StudioTask
    /// That page's well, for the model its draft runs.
    let inputSlots: [StudioAttachmentSlot]
    /// The other pages whose wells show a slot that takes the file for the model each runs.
    let destinations: (URL) -> [StudioSendDestination]
    let useAsInput: (URL) -> Void
    /// Fills the destination's slot, opens its page, and focuses its composer.
    let send: (URL, StudioSendDestination) -> Void

    func canUseAsInput(_ url: URL) -> Bool {
        inputSlots.contains { $0.accepts(url) }
    }

    /// Whether another page's well declares a slot for the file at all: read per extension, so
    /// a row decides whether to offer Send to without asking each page's model.
    @MainActor func offersSendTo(_ url: URL) -> Bool {
        !StudioSendDestinations.destinations(for: url, excluding: currentTask).isEmpty
    }
}

private struct StudioOutputRoutingKey: EnvironmentKey {
    static var defaultValue: StudioOutputRouting? { nil }
}

extension EnvironmentValues {
    /// Set by the window root. Without it (a view hosted alone) outputs offer neither
    /// "Use as input" nor "Send to".
    var studioOutputRouting: StudioOutputRouting? {
        get { self[StudioOutputRoutingKey.self] }
        set { self[StudioOutputRoutingKey.self] = newValue }
    }
}

// MARK: - Send to

/// "Use as input" and the "Send to" submenu, for a context menu or a card's "…" menu. Each is
/// left out when nothing takes the file.
struct StudioSendToMenuItems: View {
    let url: URL

    @Environment(\.studioOutputRouting) private var routing

    var body: some View {
        if let routing {
            if routing.canUseAsInput(url) {
                Button { routing.useAsInput(url) } label: {
                    Label("Use as input", systemImage: StudioSendToSymbols.useAsInput)
                }
            }
            if routing.offersSendTo(url) {
                Menu {
                    StudioSendToSections(url: url, routing: routing)
                } label: {
                    Label("Send to", systemImage: StudioSendToSymbols.sendTo)
                }
            }
        }
    }
}

/// A card's visible "Send to" control: "Use as input" first when the page takes the file, then
/// every other page that does, by section. Hidden when nothing takes the file.
struct StudioSendToButton: View {
    let url: URL

    @Environment(\.studioOutputRouting) private var routing

    var body: some View {
        if let routing {
            let usable = routing.canUseAsInput(url)
            let sendable = routing.offersSendTo(url)
            if usable || sendable {
                Menu {
                    if usable {
                        Button { routing.useAsInput(url) } label: {
                            Label("Use as input", systemImage: StudioSendToSymbols.useAsInput)
                        }
                        if sendable { Divider() }
                    }
                    if sendable { StudioSendToSections(url: url, routing: routing) }
                } label: {
                    MereSecondaryMenuLabel("Send to", systemImage: StudioSendToSymbols.sendTo, showsChevron: true)
                }
                .menuStyle(.button)
                .buttonStyle(.plain)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("Use \(url.lastPathComponent) as the input of another task")
                .accessibilityLabel("Send to")
                .accessibilityHint("Use this output as the input of another task")
            }
        }
    }
}

/// The destinations by domain, each section headed by the domain's name ("Video") and each item
/// the task, with the slot when the task takes the file in more than one. Read from each page's
/// model when the menu's items are built, so a slot the model hides is never offered.
private struct StudioSendToSections: View {
    let url: URL
    let routing: StudioOutputRouting

    var body: some View {
        let sections = StudioSendDestinations.sections(routing.destinations(url))
        if sections.isEmpty {
            Text("No other task's model takes this file")
        }
        ForEach(sections) { section in
            Section(section.title) {
                ForEach(section.items) { item in
                    Button { routing.send(url, item.destination) } label: {
                        Label(item.title, systemImage: item.destination.task.presentation.systemImage)
                    }
                    .accessibilityLabel("\(section.title), \(item.title)")
                }
            }
        }
    }
}

private enum StudioSendToSymbols {
    static let useAsInput = "arrow.down.to.line"
    static let sendTo = "arrow.turn.up.right"
}

// MARK: - Share

/// The view the share picker opens from: the row or card a context menu was opened on, or the
/// share button itself.
@MainActor
final class StudioShareAnchor {
    fileprivate weak var view: NSView?
    /// The picker on screen; AppKit does not keep it alive while it shows.
    private var picker: NSSharingServicePicker?

    func share(_ urls: [URL]) {
        guard let view, !urls.isEmpty else { return }
        let picker = NSSharingServicePicker(items: urls)
        self.picker = picker
        picker.show(relativeTo: view.bounds, of: view, preferredEdge: .minY)
    }
}

private struct StudioShareAnchorKey: EnvironmentKey {
    static let defaultValue: StudioShareAnchor? = nil
}

extension EnvironmentValues {
    /// The anchor `studioShareAnchor()` set for the views under it and their context menus.
    var studioShareAnchor: StudioShareAnchor? {
        get { self[StudioShareAnchorKey.self] }
        set { self[StudioShareAnchorKey.self] = newValue }
    }
}

extension View {
    /// Makes this view where "Share…" opens the share picker from, for the items under it and
    /// its context menu. Apply it outside `.contextMenu`, so the menu's items read it.
    func studioShareAnchor() -> some View {
        modifier(StudioShareAnchorModifier())
    }

    /// Every action on an output file — "Use as input", "Send to", "Share…" — plus Quick Look,
    /// Open, and Reveal in Finder, as the view's context menu.
    func studioOutputContextMenu(_ url: URL) -> some View {
        contextMenu {
            StudioSendToMenuItems(url: url)
            StudioShareMenuItem(urls: [url])
            Divider()
            Button("Quick Look") { QuickLookCoordinator.shared.preview(url) }
            Button("Open") { NSWorkspace.shared.open(url) }
            Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
        }
        .studioShareAnchor()
    }
}

private struct StudioShareAnchorModifier: ViewModifier {
    @State private var anchor = StudioShareAnchor()

    func body(content: Content) -> some View {
        content
            .environment(\.studioShareAnchor, anchor)
            .background(StudioShareAnchorView(anchor: anchor))
    }
}

private struct StudioShareAnchorView: NSViewRepresentable {
    let anchor: StudioShareAnchor

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        anchor.view = view
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        anchor.view = nsView
    }
}

/// "Share…" in a menu: the system share picker for `urls`, opened from the view the menu belongs
/// to. Left out where no anchor is set.
struct StudioShareMenuItem: View {
    let urls: [URL]

    @Environment(\.studioShareAnchor) private var anchor

    var body: some View {
        if let anchor {
            Button { anchor.share(urls) } label: {
                Label("Share…", systemImage: "square.and.arrow.up")
            }
            .disabled(urls.isEmpty)
        }
    }
}

/// A card's share button: the system share picker for `urls`, opened from the button.
struct StudioShareButton: View {
    let urls: [URL]

    var body: some View {
        StudioShareButtonFace(urls: urls)
            .studioShareAnchor()
    }
}

private struct StudioShareButtonFace: View {
    let urls: [URL]

    @Environment(\.studioShareAnchor) private var anchor

    var body: some View {
        Button { anchor?.share(urls) } label: {
            Image(systemName: "square.and.arrow.up")
                .font(.callout.weight(.medium))
                .frame(width: 28, height: 28)
        }
        .buttonStyle(.mereIcon)
        .help(urls.count == 1 ? "Share \(urls[0].lastPathComponent)" : "Share \(urls.count) files")
        .accessibilityLabel("Share")
        .accessibilityHint(urls.count == 1 ? "Opens the share picker for this output" : "Opens the share picker for these outputs")
    }
}
