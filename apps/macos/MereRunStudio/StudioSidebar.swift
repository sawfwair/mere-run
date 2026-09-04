import AppKit
import SwiftUI

/// The Studio's one navigation: fifteen domains in four sections inside the native sidebar
/// column, with the machine status as the only footer element. Native collapse, resize, and
/// material come from `NavigationSplitView`; shortcuts live in the Go menu.
///
/// The list keeps `List(selection:)` for keyboard navigation and VoiceOver, but draws its own
/// selection: the native highlight is switched off on the backing table view and the selected row
/// paints a solid `accent` pill with `onAccent` glyph and label, so it looks the same whether or
/// not the window is key.
struct StudioSidebar: View {
    @Binding var selectedDomain: StudioDomain
    let unavailableMessages: [StudioDomain: String]
    let status: StudioMachineStatus
    /// How many jobs are running right now; the footer pill counts them beside the machine state.
    let runningJobs: Int
    /// Whether the Activity popover is open. The popover itself is drawn by the shell, over the
    /// whole window: it is wider than this column and would otherwise be clipped by it.
    @Binding var isActivityOpen: Bool

    var body: some View {
        List(selection: selectionBinding) {
            ForEach(StudioDomainGroup.allCases) { group in
                Section {
                    ForEach(group.domains) { domain in
                        row(domain)
                    }
                } header: {
                    MereEyebrow(group.rawValue)
                        .padding(.leading, 10)
                        .padding(.bottom, 1)
                        .padding(.top, group == StudioDomainGroup.allCases.first ? 0 : Self.groupSpacing)
                }
            }
        }
        .listStyle(.sidebar)
        .scrollContentBackground(.hidden)
        .background(NativeSelectionHighlightHider())
        .environment(\.defaultMinListRowHeight, Self.rowHeight)
        .safeAreaInset(edge: .top, spacing: 0) { brandHeader }
        .safeAreaInset(edge: .bottom, spacing: 0) { footer }
        .navigationSplitViewColumnWidth(min: 200, ideal: StudioLayoutPolicy.sidebarWidth, max: 300)
    }

    static let rowHeight: CGFloat = 30
    /// Extra space above a section's eyebrow; the sidebar list's own section gap already lands
    /// close to the design's 18pt between groups.
    static let groupSpacing: CGFloat = 0
    /// The sidebar list insets its rows by this much on each side; the rows pull back out so the
    /// selection pill runs 10pt from the column edges like the design.
    static let nativeRowInset: CGFloat = 6

    /// `List(selection:)` wants an optional; the shell always has a domain, so deselection
    /// (clicking empty space) keeps the current one.
    private var selectionBinding: Binding<StudioDomain?> {
        Binding(
            get: { selectedDomain },
            set: { next in
                if let next { selectedDomain = next }
            }
        )
    }

    // The wordmark sits under the traffic lights at the top of the sidebar; it is the only
    // branding the shell carries.
    private var brandHeader: some View {
        HStack {
            StudioWordmark()
            Spacer(minLength: 0)
        }
        .padding(.leading, 20)
        .padding(.bottom, 6)
    }

    private func row(_ domain: StudioDomain) -> some View {
        let unavailable = unavailableMessages[domain]
        let selected = domain == selectedDomain
        return HStack(spacing: 10) {
            Image(systemName: domain.systemImage)
                .font(.system(size: 13, weight: .medium))
                .frame(width: 16, height: 16)
                .foregroundStyle(selected ? MereRunTheme.onAccent : MereRunTheme.textMuted)
            Text(domain.title)
                .font(.system(size: 13, weight: selected ? .semibold : .medium))
                .foregroundStyle(selected ? MereRunTheme.onAccent : MereRunTheme.textPrimary)
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .frame(height: Self.rowHeight)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                .fill(selected ? MereRunTheme.accent : Color.clear)
        }
        .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        .opacity(unavailable == nil ? 1 : 0.5)
        .listRowInsets(EdgeInsets(top: 0, leading: -Self.nativeRowInset, bottom: 0, trailing: -Self.nativeRowInset))
        .listRowSeparator(.hidden)
        .help(unavailable ?? domain.subtitle)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(domain.title)
        .accessibilityHint(unavailable ?? domain.subtitle)
        .accessibilityAddTraits(selected ? .isSelected : [])
        .tag(domain)
    }

    private var footer: some View {
        StudioStatusCluster(status: status, runningJobs: runningJobs, isActivityOpen: $isActivityOpen)
            .padding(.horizontal, 10)
            .padding(.top, 10)
            .padding(.bottom, 14)
    }
}

/// `mere` and a green period in Caveat Medium, 27pt: the brand, drawn once.
struct StudioWordmark: View {
    var body: some View {
        (Text("mere").foregroundStyle(MereRunTheme.textPrimary)
            + Text(".").foregroundStyle(MereRunTheme.wordmarkGreen))
            .font(MereRunTheme.Brand.font())
            .lineLimit(1)
            .fixedSize()
            .accessibilityLabel("mere.run")
            .accessibilityAddTraits(.isHeader)
    }
}

/// What the footer says about this Mac, derived from the status probe. Keeping the mapping here
/// (not in the view) lets every state, including the probe never answering, be tested.
enum StudioMachineStatus: Equatable {
    /// The probe has not answered yet.
    case checking
    /// The probe did not answer within its grace period: the CLI could not report on the server.
    case unreachable
    /// The CLI answered and the local server is idle — it starts on demand, so this is "ready".
    case ready(installedModels: Int)
    /// The CLI answered and the local server is up.
    case serving(installedModels: Int, loadedModel: String?)

    /// The footer waits this long for a first answer before calling the server unreachable. The
    /// probe itself times out after about a second; two probe intervals cover a slow first launch.
    static let checkingGracePeriod: TimeInterval = 6

    init(serverStatus: StudioServerStatus?, probeTimedOut: Bool) {
        guard let serverStatus else {
            self = probeTimedOut ? .unreachable : .checking
            return
        }
        if serverStatus.isReachable {
            self = .serving(
                installedModels: serverStatus.installedCount,
                loadedModel: serverStatus.loadedModelSummary
            )
        } else {
            self = .ready(installedModels: serverStatus.installedCount)
        }
    }

    /// The one line in the footer pill.
    var summary: String {
        switch self {
        case .checking:
            return "Checking…"
        case .unreachable:
            return "Server unreachable"
        case .ready(let count):
            return "Ready · \(Self.modelCount(count))"
        case .serving(let count, _):
            return "Serving · \(Self.modelCount(count))"
        }
    }

    /// The footer pill with the work count appended: "Ready · 92 models · 2 running". The count is
    /// dropped when nothing is in flight, so a quiet machine keeps the quiet line.
    func summary(runningJobs: Int) -> String {
        guard let work = Self.workCount(runningJobs) else { return summary }
        return "\(summary) · \(work)"
    }

    /// "2 running", or nothing at all when the machine is idle. The pill puts this on its own
    /// line — the 212pt sidebar has no room for it beside the state — and reads the two together
    /// as one value for VoiceOver and the tooltip.
    static func workCount(_ runningJobs: Int) -> String? {
        runningJobs > 0 ? "\(runningJobs) running" : nil
    }

    /// The server line in the details popover.
    var serverDetail: String {
        switch self {
        case .checking:
            return "Checking…"
        case .unreachable:
            return "The status probe did not answer. Check the CLI path in Settings."
        case .ready:
            return "Not running — starts on demand"
        case .serving(_, let loadedModel):
            return loadedModel.map { "Up · \($0)" } ?? "Up"
        }
    }

    /// The models line in the details popover.
    var modelsDetail: String {
        switch self {
        case .checking, .unreachable:
            return "—"
        case .ready(let count), .serving(let count, _):
            return "\(count) installed"
        }
    }

    var dotColor: Color {
        switch self {
        case .checking: return MereRunTheme.yellow
        case .unreachable: return MereRunTheme.red
        case .ready, .serving: return MereRunTheme.green
        }
    }

    /// The pill's label color. Only a machine we cannot reach is worth colouring: it is the one
    /// state the user has to act on.
    var summaryColor: Color {
        self == .unreachable ? MereRunTheme.red : MereRunTheme.textSecondary
    }

    var isServing: Bool {
        if case .serving = self { return true }
        return false
    }

    private static func modelCount(_ count: Int) -> String {
        count == 1 ? "1 model" : "\(count) models"
    }
}

/// One quiet line for machine state — dot, word, count — and the button that opens the Activity
/// popover with the work in flight.
private struct StudioStatusCluster: View {
    let status: StudioMachineStatus
    let runningJobs: Int
    @Binding var isActivityOpen: Bool

    @State private var hovering = false

    private var summary: String {
        status.summary(runningJobs: runningJobs)
    }

    var body: some View {
        Button {
            isActivityOpen.toggle()
        } label: {
            HStack(spacing: 8) {
                Circle()
                    .fill(status.dotColor)
                    .frame(width: 8, height: 8)
                // The 212pt sidebar cannot hold the state and the work count on one line, so the
                // count takes a second line rather than truncating the state the user came for.
                VStack(alignment: .leading, spacing: 0) {
                    Text(status.summary)
                        .foregroundStyle(status.summaryColor)
                    if let work = StudioMachineStatus.workCount(runningJobs) {
                        Text(work)
                            .foregroundStyle(MereRunTheme.textMuted)
                    }
                }
                .font(.system(size: 12, weight: .medium))
                .lineLimit(1)
                Spacer(minLength: 0)
                Image(systemName: "chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 10)
            // Tight enough that the two-line pill still clears the Activity popover above it.
            .padding(.vertical, 4)
            .frame(minHeight: 32)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(hovering || isActivityOpen ? MereRunTheme.hoverFill : Color.clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        // The pill splits the line in two; the tooltip and VoiceOver read it back as one.
        .help(summary)
        .accessibilityLabel("Activity and machine status")
        .accessibilityValue(summary)
        .accessibilityHint("Shows the jobs in flight, the local server, and a way into the Server page")
    }
}

/// Switches off the native selection highlight of the `NSTableView` that backs the sidebar
/// `List`, so the rows' own `accent` pill is the only selection drawn. Selection, keyboard
/// navigation, and accessibility are untouched; only the highlight style changes.
private struct NativeSelectionHighlightHider: NSViewRepresentable {
    func makeNSView(context: Context) -> HiderView {
        HiderView(frame: .zero)
    }

    func updateNSView(_ nsView: HiderView, context: Context) {
        nsView.apply()
    }

    final class HiderView: NSView {
        private weak var appliedTo: NSTableView?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            // The table view is created by the List on the same layout pass; look again once it exists.
            DispatchQueue.main.async { [weak self] in self?.apply() }
        }

        override func layout() {
            super.layout()
            apply()
        }

        func apply() {
            if let appliedTo, appliedTo.selectionHighlightStyle == .none { return }
            guard let tableView = enclosingTableView() else { return }
            tableView.selectionHighlightStyle = .none
            appliedTo = tableView
        }

        private func enclosingTableView() -> NSTableView? {
            var ancestor = superview
            var climbed = 0
            while let current = ancestor, climbed < 6 {
                if let found = current.firstDescendant(NSTableView.self, maxDepth: 8) { return found }
                ancestor = current.superview
                climbed += 1
            }
            return nil
        }
    }
}

private extension NSView {
    func firstDescendant<T: NSView>(_ type: T.Type, maxDepth: Int) -> T? {
        guard maxDepth > 0 else { return nil }
        for subview in subviews {
            if let match = subview as? T { return match }
            if let match = subview.firstDescendant(type, maxDepth: maxDepth - 1) { return match }
        }
        return nil
    }
}
