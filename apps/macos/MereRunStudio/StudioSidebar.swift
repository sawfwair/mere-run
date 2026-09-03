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
    let serverStatus: StudioServerStatus?
    let resolvedCLI: String
    let onShowServer: () -> Void
    let onShowModels: () -> Void

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
        StudioStatusCluster(
            serverStatus: serverStatus,
            resolvedCLI: resolvedCLI,
            onShowServer: onShowServer,
            onShowModels: onShowModels
        )
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

    var isServing: Bool {
        if case .serving = self { return true }
        return false
    }

    private static func modelCount(_ count: Int) -> String {
        count == 1 ? "1 model" : "\(count) models"
    }
}

/// One quiet line for machine state — dot, word, count — with the full story in a popover.
private struct StudioStatusCluster: View {
    let serverStatus: StudioServerStatus?
    let resolvedCLI: String
    let onShowServer: () -> Void
    let onShowModels: () -> Void

    @State private var showDetails = false
    @State private var hovering = false
    @State private var probeTimedOut = false

    private var status: StudioMachineStatus {
        StudioMachineStatus(serverStatus: serverStatus, probeTimedOut: probeTimedOut)
    }

    var body: some View {
        Button {
            showDetails.toggle()
        } label: {
            HStack(spacing: 8) {
                Circle()
                    .fill(status.dotColor)
                    .frame(width: 8, height: 8)
                Text(status.summary)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
                Spacer(minLength: 0)
                Image(systemName: "chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 10)
            .frame(height: 32)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(hovering || showDetails ? MereRunTheme.hoverFill : Color.clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        // The controller publishes nil until the probe answers; a probe that never answers must
        // still resolve, so the footer stops saying "Checking…" after the grace period.
        .task(id: serverStatus == nil) {
            guard serverStatus == nil else {
                probeTimedOut = false
                return
            }
            try? await Task.sleep(for: .seconds(StudioMachineStatus.checkingGracePeriod))
            guard !Task.isCancelled, serverStatus == nil else { return }
            probeTimedOut = true
        }
        .popover(isPresented: $showDetails, arrowEdge: .top) {
            StudioStatusDetails(
                status: status,
                resolvedCLI: resolvedCLI,
                onShowServer: {
                    showDetails = false
                    onShowServer()
                },
                onShowModels: {
                    showDetails = false
                    onShowModels()
                }
            )
        }
        .accessibilityLabel("Machine status")
        .accessibilityValue(status.summary)
        .accessibilityHint("Shows local server details and opens the Server domain")
    }
}

private struct StudioStatusDetails: View {
    let status: StudioMachineStatus
    let resolvedCLI: String
    let onShowServer: () -> Void
    let onShowModels: () -> Void

    @State private var copiedCLI = false

    var body: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            detailRow(
                icon: "bolt.horizontal.circle",
                title: "Local server",
                value: status.serverDetail,
                valueColor: serverValueColor
            )

            Button {
                onShowServer()
            } label: {
                Label("Open Server", systemImage: "network")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.merePrimary)

            HStack(alignment: .firstTextBaseline) {
                detailRow(
                    icon: "shippingbox",
                    title: "Models",
                    value: status.modelsDetail,
                    valueColor: MereRunTheme.textSecondary
                )
                Spacer(minLength: 12)
                Button("Browse…", action: onShowModels)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 7) {
                    Image(systemName: "terminal")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(MereRunTheme.accent)
                        .frame(width: 16)
                    Text("CLI")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textMuted)
                    Spacer(minLength: 12)
                    Button {
                        copyCLI()
                    } label: {
                        Image(systemName: copiedCLI ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 10, weight: .semibold))
                            .padding(3)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Copy CLI path")
                    .accessibilityLabel("Copy CLI path")
                }
                Text(resolvedCLI)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .textSelection(.enabled)
                    .lineLimit(2)
                    .truncationMode(.middle)
            }
        }
        .padding(MereRunTheme.Spacing.lg)
        .frame(width: 320)
    }

    private var serverValueColor: Color {
        switch status {
        case .serving: return MereRunTheme.green
        case .unreachable: return MereRunTheme.red
        case .checking, .ready: return MereRunTheme.textSecondary
        }
    }

    private func detailRow(icon: String, title: String, value: String, valueColor: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 7) {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(width: 16)
                Text(title)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            Text(value)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(valueColor)
                .lineLimit(2)
        }
    }

    private func copyCLI() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(resolvedCLI, forType: .string)
        copiedCLI = true
        Task {
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            copiedCLI = false
        }
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
