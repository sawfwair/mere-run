import AppKit
import SwiftUI

/// The Studio's one navigation: fifteen domains in four sections inside the native sidebar
/// column, with the machine status as the only footer element. Native collapse, resize, and
/// material come from `NavigationSplitView`; shortcuts live in the Go menu.
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
                Section(group.rawValue) {
                    ForEach(group.domains) { domain in
                        row(domain)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .tint(MereRunTheme.accent)
        .safeAreaInset(edge: .top, spacing: 0) { brandHeader }
        .safeAreaInset(edge: .bottom, spacing: 0) { footer }
        .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 300)
    }

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

    // The wordmark sits under the traffic lights at the top of the sidebar; the serif treatment is
    // the only branding the shell carries.
    private var brandHeader: some View {
        HStack {
            Text("mere.run")
                .font(.system(size: 16, weight: .semibold, design: .serif))
                .foregroundStyle(MereRunTheme.textPrimary)
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 0)
        }
        .padding(.leading, 20)
        .padding(.top, MereRunTheme.Spacing.xs)
        .padding(.bottom, MereRunTheme.Spacing.xs)
    }

    private func row(_ domain: StudioDomain) -> some View {
        let unavailable = unavailableMessages[domain]
        return Label(domain.title, systemImage: domain.systemImage)
            .opacity(unavailable == nil ? 1 : 0.5)
            .help(unavailable ?? domain.subtitle)
            .accessibilityHint(unavailable ?? domain.subtitle)
            .tag(domain)
    }

    private var footer: some View {
        StudioStatusCluster(
            serverStatus: serverStatus,
            resolvedCLI: resolvedCLI,
            onShowServer: onShowServer,
            onShowModels: onShowModels
        )
        .padding(.horizontal, MereRunTheme.Spacing.sm)
        .padding(.vertical, MereRunTheme.Spacing.sm)
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

    var body: some View {
        Button {
            showDetails.toggle()
        } label: {
            HStack(spacing: 7) {
                Circle()
                    .fill(dotColor)
                    .frame(width: 7, height: 7)
                Text(summary)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
                Spacer(minLength: 0)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 9)
            .frame(height: 30)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                    .fill(hovering ? MereRunTheme.hoverFill : MereRunTheme.surface.opacity(0.4))
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                            .strokeBorder(MereRunTheme.border.opacity(0.5), lineWidth: 1)
                    }
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.base))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .popover(isPresented: $showDetails, arrowEdge: .top) {
            StudioStatusDetails(
                serverStatus: serverStatus,
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
        .accessibilityValue(summary)
        .accessibilityHint("Shows local server details and opens the Server domain")
    }

    private var summary: String {
        guard let serverStatus else { return "Checking status…" }
        if serverStatus.isReachable {
            return serverStatus.loadedModelSummary.map { "Server up · \($0)" } ?? "Server up"
        }
        let count = serverStatus.installedCount
        return count == 1 ? "1 model local" : "\(count) models local"
    }

    private var dotColor: Color {
        guard let serverStatus else { return MereRunTheme.textMuted }
        return serverStatus.isReachable ? MereRunTheme.green : MereRunTheme.textMuted.opacity(0.7)
    }
}

private struct StudioStatusDetails: View {
    let serverStatus: StudioServerStatus?
    let resolvedCLI: String
    let onShowServer: () -> Void
    let onShowModels: () -> Void

    @State private var copiedCLI = false

    var body: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            detailRow(
                icon: "bolt.horizontal.circle",
                title: "Local server",
                value: serverValue,
                valueColor: serverStatus?.isReachable == true ? MereRunTheme.green : MereRunTheme.textSecondary
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
                    value: modelsValue,
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

    private var serverValue: String {
        guard let serverStatus else { return "Checking…" }
        if serverStatus.isReachable {
            return serverStatus.loadedModelSummary.map { "Up · \($0)" } ?? "Up"
        }
        return "Not running — starts on demand"
    }

    private var modelsValue: String {
        guard let serverStatus else { return "—" }
        return "\(serverStatus.installedCount) installed"
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
