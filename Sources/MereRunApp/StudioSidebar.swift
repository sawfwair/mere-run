import AppKit
import SwiftUI

/// The Studio's primary navigation: grouped modes over the native translucent sidebar
/// material, with the machine status living quietly in the footer. Replaces the old
/// horizontally scrolling chip rail and the always-on status pill row.
struct StudioSidebar: View {
    @Binding var mode: StudioMode
    let modeCapabilities: [StudioMode: StudioModelCapability]
    let serverStatus: StudioServerStatus?
    let resolvedCLI: String
    let onShowServing: () -> Void
    let onShowModels: () -> Void
    let onShowHelp: () -> Void

    @Namespace private var selectionNamespace
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Modes in navigation order, for stable ⌘1…⌘9 shortcuts.
    private static let orderedModes: [StudioMode] = StudioModeGroup.allCases.flatMap(\.modes)

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            brandHeader

            ScrollView {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                    ForEach(StudioModeGroup.allCases) { group in
                        section(group)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.top, 2)
                .padding(.bottom, MereRunTheme.Spacing.md)
            }

            footer
        }
        .frame(width: 212)
        .background {
            VisualEffectBackground(material: .sidebar, blendingMode: .behindWindow)
                .ignoresSafeArea()
        }
    }

    // Sits below the traffic lights; the serif wordmark is the only branding the shell carries.
    private var brandHeader: some View {
        Text("mere.run")
            .font(.system(size: 16, weight: .semibold, design: .serif))
            .foregroundStyle(MereRunTheme.textPrimary)
            .padding(.leading, 20)
            .padding(.top, 44)
            .padding(.bottom, MereRunTheme.Spacing.md)
            .accessibilityAddTraits(.isHeader)
    }

    private func section(_ group: StudioModeGroup) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(group.rawValue.uppercased())
                .font(.system(size: 10.5, weight: .semibold))
                .kerning(0.7)
                .foregroundStyle(MereRunTheme.textMuted)
                .padding(.leading, 10)
                .padding(.bottom, 3)
                .accessibilityAddTraits(.isHeader)

            ForEach(group.modes) { candidate in
                StudioSidebarRow(
                    mode: candidate,
                    isSelected: candidate == mode,
                    unavailableMessage: modeCapabilities[candidate]?.unavailableMessage,
                    shortcutNumber: Self.orderedModes.firstIndex(of: candidate).flatMap { index in
                        index < 9 ? index + 1 : nil
                    },
                    namespace: selectionNamespace
                ) {
                    guard candidate != mode else { return }
                    if reduceMotion {
                        mode = candidate
                    } else {
                        withAnimation(MereRunTheme.Motion.spring) { mode = candidate }
                    }
                }
            }
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.xs) {
            StudioStatusCluster(
                serverStatus: serverStatus,
                resolvedCLI: resolvedCLI,
                onShowServing: onShowServing,
                onShowModels: onShowModels
            )

            Button(action: onShowServing) {
                Label("Serving & Agents", systemImage: "network")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.merePrimary)
            .keyboardShortcut("s", modifiers: [.command, .shift])
            .help("Operate serving, models, resources, agents, and clients (⇧⌘S)")

            HStack(spacing: 8) {
                Button(action: onShowModels) {
                    Label("Models", systemImage: "shippingbox")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.mereSecondary)
                .help("Browse and manage local models (⇧⌘M)")

                Button(action: onShowHelp) {
                    Label("Guide", systemImage: "questionmark.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.mereSecondary)
                .help("Open the offline guide")
            }
        }
        .padding(MereRunTheme.Spacing.sm)
    }
}

/// One mode row. Selection is a bronze pill that slides between rows; hover is a soft fill.
private struct StudioSidebarRow: View {
    let mode: StudioMode
    let isSelected: Bool
    let unavailableMessage: String?
    let shortcutNumber: Int?
    let namespace: Namespace.ID
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 9) {
                Image(systemName: mode.systemImage)
                    .font(.system(size: 12.5, weight: .semibold))
                    .frame(width: 19)
                Text(mode.title)
                    .font(.system(size: 13, weight: isSelected ? .semibold : .medium))
                Spacer(minLength: 0)
            }
            .foregroundStyle(foregroundColor)
            .padding(.horizontal, 10)
            .frame(height: 30)
            .background {
                ZStack {
                    if isSelected {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                            .fill(MereRunTheme.accent)
                            .matchedGeometryEffect(id: "studio.sidebar.selection", in: namespace)
                    } else if hovering && unavailableMessage == nil {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                            .fill(MereRunTheme.hoverFill)
                    }
                }
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .disabled(unavailableMessage != nil)
        .opacity(unavailableMessage == nil ? 1 : 0.45)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .help(helpText)
        .modifier(StudioModeShortcut(number: shortcutNumber))
        .accessibilityLabel(mode.title)
        .accessibilityHint(unavailableMessage ?? mode.subtitle)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var foregroundColor: Color {
        if unavailableMessage != nil { return MereRunTheme.textMuted }
        return isSelected ? MereRunTheme.background : MereRunTheme.textSecondary
    }

    private var helpText: String {
        if let unavailableMessage { return unavailableMessage }
        if let shortcutNumber { return "\(mode.subtitle) (⌘\(shortcutNumber))" }
        return mode.subtitle
    }
}

/// The narrow-window navigation: an icon-only rail so every mode stays reachable when the
/// full sidebar doesn't fit — instead of hiding navigation behind a subtle header menu.
struct StudioSidebarRail: View {
    @Binding var mode: StudioMode
    let modeCapabilities: [StudioMode: StudioModelCapability]
    let onShowServing: () -> Void
    let onShowModels: () -> Void
    let onShowHelp: () -> Void

    @Namespace private var selectionNamespace
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private static let orderedModes: [StudioMode] = StudioModeGroup.allCases.flatMap(\.modes)

    var body: some View {
        VStack(spacing: 0) {
            Text("m")
                .font(.system(size: 18, weight: .semibold, design: .serif))
                .foregroundStyle(MereRunTheme.textPrimary)
                .frame(height: 30)
                .padding(.top, 42)
                .padding(.bottom, MereRunTheme.Spacing.sm)
                .accessibilityHidden(true)

            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 5) {
                    ForEach(Array(StudioModeGroup.allCases.enumerated()), id: \.element) { index, group in
                        if index > 0 {
                            Divider()
                                .overlay(MereRunTheme.border.opacity(0.4))
                                .frame(width: 22)
                                .padding(.vertical, 3)
                        }
                        ForEach(group.modes) { candidate in
                            StudioRailRow(
                                mode: candidate,
                                isSelected: candidate == mode,
                                unavailableMessage: modeCapabilities[candidate]?.unavailableMessage,
                                shortcutNumber: Self.orderedModes.firstIndex(of: candidate).flatMap { index in
                                    index < 9 ? index + 1 : nil
                                },
                                namespace: selectionNamespace
                            ) {
                                guard candidate != mode else { return }
                                if reduceMotion {
                                    mode = candidate
                                } else {
                                    withAnimation(MereRunTheme.Motion.spring) { mode = candidate }
                                }
                            }
                        }
                    }
                }
                .padding(.vertical, 4)
            }

            Spacer(minLength: 0)

            VStack(spacing: 4) {
                railAction(system: "network", help: "Serving & Agents (⇧⌘S)", label: "Serving & Agents", action: onShowServing)
                railAction(system: "shippingbox", help: "Models (⇧⌘M)", label: "Models", action: onShowModels)
                railAction(system: "questionmark.circle", help: "Guide", label: "Guide", action: onShowHelp)
            }
            .padding(.bottom, MereRunTheme.Spacing.sm)
        }
        .frame(width: StudioLayoutPolicy.railWidth)
        .background {
            VisualEffectBackground(material: .sidebar, blendingMode: .behindWindow)
                .ignoresSafeArea()
        }
    }

    private func railAction(
        system: String,
        help: String,
        label: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: system)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(MereRunTheme.textSecondary)
                .frame(width: 38, height: 32)
        }
        .buttonStyle(.mereIcon)
        .help(help)
        .accessibilityLabel(label)
    }
}

/// One icon-only rail row: the same bronze sliding selection pill as the full sidebar,
/// with a soft hover fill and the mode title in a tooltip.
private struct StudioRailRow: View {
    let mode: StudioMode
    let isSelected: Bool
    let unavailableMessage: String?
    let shortcutNumber: Int?
    let namespace: Namespace.ID
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: mode.systemImage)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(foregroundColor)
                .frame(width: 40, height: 34)
                .background {
                    ZStack {
                        if isSelected {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                                .fill(MereRunTheme.accent)
                                .matchedGeometryEffect(id: "studio.rail.selection", in: namespace)
                        } else if hovering && unavailableMessage == nil {
                            RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                                .fill(MereRunTheme.hoverFill)
                        }
                    }
                }
                .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
        }
        .buttonStyle(.plain)
        .disabled(unavailableMessage != nil)
        .opacity(unavailableMessage == nil ? 1 : 0.4)
        .onHover { hovering = $0 }
        .animation(MereRunTheme.Motion.quick, value: hovering)
        .help(helpText)
        .modifier(StudioModeShortcut(number: shortcutNumber))
        .accessibilityLabel(mode.title)
        .accessibilityHint(unavailableMessage ?? mode.subtitle)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private var foregroundColor: Color {
        if unavailableMessage != nil { return MereRunTheme.textMuted }
        return isSelected ? MereRunTheme.background : MereRunTheme.textSecondary
    }

    private var helpText: String {
        if let unavailableMessage { return unavailableMessage }
        if let shortcutNumber { return "\(mode.title) (⌘\(shortcutNumber))" }
        return mode.title
    }
}

/// Applies ⌘1…⌘9 to the first nine modes without duplicating the row body per branch.
struct StudioModeShortcut: ViewModifier {
    let number: Int?

    func body(content: Content) -> some View {
        if let number, let key = "\(number)".first {
            content.keyboardShortcut(KeyEquivalent(key), modifiers: .command)
        } else {
            content
        }
    }
}

/// One quiet line for machine state — dot, word, count — with the full story in a popover.
private struct StudioStatusCluster: View {
    let serverStatus: StudioServerStatus?
    let resolvedCLI: String
    let onShowServing: () -> Void
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
                onShowServing: {
                    showDetails = false
                    onShowServing()
                },
                onShowModels: {
                    showDetails = false
                    onShowModels()
                }
            )
        }
        .accessibilityLabel("Machine status")
        .accessibilityValue(summary)
        .accessibilityHint("Shows local server details and opens the Serving and Agents console")
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
    let onShowServing: () -> Void
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
                onShowServing()
            } label: {
                Label("Open Serving & Agents", systemImage: "network")
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
