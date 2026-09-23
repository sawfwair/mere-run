import AppKit
import StudioKit
import SwiftUI

/// The menu bar extra: the local API server at a glance with its one control, the models it holds,
/// the Studio work in flight, and the way back into the Studio. It reads the same
/// `StudioLocalServer` the Server page drives, so the two never disagree, and it works with no
/// Studio window open.
package enum StudioMenuBar {
    /// Whether the extra is in the menu bar. On by default; Settings ▸ Server turns it off, and so
    /// does ⌘-dragging it out of the menu bar.
    package static let visibilityDefaultsKey = "mererun.app.showsMenuBarExtra"
}

/// The menu bar glyph: the app icon's Caveat "m." with the period as a power light — filled while
/// a server answers, hollow while none does. A template image, so the menu bar tints it for its
/// appearance.
package enum StudioMenuBarIcon {
    private static let serving = draw(isServing: true)
    private static let idle = draw(isServing: false)

    package static func image(isServing: Bool) -> NSImage {
        isServing ? serving : idle
    }

    private static func draw(isServing: Bool) -> NSImage {
        let image = NSImage(size: NSSize(width: 22, height: 16), flipped: false) { _ in
            let font = glyphFont(size: 22)
            let glyph = NSAttributedString(string: "m", attributes: [.font: font, .foregroundColor: NSColor.black])
            let baseline: CGFloat = 3.5
            glyph.draw(at: NSPoint(x: 0, y: baseline + font.descender))
            // Caveat's "m" ends in an exit stroke past its advance; the period sits clear of it.
            let dot = NSRect(x: glyph.size().width + 1.4, y: baseline - 0.1, width: 4.4, height: 4.4)
            NSColor.black.set()
            if isServing {
                NSBezierPath(ovalIn: dot).fill()
            } else {
                let ring = NSBezierPath(ovalIn: dot.insetBy(dx: 0.55, dy: 0.55))
                ring.lineWidth = 1.1
                ring.stroke()
            }
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = isServing ? "mere.run, server running" : "mere.run"
        return image
    }

    /// Caveat SemiBold, the face of the icon's "m.", or the system's rounded face without it.
    private static func glyphFont(size: CGFloat) -> NSFont {
        let caveat = NSFontDescriptor(fontAttributes: [
            .family: MereRunTheme.Brand.familyName,
            .traits: [NSFontDescriptor.TraitKey.weight: NSFont.Weight.semibold],
        ])
        if MereRunTheme.Brand.isAvailable, let font = NSFont(descriptor: caveat, size: size) {
            return font
        }
        let system = NSFont.systemFont(ofSize: size * 0.7, weight: .semibold)
        return system.fontDescriptor.withDesign(.rounded).flatMap { NSFont(descriptor: $0, size: size * 0.7) } ?? system
    }
}

/// The menu bar extra's label: the glyph, lit while a server answers.
package struct StudioMenuBarLabel: View {
    @ObservedObject private var server: StudioLocalServer

    package init(server: StudioLocalServer) {
        _server = ObservedObject(wrappedValue: server)
    }

    package var body: some View {
        Image(nsImage: StudioMenuBarIcon.image(isServing: server.phase.isServing))
            .accessibilityLabel(server.phase.isServing ? "mere.run, server running" : "mere.run")
    }
}

/// The panel the menu bar extra opens.
package struct StudioMenuBarPanel: View {
    @ObservedObject private var server: StudioLocalServer
    @ObservedObject private var monitor: StudioServingMonitor
    @ObservedObject private var jobs: JobStore
    private let controller: MereRunController
    private let onOpenStudio: () -> Void
    private let onOpenServer: () -> Void

    /// Bumped on every job event: lane membership is not itself published.
    @State private var generation = 0
    @State private var message: String?
    @State private var unloadingModelID: String?

    package static let width: CGFloat = StudioActivityPopover.width

    package init(
        controller: MereRunController,
        onOpenStudio: @escaping () -> Void,
        onOpenServer: @escaping () -> Void
    ) {
        self.controller = controller
        _server = ObservedObject(wrappedValue: controller.localServer)
        _monitor = ObservedObject(wrappedValue: controller.servingMonitor)
        _jobs = ObservedObject(wrappedValue: controller.jobs)
        self.onOpenStudio = onOpenStudio
        self.onOpenServer = onOpenServer
    }

    package var body: some View {
        _ = generation
        let rows = StudioActivity.rows(in: jobs)
        return VStack(alignment: .leading, spacing: 0) {
            header
            serverRow
            StudioMenuBarResidentServerRow(title: "Vision server", server: controller.visionServer)
            StudioMenuBarResidentServerRow(title: "Music server", server: controller.musicServer)
            if server.phase.isServing {
                models
            }
            if !rows.isEmpty {
                section("Activity", trailing: StudioActivity.summary(rows))
                ForEach(rows) { row in
                    if let job = jobs.job(row.id) {
                        StudioActivityJobRow(job: job, row: row) { jobs.cancel(row.id) }
                    }
                }
            }
            divider
            menuRow("Open Studio", action: onOpenStudio)
            menuRow("Server Settings…", action: onOpenServer)
            divider
            menuRow("Quit mere.run", shortcut: "⌘Q") { NSApp.terminate(nil) }
                .keyboardShortcut("q")
        }
        .padding(.vertical, 6)
        .frame(width: Self.width)
        .background(MereRunTheme.surface)
        .foregroundStyle(MereRunTheme.textPrimary)
        .onReceive(jobs.events) { _ in generation &+= 1 }
        .onChange(of: server.phase) { _, _ in message = nil }
    }

    // MARK: Header and server

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            (Text("mere").foregroundStyle(MereRunTheme.textPrimary)
                + Text(".").foregroundStyle(MereRunTheme.wordmarkGreen))
                .font(MereRunTheme.Brand.font(size: 22))
                .accessibilityLabel("mere.run")
            Spacer(minLength: 12)
            if let version = controller.cliVersion {
                Text("CLI \(version)")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 2)
        .padding(.bottom, 4)
    }

    private var serverRow: some View {
        HStack(alignment: .top, spacing: 10) {
            StudioMenuBarStatusDot(phase: server.phase)
                .padding(.top, 5)
            VStack(alignment: .leading, spacing: 3) {
                Text("API server")
                    .font(.callout.weight(.semibold))
                Text(StudioMenuBarCopy.stateLine(phase: server.phase, endpoint: server.endpoint))
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let detail = StudioMenuBarCopy.detail(
                    phase: server.phase,
                    safety: server.safety,
                    runtime: monitor.runtime,
                    connection: monitor.isReachable ? nil : monitor.connectionDetail
                ) {
                    Text(detail)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(isFailure ? MereRunTheme.red : MereRunTheme.textMuted)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                if let message {
                    Text(message)
                        .font(.caption.weight(.medium))
                        .foregroundStyle(MereRunTheme.red)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            serverControl
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .accessibilityElement(children: .contain)
    }

    private var isFailure: Bool {
        if case .failed = server.phase { return true }
        return false
    }

    @ViewBuilder
    private var serverControl: some View {
        switch server.phase {
        case .stopped, .failed:
            Button("Start") { message = server.start() }
                .buttonStyle(.merePrimary)
                .disabled(server.safety == .exposedWithoutAuthentication)
                .help("Start mere.run api serve on \(server.endpoint)")
        case .starting, .running:
            HStack(spacing: 2) {
                if server.phase == .running {
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(server.endpoint, forType: .string)
                    } label: {
                        Image(systemName: "doc.on.doc")
                            .font(.callout.weight(.medium))
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.mereIcon)
                    .help("Copy \(server.endpoint)")
                    .accessibilityLabel("Copy endpoint")
                }
                Button("Stop") {
                    message = nil
                    server.stop()
                }
                .buttonStyle(.mereSecondary)
                .help("Stop the server Studio started")
            }
        case .stopping:
            ProgressView()
                .controlSize(.small)
                .frame(height: 26)
        case .external:
            Button {
                Task { await monitor.refreshRuntimeNow(controller: controller) }
            } label: {
                Image(systemName: "arrow.clockwise")
                    .font(.callout.weight(.medium))
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.mereIcon)
            .help("Check the endpoint again")
            .accessibilityLabel("Refresh")
        }
    }

    // MARK: Models

    @ViewBuilder
    private var models: some View {
        let runtime = monitor.runtime
        section("Loaded", trailing: runtime.flatMap { StudioMenuBarCopy.memoryLine($0.memory) })
        let textModels = runtime?.loadedTextModels ?? []
        let sidecars = runtime?.loadedSidecars ?? []
        if textModels.isEmpty && sidecars.isEmpty {
            Text(runtime == nil ? "Waiting for the model pool…" : "No models loaded. They load on first request.")
                .font(.caption.weight(.medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
        }
        ForEach(textModels) { model in
            modelRow(
                title: model.id,
                state: model.state,
                unload: model.activeRequests > 0 || unloadingModelID != nil
                    ? nil
                    : { unload(model.id) }
            )
        }
        ForEach(sidecars) { sidecar in
            modelRow(title: "\(sidecar.kind.capitalized) · \(sidecar.displayModel)", state: sidecar.state, unload: nil)
        }
    }

    private func modelRow(title: String, state: String, unload: (() -> Void)?) -> some View {
        HStack(spacing: 10) {
            Circle()
                .fill(StudioMenuBarCopy.stateColor(state))
                .frame(width: 6, height: 6)
                .padding(.horizontal, 1)
            Text(title)
                .font(.caption.weight(.medium))
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 8)
            Text(state)
                .font(.caption.weight(.medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Button {
                unload?()
            } label: {
                Image(systemName: "eject")
                    .font(.caption.weight(.medium))
                    .frame(width: 22, height: 22)
            }
            .buttonStyle(.mereIcon)
            .disabled(unload == nil)
            .opacity(unload == nil ? 0 : 1)
            .help("Unload \(title)")
            .accessibilityLabel("Unload \(title)")
            .accessibilityHidden(unload == nil)
        }
        .padding(.leading, 14)
        .padding(.trailing, 10)
        .frame(minHeight: 26)
        .accessibilityElement(children: .contain)
    }

    private func unload(_ modelID: String) {
        unloadingModelID = modelID
        Task { @MainActor in
            message = await monitor.setModel(modelID, loaded: false, controller: controller)
            unloadingModelID = nil
        }
    }

    // MARK: Chrome

    private func section(_ title: String, trailing: String?) -> some View {
        HStack(spacing: 8) {
            Text(title.uppercased())
                .font(.caption2.weight(.semibold))
                .tracking(0.6)
                .foregroundStyle(MereRunTheme.textMuted)
            Spacer(minLength: 8)
            if let trailing {
                Text(trailing)
                    .font(.caption.weight(.medium))
                    .monospacedDigit()
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 4)
        .accessibilityAddTraits(.isHeader)
    }

    private var divider: some View {
        Divider()
            .overlay(MereRunTheme.border.opacity(0.4))
            .padding(.vertical, 5)
    }

    private func menuRow(_ title: String, shortcut: String? = nil, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                Text(title)
                Spacer(minLength: 12)
                if let shortcut {
                    Text(shortcut)
                        .foregroundStyle(MereRunTheme.textMuted)
                }
            }
        }
        .buttonStyle(StudioMenuBarRowStyle())
    }
}

/// A resident server Studio started — `vision serve`, `music serve` — while it runs: its state and
/// Stop, so the menu bar can end anything Studio left running. Nothing while it is stopped.
private struct StudioMenuBarResidentServerRow: View {
    let title: String
    @ObservedObject var server: StudioServiceProcess

    var body: some View {
        if server.state.isRunning {
            HStack(spacing: 10) {
                Circle()
                    .fill(StudioResidentServerCopy.color(server.state))
                    .frame(width: 8, height: 8)
                    .accessibilityHidden(true)
                Text(title)
                    .font(.callout.weight(.semibold))
                Text(StudioResidentServerCopy.title(server.state))
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                Spacer(minLength: 8)
                Button("Stop") { server.stop() }
                    .buttonStyle(.mereSecondary)
                    .disabled(server.state.isStopping)
                    .help("Stop the \(title.lowercased()) Studio started")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .accessibilityElement(children: .contain)
        }
    }
}

/// A menu item drawn in the panel: full width, highlighted under the pointer.
private struct StudioMenuBarRowStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        RowBody(configuration: configuration)
    }

    private struct RowBody: View {
        let configuration: Configuration
        @State private var hovering = false

        var body: some View {
            configuration.label
                .font(.callout)
                .padding(.horizontal, 9)
                .frame(maxWidth: .infinity, minHeight: 24, alignment: .leading)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                        .fill(hovering || configuration.isPressed ? MereRunTheme.hoverFill : .clear)
                }
                .contentShape(Rectangle())
                .padding(.horizontal, 5)
                .onHover { hovering = $0 }
        }
    }
}

/// The server's state as a dot: green while a server answers, amber while Studio's is starting or
/// stopping, red when it failed, hollow when nothing is serving.
private struct StudioMenuBarStatusDot: View {
    let phase: StudioLocalServer.Phase

    var body: some View {
        Group {
            switch phase {
            case .running, .external:
                Circle().fill(MereRunTheme.green)
            case .starting, .stopping:
                Circle().fill(MereRunTheme.yellow)
            case .failed:
                Circle().fill(MereRunTheme.red)
            case .stopped:
                Circle().strokeBorder(MereRunTheme.textMuted, lineWidth: 1.5)
            }
        }
        .frame(width: 8, height: 8)
        .accessibilityHidden(true)
    }
}

/// Every string the panel shows about the server, as pure functions of its state, so each one is
/// testable without a view.
package enum StudioMenuBarCopy {
    /// "Running · 127.0.0.1:8080": the phase and where it serves.
    package static func stateLine(phase: StudioLocalServer.Phase, endpoint: String) -> String {
        let address = endpoint.replacingOccurrences(of: "http://", with: "")
        return "\(phase.title) · \(address)"
    }

    /// The line under the state, when there is something to say: why it stopped, who owns it,
    /// what it is doing, or why it cannot start.
    package static func detail(
        phase: StudioLocalServer.Phase,
        safety: StudioServingSafety,
        runtime: StudioRuntimeSnapshot?,
        connection: String?
    ) -> String? {
        switch phase {
        case .failed(let message):
            return message
        case .stopped:
            return safety == .exposedWithoutAuthentication
                ? "Add an API key in Settings before serving beyond this Mac."
                : nil
        case .external:
            if let connection { return connection }
            let started = "Stop it where it was started."
            return trafficLine(runtime).map { "\($0)\n\(started)" } ?? started
        case .running:
            if let connection { return connection }
            return trafficLine(runtime)
        case .starting, .stopping:
            return nil
        }
    }

    /// "2 active · 1 queued · up 2h 15m", or nil when the runtime reports nothing worth a line.
    package static func trafficLine(_ runtime: StudioRuntimeSnapshot?) -> String? {
        guard let runtime else { return nil }
        var parts: [String] = []
        let active = runtime.admission?.activeRequests ?? runtime.activeRequests ?? 0
        let queued = runtime.admission?.queuedRequests ?? 0
        if active > 0 { parts.append("\(active) active") }
        if queued > 0 { parts.append("\(queued) queued") }
        if let uptime = runtime.process?.uptimeSeconds {
            parts.append("up \(Self.uptime(uptime))")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// "45m", "2h 15m", "3d 4h": the two largest units, which is all a glance needs.
    package static func uptime(_ seconds: Double) -> String {
        let minutes = Int(seconds) / 60
        let (days, hours) = (minutes / 1_440, minutes / 60 % 24)
        if days > 0 { return hours > 0 ? "\(days)d \(hours)h" : "\(days)d" }
        if minutes >= 60 { return minutes % 60 > 0 ? "\(minutes / 60)h \(minutes % 60)m" : "\(minutes / 60)h" }
        return "\(max(minutes, 1))m"
    }

    /// "12.4 GB in use", from the runtime's own footprint.
    package static func memoryLine(_ memory: StudioRuntimeMemory?) -> String? {
        guard let bytes = memory?.currentBytes else { return nil }
        return "\(StudioServingFormat.bytes(bytes)) in use"
    }

    static func stateColor(_ state: String) -> Color {
        switch state {
        case "Ready": return MereRunTheme.green
        case "Active", "Loading", "Queued": return MereRunTheme.yellow
        default: return MereRunTheme.textMuted
        }
    }
}
