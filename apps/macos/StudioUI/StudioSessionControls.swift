import StudioKit
import SwiftUI

// The chrome every Session surface shares, lifted from Music ▸ Realtime so Live listen and Live
// track draw the same transport row, badges, hairlines, and secondary buttons without touching
// Realtime's behaviour. `StudioSessionSurface` is the frame the live pages compose into:
// transport on top, the live panel in the middle, the log folded away underneath.

/// The 1pt rule between a session's rows.
struct StudioSessionHairline: View {
    var body: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.4))
            .frame(height: 1)
    }
}

/// `btnSecondary`: 26pt, raised fill, hairline border, 11.5pt medium.
struct StudioSessionSecondaryButton: View {
    let label: String
    let action: () -> Void
    @Environment(\.isEnabled) private var isEnabled

    init(_ label: String, action: @escaping () -> Void) {
        self.label = label
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(isEnabled ? MereRunTheme.textPrimary : MereRunTheme.textMuted)
                .lineLimit(1)
                .padding(.horizontal, 10)
                .frame(height: 26)
                .background {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(MereRunTheme.surfaceRaised)
                        .overlay {
                            RoundedRectangle(cornerRadius: 6)
                                .strokeBorder(MereRunTheme.border.opacity(0.6), lineWidth: 1)
                        }
                }
                .contentShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }
}

/// The LIVE pill: tinted fill, dot, 10.5pt bold tracked caps.
struct StudioSessionBadge: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text)
                .font(.system(size: 10.5, weight: .bold))
                .tracking(0.63)
        }
        .foregroundStyle(color)
        .padding(.horizontal, 7)
        .frame(height: 18)
        .background(Capsule().fill(color.opacity(0.13)))
        .accessibilityElement(children: .combine)
    }
}

/// Stroke-only stop square / play triangle on the 24-unit icon grid the mockup uses.
struct StudioSessionTransportGlyph: Shape {
    let stop: Bool

    func path(in rect: CGRect) -> Path {
        let unit = rect.width / 24
        var path = Path()
        if stop {
            path.addRoundedRect(
                in: CGRect(x: rect.minX + 6 * unit, y: rect.minY + 6 * unit, width: 12 * unit, height: 12 * unit),
                cornerSize: CGSize(width: 2 * unit, height: 2 * unit)
            )
        } else {
            path.move(to: CGPoint(x: rect.minX + 7 * unit, y: rect.minY + 5 * unit))
            path.addLine(to: CGPoint(x: rect.minX + 19 * unit, y: rect.minY + 12 * unit))
            path.addLine(to: CGPoint(x: rect.minX + 7 * unit, y: rect.minY + 19 * unit))
            path.closeSubpath()
        }
        return path
    }
}

/// The big Start/Stop circle at the head of a transport row: accent while idle, raised while a
/// session runs, with the stroke glyph inside.
struct StudioSessionTransportButton: View {
    let isRunning: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().fill(isRunning ? MereRunTheme.surfaceRaised : MereRunTheme.accent)
                StudioSessionTransportGlyph(stop: isRunning)
                    .stroke(isRunning ? MereRunTheme.textPrimary : MereRunTheme.onAccent, lineWidth: 1.8)
                    .frame(width: 24, height: 24)
            }
            .frame(width: 44, height: 44)
        }
        .buttonStyle(.plain)
        .help(isRunning ? "Stop" : "Start")
        .accessibilityLabel(isRunning ? "Stop session" : "Start session")
    }
}

/// Where a session is: idle, waiting for a lane, live, or over. Mirrors `StudioRealtimeJobBar.Phase`
/// so the badge and the job bar read the same words on every session page.
typealias StudioSessionPhase = StudioRealtimeJobBar.Phase

extension StudioSessionPhase {
    /// The badge the transport row shows, or nil while idle.
    @ViewBuilder
    var badge: some View {
        switch self {
        case .idle:
            EmptyView()
        case .queued:
            StudioSessionBadge(text: "QUEUED", color: MereRunTheme.yellow)
        case .live:
            StudioSessionBadge(text: "LIVE", color: MereRunTheme.red)
        case .ended(let exitCode):
            if let exitCode, exitCode != 0 {
                StudioSessionBadge(text: "FAILED", color: MereRunTheme.red)
            } else {
                StudioSessionBadge(text: "ENDED", color: MereRunTheme.textMuted)
            }
        }
    }
}

/// The Session shell: a transport row (Start/Stop, the phase badge, the elapsed clock, and the
/// page's own chips such as a device or camera picker), the live panel, and the session log
/// behind a disclosure. Live listen and Live track compose into it; Realtime keeps its own
/// richer layout and shares only the pieces above.
struct StudioSessionSurface<Chips: View, Live: View>: View {
    let phase: StudioSessionPhase
    /// "04:12" — the page's own clock, since what it counts differs per session.
    let clock: String
    let logLines: [String]
    let onToggle: () -> Void
    @ViewBuilder let chips: () -> Chips
    @ViewBuilder let live: () -> Live

    @State private var showsLog = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 14) {
                StudioSessionTransportButton(isRunning: phase == .live || phase == .queued, action: onToggle)
                Text(clock)
                    .font(.system(size: 22, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .accessibilityLabel("Elapsed \(clock)")
                phase.badge
                Spacer(minLength: 8)
                chips()
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 18)
            StudioSessionHairline()
            live()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            StudioSessionHairline()
            logDisclosure
        }
        .background(MereRunTheme.background)
    }

    private var logDisclosure: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button {
                    withAnimation(MereRunTheme.Motion.quick) { showsLog.toggle() }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: showsLog ? "chevron.down" : "chevron.right")
                            .font(.system(size: 9, weight: .semibold))
                        Text(showsLog ? "Hide log" : "Log")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(MereRunTheme.textMuted)
                }
                .buttonStyle(.plain)
                Spacer()
                if showsLog, !logLines.isEmpty {
                    StudioSessionSecondaryButton("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(StudioRealtimeSessionLog.copyText(logLines), forType: .string)
                    }
                }
            }
            if showsLog {
                ScrollView {
                    VStack(alignment: .leading, spacing: 3) {
                        ForEach(Array(logLines.suffix(200).enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(MereRunTheme.textMuted)
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 200)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
    }
}
