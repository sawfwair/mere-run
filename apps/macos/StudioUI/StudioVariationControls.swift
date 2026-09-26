import StudioKit
import SwiftUI

/// "Run variations" as a submenu, for a card's "…" menu and a Library row's context menu: the
/// same command 2, 4, or 8 times, each with its own recorded seed. The surface leaves it out when
/// the command takes no seed (`StudioVariations.applies`).
struct StudioVariationsMenuItems: View {
    let run: (StudioVariationCount) -> Void

    var body: some View {
        Menu {
            ForEach(StudioVariationCount.allCases) { count in
                Button(count.title) { run(count) }
            }
        } label: {
            Label("Run variations", systemImage: StudioVariationSymbols.variations)
        }
    }
}

/// The composer's variations control beside Run: a compact menu of counts. Run itself still runs
/// once; this submits the draft 2, 4, or 8 times with a random seed each.
struct StudioVariationsRunButton: View {
    let isEnabled: Bool
    var disabledReason: String?
    let run: (StudioVariationCount) -> Void

    var body: some View {
        Menu {
            Section("Run with a new seed each") {
                ForEach(StudioVariationCount.allCases) { count in
                    Button("Run \(count.title)") { run(count) }
                }
            }
        } label: {
            Image(systemName: StudioVariationSymbols.variations)
                .font(.system(size: 14, weight: .medium))
                .frame(width: 28, height: 28)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .foregroundStyle(isEnabled ? MereRunTheme.textSecondary : MereRunTheme.textMuted)
        .disabled(!isEnabled)
        .help(isEnabled ? "Run variations" : disabledReason ?? "Run variations")
        .accessibilityLabel("Run variations")
        .accessibilityHint(isEnabled ? "Runs this draft 2, 4, or 8 times, each with its own seed" : disabledReason ?? "")
    }
}

/// The feed card's Vary control where the command takes a seed: a click varies once, as the plain
/// icon did; the menu runs a group.
struct StudioVaryMenuButton: View {
    let vary: () -> Void
    let run: (StudioVariationCount) -> Void

    var body: some View {
        Menu {
            Button("Vary once") { vary() }
            Divider()
            ForEach(StudioVariationCount.allCases) { count in
                Button("Run \(count.title)") { run(count) }
            }
        } label: {
            Image(systemName: "shuffle")
                .font(.callout.weight(.medium))
        } primaryAction: {
            vary()
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .frame(height: 28)
        .foregroundStyle(MereRunTheme.textSecondary)
        .help("Vary with a new seed; hold for more variations")
        .accessibilityLabel("Vary with a new seed")
        .accessibilityHint("Open the menu to run 2, 4, or 8 variations")
    }
}

enum StudioVariationSymbols {
    static let variations = "square.grid.2x2"
    static let compare = "rectangle.split.2x1"
}
