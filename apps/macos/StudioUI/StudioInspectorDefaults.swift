import StudioKit
import SwiftUI

/// What an inspector's Defaults section acts on: where the page's defaults stand for the draft
/// it shows, and the two writes, each one undo step (`StudioPageDefaults`).
struct StudioInspectorPageDefaults {
    let status: StudioPageDefaultsStatus
    let save: () -> Void
    let restore: () -> Void
}

/// The last section of an inspector: "Save as my defaults" keeps the page's settings as where
/// its new drafts and Reset start, and "Restore app defaults" forgets them and puts the settings
/// back to the app's. Neither touches the inputs, the prompt, the seed, the model, or where the
/// run saves.
struct StudioInspectorDefaultsSection: View {
    let defaults: StudioInspectorPageDefaults

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Defaults")
                .font(.callout.weight(.semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
                .accessibilityAddTraits(.isHeader)
            Text(defaults.status.hasSaved
                 ? "New drafts and Reset start from your saved settings."
                 : "New drafts and Reset start from the app’s settings.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .fixedSize(horizontal: false, vertical: true)
            // Side by side when the column fits both, stacked when a larger text size does not.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) { buttons }
                VStack(alignment: .leading, spacing: 8) { buttons }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Defaults")
    }

    @ViewBuilder
    private var buttons: some View {
        Button("Save as my defaults", action: defaults.save)
            .buttonStyle(.mereSecondary)
            .disabled(!defaults.status.canSave)
            .help(defaults.status.canSave
                  ? "Start new drafts from these settings. Inputs, prompt, seed, model, and output stay with each draft."
                  : "These settings are already where new drafts start")
        Button("Restore app defaults", action: defaults.restore)
            .buttonStyle(.mereSecondary)
            .disabled(!defaults.status.canRestore)
            .help(defaults.status.canRestore
                  ? "Forget your saved defaults and put these settings back to the app’s"
                  : "These settings are already the app’s")
    }
}
