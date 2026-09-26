import StudioKit
import SwiftUI

/// The Library column's filter, in a popover beside the search field: the kind of output, the task
/// that made the run, the model it ran, and favorites. Tasks and models list only what the rows
/// in scope hold, so no choice ever leads to an empty column on its own.
struct StudioLibraryFilterPanel: View {
    @Binding var kind: StudioLibraryKind
    @Binding var favoritesOnly: Bool
    @Binding var task: StudioTask?
    @Binding var modelID: String?
    let taskOptions: [StudioLibraryFilterOption<StudioTask>]
    let modelOptions: [StudioLibraryFilterOption<String>]

    static let width: CGFloat = 264

    private var isFiltering: Bool {
        kind != .all || favoritesOnly || task != nil || modelID != nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.md) {
            HStack {
                Text("Filter")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Spacer(minLength: 0)
                Button("Clear") {
                    kind = .all
                    favoritesOnly = false
                    task = nil
                    modelID = nil
                }
                .buttonStyle(.borderless)
                .font(.caption.weight(.medium))
                .foregroundStyle(isFiltering ? MereRunTheme.accent : MereRunTheme.textMuted)
                .disabled(!isFiltering)
                .accessibilityLabel("Clear filters")
            }

            VStack(alignment: .leading, spacing: 4) {
                MereEyebrow("Kind")
                VStack(spacing: 0) {
                    ForEach(StudioLibraryKind.allCases) { option in
                        kindRow(option)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                MereEyebrow("Task")
                Picker("Task", selection: $task) {
                    Text("Any task").tag(StudioTask?.none)
                    if !taskOptions.isEmpty { Divider() }
                    ForEach(taskOptions) { option in
                        Text(option.title).tag(StudioTask?.some(option.value))
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityLabel("Task")
            }

            VStack(alignment: .leading, spacing: 6) {
                MereEyebrow("Model")
                Picker("Model", selection: $modelID) {
                    Text("Any model").tag(String?.none)
                    if !modelOptions.isEmpty { Divider() }
                    ForEach(modelOptions) { option in
                        Text(option.title).tag(String?.some(option.value))
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityLabel("Model")
            }

            Divider()

            HStack {
                Text("Favorites only")
                    .font(.callout)
                    .foregroundStyle(MereRunTheme.textPrimary)
                Spacer(minLength: 0)
                Toggle("Favorites only", isOn: $favoritesOnly)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)
            }
        }
        .padding(MereRunTheme.Spacing.md)
        .frame(width: Self.width, alignment: .leading)
    }

    private func kindRow(_ option: StudioLibraryKind) -> some View {
        Button {
            kind = option
        } label: {
            HStack(spacing: 8) {
                Image(systemName: option.systemImage)
                    .font(.caption.weight(.medium))
                    .frame(width: 16)
                    .foregroundStyle(kind == option ? MereRunTheme.accent : MereRunTheme.textMuted)
                Text(option.title)
                    .font(.callout)
                    .foregroundStyle(MereRunTheme.textPrimary)
                Spacer(minLength: 0)
                if kind == option {
                    Image(systemName: "checkmark")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(MereRunTheme.accent)
                }
            }
            .padding(.horizontal, 6)
            .frame(height: 24)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.sm)
                    .fill(kind == option ? MereRunTheme.accentSoft : .clear)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(option.title)
        .accessibilityAddTraits(kind == option ? [.isButton, .isSelected] : .isButton)
    }
}
