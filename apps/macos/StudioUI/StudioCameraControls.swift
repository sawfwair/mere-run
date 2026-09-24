import StudioKit
import SwiftUI
import UniformTypeIdentifiers

// The numeric pieces the two camera editors (`StudioGeometryCameraEditor`,
// `StudioInstantMeshCameraEditor`) are built from, kept apart so the Vision and 3D page PRs
// never touch the same file.

/// The section frame both editors share: the toggle, the cards, a Match views button when the
/// counts differ, the import and export menu, and the CLI's checks.
struct StudioCameraSection<Cards: View>: View {
    @Binding var enabled: Bool
    let count: Int
    let viewCount: Int
    let problems: [String]
    let onEnable: () -> Void
    let onMatchViews: () -> Void
    let onImport: () -> Void
    let onExport: () -> Void
    let canExport: Bool
    @ViewBuilder let cards: () -> Cards

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Toggle("Supply calibrated cameras", isOn: $enabled)
                    .onChange(of: enabled) { _, enabled in if enabled { onEnable() } }
                Spacer()
                Menu {
                    Button("Import camera file…", action: onImport)
                    Button("Export camera file…", action: onExport)
                        .disabled(!canExport)
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("Import and export a camera file")
                .accessibilityLabel("Camera file actions")
            }
            if enabled {
                Text("One camera per view, in view order. Without cameras the model estimates them.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
                cards()
                if count != viewCount {
                    Button {
                        onMatchViews()
                    } label: {
                        Label(
                            count < viewCount ? "Add cameras for the other views" : "Drop the extra cameras",
                            systemImage: count < viewCount ? "plus" : "minus"
                        )
                    }
                    .buttonStyle(.mereSecondary)
                    .controlSize(.small)
                }
                ForEach(problems, id: \.self) { problem in
                    Label(problem, systemImage: "exclamationmark.circle")
                        .font(MereRunTheme.captionFont)
                        .foregroundStyle(MereRunTheme.textSecondary)
                }
            }
        }
    }
}

struct StudioCameraCard<Fields: View>: View {
    let title: String
    let subtitle: String
    let problems: [String]
    let onRemove: () -> Void
    @ViewBuilder let fields: () -> Fields

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 6) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                Text(subtitle)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button(action: onRemove) {
                    Image(systemName: "minus.circle")
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(.mereIcon)
                .help("Remove")
                .accessibilityLabel("Remove \(title.lowercased())")
            }
            fields()
        }
        .padding(10)
        .background {
            RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                .fill(MereRunTheme.surface.opacity(0.72))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(
                            problems.isEmpty ? MereRunTheme.border.opacity(0.55) : MereRunTheme.yellow.opacity(0.6),
                            lineWidth: 1
                        )
                }
        }
    }
}

struct StudioNumberGroup<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    init(_ title: String, @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 10.5))
                .foregroundStyle(MereRunTheme.textMuted)
            HStack(spacing: 4) { content() }
        }
    }
}

/// A grid of number cells over a slice of a flat row-major array.
struct StudioMatrixGrid: View {
    @Binding var values: [Double]
    let columns: Int
    var range: Range<Int>?
    let label: String

    var body: some View {
        let indices = Array(range ?? values.indices)
        let rows = stride(from: 0, to: indices.count, by: columns).map { Array(indices[$0..<min($0 + columns, indices.count)]) }
        VStack(spacing: 4) {
            ForEach(rows, id: \.first) { row in
                HStack(spacing: 4) {
                    ForEach(row, id: \.self) { index in
                        StudioNumberCell(value: $values[index], label: "\(label) \(index + 1)")
                    }
                }
            }
        }
    }
}

struct StudioNumberCell: View {
    @Binding var value: Double
    let label: String

    var body: some View {
        TextField("", value: $value, format: .number.precision(.fractionLength(0...6)).grouping(.never))
            .textFieldStyle(.plain)
            .font(.system(size: 11, design: .monospaced))
            .multilineTextAlignment(.trailing)
            .padding(.horizontal, 5)
            .padding(.vertical, 4)
            .frame(width: 58)
            .merePanel()
            .accessibilityLabel(label)
    }
}

struct StudioIntegerCell: View {
    @Binding var value: Int
    let label: String

    var body: some View {
        TextField("", value: $value, format: .number.grouping(.never))
            .textFieldStyle(.plain)
            .font(.system(size: 11, design: .monospaced))
            .multilineTextAlignment(.trailing)
            .padding(.horizontal, 5)
            .padding(.vertical, 4)
            .frame(width: 58)
            .merePanel()
            .accessibilityLabel(label)
    }
}
