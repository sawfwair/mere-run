import AppKit
import StudioKit
import SwiftUI
import UniformTypeIdentifiers

/// The inspector's MiniMax-H3 controls: the adaptive-schedule override that replaces the step
/// slider, and the ordered Ref2VA reference list. The CLI itself aligns an H3 run's size and
/// frame count and fixes its frame rate, and the scope hides the inputs a family does not take,
/// so the draft is never rewritten here.
extension StudioInspector {
    /// MiniMax-H3 families take an adaptive schedule; they alone use `--h3-acceleration`.
    private var usesMiniMaxH3Schedule: Bool {
        guard let scope = scopeSource.scope(mode: .video, draft: draft), scope.family != nil else { return false }
        return scope.allows(CommandFlags.VideoGenerate.h3Acceleration)
    }

    @ViewBuilder
    var videoStepsControl: some View {
        if usesMiniMaxH3Schedule {
            VStack(alignment: .leading, spacing: 8) {
                Toggle(
                    "Override adaptive schedule",
                    isOn: Binding(get: { draft.h3Steps != nil }, set: { draft.h3Steps = $0 ? 21 : nil })
                )
                .toggleStyle(.checkbox)
                .font(.callout.weight(.medium))
                .foregroundStyle(MereRunTheme.textSecondary)
                if draft.h3Steps != nil {
                    StudioInspectorSlider(
                        label: "Schedule points",
                        value: Binding(get: { Double(draft.h3Steps ?? 21) }, set: { draft.h3Steps = Int($0.rounded()) }),
                        range: 1...64, step: 1, format: { String(Int($0)) }
                    )
                }
            }
        } else {
            stepsSlider(range: 1...60)
        }
    }

    @ViewBuilder
    var orderedReferences: some View {
        let references = draft.h3ReferenceInputs ?? []
        VStack(alignment: .leading, spacing: 6) {
            Text("Ordered Ref2VA references")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            ForEach(Array(references.enumerated()), id: \.offset) { index, reference in
                HStack(spacing: 5) {
                    Text(reference)
                        .font(MereRunTheme.captionFont)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button { moveH3Reference(index, by: -1) } label: { Image(systemName: "arrow.up") }
                        .disabled(index == 0)
                        .accessibilityLabel("Move up")
                    Button { moveH3Reference(index, by: 1) } label: { Image(systemName: "arrow.down") }
                        .disabled(index == references.count - 1)
                        .accessibilityLabel("Move down")
                    Button(role: .destructive) { removeH3Reference(index) } label: { Image(systemName: "trash") }
                        .accessibilityLabel("Remove reference")
                }
            }
            HStack {
                Button("Image") { chooseH3Reference(kind: "image", type: .image) }
                Button("Video") { chooseH3Reference(kind: "video", type: .movie) }
                Button("Audio") { chooseH3Reference(kind: "audio", type: .audio) }
            }
            .controlSize(.small)
        }
    }

    private func chooseH3Reference(kind: String, type: UTType) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [type]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        draft.h3ReferenceInputs = (draft.h3ReferenceInputs ?? []) + ["\(kind):\(url.path)"]
    }

    private func moveH3Reference(_ index: Int, by offset: Int) {
        var references = draft.h3ReferenceInputs ?? []
        references.swapAt(index, index + offset)
        draft.h3ReferenceInputs = references
    }

    private func removeH3Reference(_ index: Int) {
        var references = draft.h3ReferenceInputs ?? []
        references.remove(at: index)
        draft.h3ReferenceInputs = references
    }
}
