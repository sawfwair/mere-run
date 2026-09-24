import MereRunContract
import StudioKit
import SwiftUI

/// The task inspector's editor for the `--face-index` family (`.faceIndex`): which face of the
/// picture a command means, chosen by clicking it once a Face detection run has drawn boxes on
/// that picture. Until then the flag is a plain number field with a hint, so an image nobody has
/// detected faces in is still reachable. Compare shows one picker per picture.
struct StudioFaceIndexEditor: View {
    @Binding var draft: StudioTaskDraft
    /// The fields of the section being drawn; the one with the `.faceIndex` override names the
    /// flags this editor owns.
    let fields: [StudioContractField<StudioTaskDraft>]
    @EnvironmentObject private var library: StudioLibraryStore

    private var options: [MereRunCapabilityOption] {
        guard let field = fields.first(where: { $0.overrideID == .faceIndex }), let capability = draft.capability else {
            return []
        }
        return capability.options.filter { option in field.bindings.contains { $0.fieldID == option.flag } }
    }

    var body: some View {
        ForEach(options, id: \.flag) { option in
            row(option)
        }
    }

    @ViewBuilder
    private func row(_ option: MereRunCapabilityOption) -> some View {
        let imagePath = StudioFacePick.argumentIndex(forFlag: option.flag).map { draft.argument($0) } ?? ""
        if let document = StudioFacePick.detectionDocumentURL(for: imagePath, in: library.items) {
            VStack(alignment: .leading, spacing: 6) {
                StudioInspectorLabeledRow(option.label) {
                    Text("Face \(selection(for: option.flag).wrappedValue)")
                        .font(.callout)
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .accessibilityLabel("\(option.label): face \(selection(for: option.flag).wrappedValue)")
                }
                StudioFacePickerView(
                    imageURL: URL(fileURLWithPath: NSString(string: imagePath).expandingTildeInPath),
                    documentURL: document,
                    selection: selection(for: option.flag)
                )
                .frame(height: 200)
                .background(MereRunTheme.surfaceRaised)
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                        .strokeBorder(MereRunTheme.border, lineWidth: 1)
                }
                Text("Click a face to choose it.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .help(option.flag)
        } else {
            VStack(alignment: .leading, spacing: 4) {
                ContractFormControl(field: StudioContractField(option: option, bindings: [.flag(option.flag)]), draft: $draft)
                Text(imagePath.isBlank
                     ? "Attach the picture and run Detect faces on it to choose a face by clicking it."
                     : "Run Detect faces on this picture first to choose a face by clicking it.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// The flag as the face number the picker and the row read and write.
    private func selection(for flag: String) -> Binding<Int> {
        Binding(
            get: { Int(draft.text(flag)) ?? 0 },
            set: { draft.form[flag] = .integer($0) }
        )
    }
}
