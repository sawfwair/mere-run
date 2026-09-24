import MereRunContract
import StudioKit
import SwiftUI

/// The task inspector's editor for the `--face-index` family (`.faceIndex`): which face of the
/// picture a command means, chosen by clicking it once a Face detection run has drawn boxes on
/// that picture. Until then the flag is a stepper with a hint, so an image nobody has detected
/// faces in is still reachable. Compare shows one picker per picture. Faces are numbered from
/// one everywhere the user reads them; the flag counts from zero.
struct StudioFaceIndexEditor: View {
    @Binding var draft: StudioTaskDraft
    /// The fields of the section being drawn; the one with the `.faceIndex` override names the
    /// flags this editor owns.
    let fields: [StudioContractField<StudioTaskDraft>]
    @EnvironmentObject private var library: StudioLibraryStore

    /// The newest Detect run's document per flag, looked up when the picture or the Library
    /// changes rather than on every render.
    @State private var documents: [String: URL] = [:]

    private var options: [MereRunCapabilityOption] {
        guard let field = fields.first(where: { $0.overrideID == .faceIndex }), let capability = draft.capability else {
            return []
        }
        return capability.options.filter { option in field.bindings.contains { $0.fieldID == option.flag } }
    }

    private struct LookupKey: Equatable {
        let paths: [String]
        let libraryCount: Int
        let newest: Date?
    }

    private var lookupKey: LookupKey {
        LookupKey(
            paths: options.map { imagePath(for: $0) },
            libraryCount: library.items.count,
            newest: library.items.map(\.updatedAt).max()
        )
    }

    var body: some View {
        ForEach(options, id: \.flag) { option in
            row(option)
        }
        .task(id: lookupKey) {
            documents = Dictionary(uniqueKeysWithValues: options.compactMap { option in
                StudioFacePick.detectionDocumentURL(for: imagePath(for: option), in: library.items).map { (option.flag, $0) }
            })
        }
    }

    private func imagePath(for option: MereRunCapabilityOption) -> String {
        StudioFacePick.argumentIndex(forFlag: option.flag).map { draft.argument($0) } ?? ""
    }

    @ViewBuilder
    private func row(_ option: MereRunCapabilityOption) -> some View {
        let imagePath = imagePath(for: option)
        VStack(alignment: .leading, spacing: 6) {
            StudioInspectorLabeledRow(option.label) {
                Stepper(value: faceNumber(for: option.flag), in: 1...101) {
                    Text("Face \(faceNumber(for: option.flag).wrappedValue)")
                        .font(.callout)
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .monospacedDigit()
                }
                .accessibilityLabel(option.label)
                .accessibilityValue("Face \(faceNumber(for: option.flag).wrappedValue)")
            }
            .help(option.flag)
            if let document = documents[option.flag] {
                StudioFacePickerView(
                    imageURL: URL(fileURLWithPath: NSString(string: imagePath).expandingTildeInPath),
                    documentURL: document,
                    selection: faceIndex(for: option.flag)
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
            } else {
                Text(imagePath.isBlank
                     ? "Attach the picture and run Detect faces on it to choose a face by clicking it."
                     : "Run Detect faces on this picture first to choose a face by clicking it.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// The flag as the CLI counts it, from zero.
    private func faceIndex(for flag: String) -> Binding<Int> {
        Binding(
            get: { Int(draft.text(flag)) ?? 0 },
            set: { draft.form[flag] = .integer($0) }
        )
    }

    /// The same face as the user counts it, from one.
    private func faceNumber(for flag: String) -> Binding<Int> {
        Binding(
            get: { faceIndex(for: flag).wrappedValue + 1 },
            set: { faceIndex(for: flag).wrappedValue = max(0, $0 - 1) }
        )
    }
}
