import StudioKit
import SwiftUI

/// The `--dimensions` row of TESSERA's inspector: only the widths `geo tessera` accepts — the
/// student widths, 1024 for the teacher checkpoint, or blank to let the checkpoint choose — where
/// the contract's bare integer would take any number and fail at launch. A width the draft
/// already carries from elsewhere (an imported page draft, the Command view) is shown as itself
/// rather than passed off as the checkpoint's default.
struct StudioEarthDimensionsControl: View {
    @Binding var draft: StudioTaskDraft

    static let flag = "--dimensions"
    /// The accepted widths in picker order; the empty string is the checkpoint's own.
    static let choices = ["", "16", "32", "64", "128", "1024"]

    private var current: String { draft.text(Self.flag) }

    private var items: [String] {
        Self.choices.contains(current) ? Self.choices : Self.choices + [current]
    }

    private var selection: Binding<String> {
        Binding(
            get: { current },
            set: { draft.form[Self.flag] = Int($0).map(StudioContractValue.integer) ?? .unset }
        )
    }

    var body: some View {
        StudioInspectorLabeledRow("Dimensions") {
            Picker("Dimensions", selection: selection) {
                ForEach(items, id: \.self) { choice in
                    Text(Self.title(for: choice)).tag(choice)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 150)
        }
        .help(Self.flag)
        .accessibilityLabel("Dimensions")
        .accessibilityValue(Self.title(for: current))
    }

    static func title(for choice: String) -> String {
        switch choice {
        case "": return "Checkpoint default"
        case "1024": return "1024 (teacher)"
        default: return choice
        }
    }
}

/// OlmoEarth's sampling rows: the spatial patch size as the four values `geo olmoearth` accepts
/// (1, 2, 4, or 8 pixels) and the input's ground sample distance in metres, which the command
/// requires above zero — the contract declares both as bare numbers.
struct StudioEarthSamplingControl: View {
    @Binding var draft: StudioTaskDraft

    static let patchSizeFlag = "--patch-size"
    static let resolutionFlag = "--input-resolution"
    static let patchSizes = [1, 2, 4, 8]
    /// The smallest ground sample distance the field accepts; the command refuses zero.
    static let minimumResolution = 0.1

    private var patchSize: Binding<Int> {
        Binding(
            get: { Int(draft.text(Self.patchSizeFlag)) ?? 4 },
            set: { draft.form[Self.patchSizeFlag] = .integer($0) }
        )
    }

    private var resolution: Binding<Double> {
        Binding(
            get: { Double(draft.text(Self.resolutionFlag)) ?? 10 },
            set: { draft.form[Self.resolutionFlag] = .number(max(Self.minimumResolution, $0)) }
        )
    }

    var body: some View {
        // The four segments under their label, the way the slider rows stack: beside it they
        // would be squeezed narrower than their digits.
        VStack(alignment: .leading, spacing: 6) {
            Text("Patch size")
                .font(.callout.weight(.medium))
                .foregroundStyle(MereRunTheme.textSecondary)
            MereSegmentedControl(Self.patchSizes, selection: patchSize, accessibilityLabel: "Patch size") { "\($0) px" }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(Self.patchSizeFlag)
        StudioInspectorLabeledRow("Input resolution") {
            HStack(spacing: 6) {
                TextField("Metres", value: resolution, format: .number)
                    .textFieldStyle(.plain)
                    .multilineTextAlignment(.trailing)
                    .font(.system(size: 12.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .padding(.horizontal, 8)
                    .frame(width: 72, height: 28)
                    .background { StudioInspectorFieldChrome() }
                    .accessibilityLabel("Input resolution in metres")
                Text("m")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            }
        }
        .help(Self.resolutionFlag)
    }
}
