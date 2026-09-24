import StudioKit
import SwiftUI

/// What a tensor file holds, as the result panel's rows: an `.npy`'s dtype, shape, and layout,
/// or one row per safetensors tensor with its dtype and shape, then the writer's metadata.
struct StudioTensorInspector: View {
    let header: StudioTensorHeader

    private static let rowsMaxHeight: CGFloat = 320
    private static let scrollingRowThreshold = 7

    var body: some View {
        switch header {
        case .npy(let npy):
            VStack(spacing: 0) {
                fact("Type", npy.descriptor)
                fact("Shape", npy.shape)
                fact("Layout", npy.fortranOrder ? "Fortran order" : "C order")
                fact("Format", "NumPy \(npy.version)")
            }
        case .safetensors(let safetensors):
            let rows = safetensors.tensors.count + safetensors.metadata.count
            if rows > Self.scrollingRowThreshold {
                ScrollView {
                    safetensorsRows(safetensors)
                }
                .frame(height: Self.rowsMaxHeight)
            } else {
                safetensorsRows(safetensors)
            }
        }
    }

    private func safetensorsRows(_ safetensors: StudioSafetensorsHeader) -> some View {
        VStack(spacing: 0) {
            ForEach(safetensors.tensors) { tensor in
                HStack(spacing: 10) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(MereRunTheme.accent)
                        .frame(width: 10, height: 10)
                    Text(tensor.name)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text(tensor.summary)
                        .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textMuted)
                        .lineLimit(1)
                        .fixedSize()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(tensor.name), \(tensor.summary)")
                hairline
            }
            ForEach(safetensors.metadata.keys.sorted(), id: \.self) { key in
                fact(key, safetensors.metadata[key] ?? "")
            }
        }
    }

    /// One labelled fact, the way the run plan report lays its facts out.
    private func fact(_ label: String, _ value: String) -> some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .frame(width: 72, alignment: .leading)
                Text(value)
                    .font(.system(size: 12.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .accessibilityElement(children: .combine)
            hairline
        }
    }

    private var hairline: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }
}
