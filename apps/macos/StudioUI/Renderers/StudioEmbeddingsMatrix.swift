import StudioKit
import SwiftUI

/// Text ▸ Embeddings' result: how many vectors of what size the run produced, the cosine
/// similarity of every pair when there is more than one, and each vector's norm with the first
/// of its values.
struct StudioEmbeddingsMatrix: View {
    let document: StudioEmbeddingDocument

    /// A matrix wider than this scrolls sideways; the labels stay legible at 360 pt.
    private static let cellWidth: CGFloat = 46
    private static let cellHeight: CGFloat = 24
    private static let shownValues = 8

    var body: some View {
        VStack(spacing: 0) {
            StudioResultMetricRow([
                ("Vectors", String(document.vectors.count)),
                ("Dimensions", String(document.dimensions)),
                ("Tokens", String(document.promptTokens)),
            ])
            if document.vectors.count > 1 {
                StudioResultCaptionRow(text: "Cosine similarity")
                matrix
                StudioResultHairline()
            }
            StudioResultBoundedRows(count: document.vectors.count) {
                ForEach(document.vectors) { vector in
                    vectorRow(vector)
                }
            }
        }
    }

    private var matrix: some View {
        ScrollView(.horizontal) {
            Grid(alignment: .center, horizontalSpacing: 4, verticalSpacing: 4) {
                GridRow {
                    Color.clear.frame(width: 28, height: Self.cellHeight)
                    ForEach(document.vectors) { vector in
                        label("#\(vector.id + 1)")
                    }
                }
                ForEach(document.vectors) { row in
                    GridRow {
                        label("#\(row.id + 1)")
                        ForEach(document.vectors) { column in
                            cell(document.cosineSimilarity(row, column), row: row.id + 1, column: column.id + 1)
                        }
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Cosine similarity matrix of \(document.vectors.count) vectors")
    }

    private func label(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundStyle(MereRunTheme.textMuted)
            .frame(width: text.count > 3 ? Self.cellWidth : 28, height: Self.cellHeight)
    }

    private func cell(_ similarity: Double, row: Int, column: Int) -> some View {
        Text(similarity.formatted(.number.precision(.fractionLength(2))))
            .font(.system(size: 11, weight: .medium, design: .monospaced))
            .foregroundStyle(MereRunTheme.textPrimary)
            .frame(width: Self.cellWidth, height: Self.cellHeight)
            .background {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Self.fill(for: similarity))
            }
            .help("#\(row) to #\(column): \(similarity.formatted(.number.precision(.fractionLength(3))))")
            .accessibilityLabel("#\(row) to #\(column), \(similarity.formatted(.number.precision(.fractionLength(3))))")
    }

    /// Close pairs read green, related ones the accent, the rest sit on the raised surface.
    private static func fill(for similarity: Double) -> Color {
        if similarity >= 0.9 { return MereRunTheme.green.opacity(0.22) }
        if similarity >= 0.6 { return MereRunTheme.accent.opacity(0.18) }
        return MereRunTheme.surfaceRaised
    }

    private func vectorRow(_ vector: StudioEmbeddingDocument.Vector) -> some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text("Vector \(vector.id + 1)")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                    Spacer(minLength: 8)
                    Text("L2 \(vector.norm.formatted(.number.precision(.fractionLength(4))))")
                        .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(MereRunTheme.textMuted)
                }
                Text(vector.values.prefix(Self.shownValues).map { String(format: "%.4f", $0) }.joined(separator: "  ") + (vector.values.count > Self.shownValues ? "  …" : ""))
                    .font(.system(size: 11, weight: .regular, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .textSelection(.enabled)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .accessibilityElement(children: .combine)
            StudioResultHairline()
        }
    }
}
