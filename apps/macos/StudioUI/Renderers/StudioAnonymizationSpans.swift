import StudioKit
import SwiftUI

/// Text ▸ Anonymize's Spans view: each input beside its protected text, with a capsule for
/// every span the privacy filter marked.
struct StudioAnonymizationSpans: View {
    let document: StudioAnonymizationDocument

    var body: some View {
        VStack(spacing: 0) {
            StudioResultMetricRow([
                ("Documents", String(document.results.count)),
                ("PII spans", String(document.spanCount)),
                ("Tokens", String(document.tokenCount)),
            ])
            StudioResultBoundedRows(count: document.results.count, threshold: 3, height: 420) {
                ForEach(document.results) { result in
                    resultRows(result)
                }
            }
        }
    }

    private func resultRows(_ result: StudioAnonymizationDocument.Result) -> some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                passage("Original", result.text, color: MereRunTheme.textSecondary)
                passage("Protected", result.anonymizedText, color: MereRunTheme.green)
                if !result.spans.isEmpty {
                    FlowLayout(spacing: 6, lineSpacing: 6) {
                        ForEach(result.spans) { span in
                            spanCapsule(span)
                        }
                    }
                    .accessibilityElement(children: .contain)
                    .accessibilityLabel("\(result.spans.count) spans")
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            StudioResultHairline()
        }
    }

    private func passage(_ label: String, _ text: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(text)
                .font(.system(size: 13))
                .foregroundStyle(color)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(label): \(text)")
    }

    private func spanCapsule(_ span: StudioAnonymizationDocument.Span) -> some View {
        HStack(spacing: 4) {
            Text(span.label)
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.yellow)
            Text(span.text)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background {
            Capsule().fill(MereRunTheme.yellow.opacity(0.14))
        }
        .help("Tokens \(span.startToken)–\(span.endToken)")
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(span.label) \(span.text), tokens \(span.startToken) to \(span.endToken)")
    }
}

/// Text ▸ Anonymize's Text view: the protected text alone, one passage per input, the way the
/// command prints it without `--json`.
struct StudioAnonymizedText: View {
    let document: StudioAnonymizationDocument

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                Text(document.protectedText)
                    .font(.system(size: 13))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
            }
            .frame(maxHeight: 320)
            StudioResultHairline()
        }
    }
}
