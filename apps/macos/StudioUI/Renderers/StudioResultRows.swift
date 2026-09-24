import AppKit
import StudioKit
import SwiftUI

// The rows a renderer contributes inside the Analyze result panel, drawn the way the panel draws
// its own: 14 pt side padding, a hairline under each row, 13 pt text. Shared by the renderers in
// this folder so a fact, a metric strip, and a file row look the same whichever document they
// come from.

/// One labelled fact: a muted label in a fixed column, the value beside it.
struct StudioResultFactRow: View {
    let label: String
    let value: String
    var monospaced = true

    init(label: String, value: String, monospaced: Bool = true) {
        self.label = label
        self.value = value
        self.monospaced = monospaced
    }

    init(_ label: String, _ value: String) {
        self.init(label: label, value: value)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .frame(width: 72, alignment: .leading)
                Text(value)
                    .font(.system(size: 12.5, weight: .medium, design: monospaced ? .monospaced : .default))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .textSelection(.enabled)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .accessibilityElement(children: .combine)
            StudioResultHairline()
        }
    }
}

/// Two or three numbers side by side — "Vectors 2 · Dimensions 1024 · Tokens 7" — as one row.
struct StudioResultMetricRow: View {
    struct Metric: Identifiable {
        let label: String
        let value: String
        var id: String { label }
    }

    let metrics: [Metric]

    init(_ metrics: [(String, String)]) {
        self.metrics = metrics.map { Metric(label: $0.0, value: $0.1) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                ForEach(metrics) { metric in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(metric.label)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(MereRunTheme.textMuted)
                        Text(metric.value)
                            .font(.system(size: 15, weight: .semibold, design: .rounded))
                            .foregroundStyle(MereRunTheme.textPrimary)
                            .monospacedDigit()
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("\(metric.label) \(metric.value)")
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            StudioResultHairline()
        }
    }
}

/// A file or folder the run wrote or read, with Reveal and — where the panel's own action row
/// does not offer it — Quick Look. The caller says which it is, so the row never asks the disk
/// while drawing; a glyph the file kind cannot say (a tensor file) is passed in.
struct StudioResultFileRow: View {
    let url: URL
    var detail: String?
    var isDirectory = false
    var glyph: String?
    var quickLook = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: isDirectory ? "folder" : glyph ?? Self.glyph(for: url))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(MereRunTheme.accent)
                    .frame(width: 14)
                VStack(alignment: .leading, spacing: 1) {
                    Text(url.lastPathComponent)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let detail {
                        Text(detail)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(MereRunTheme.textMuted)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(url.path)
                if quickLook {
                    Button("Quick Look") { QuickLookCoordinator.shared.preview(url) }
                        .buttonStyle(.mereSecondary)
                        .controlSize(.small)
                        .accessibilityLabel("Quick Look \(url.lastPathComponent)")
                }
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    .buttonStyle(.mereSecondary)
                    .controlSize(.small)
                    .accessibilityLabel("Reveal \(url.lastPathComponent) in Finder")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .accessibilityElement(children: .contain)
            .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
            StudioResultHairline()
        }
    }

    static func glyph(for url: URL) -> String {
        switch StudioOutputFileKind.classify(url) {
        case .audio: return "waveform"
        case .video: return "film"
        case .image: return "photo"
        case .model3D: return "cube.transparent"
        case .text: return "doc.text"
        case .other: return "doc"
        }
    }
}

/// A muted line where a list would be: "No artifacts in the folder."
struct StudioResultNoteRow: View {
    let text: String

    var body: some View {
        VStack(spacing: 0) {
            Text(text)
                .font(.system(size: 12.5))
                .foregroundStyle(MereRunTheme.textMuted)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
            StudioResultHairline()
        }
    }
}

/// A small uppercase caption between groups of rows, the way the run plan titles its sections.
struct StudioResultCaptionRow: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(MereRunTheme.textMuted)
            .textCase(.uppercase)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 14)
            .padding(.top, 12)
            .padding(.bottom, 4)
            .accessibilityAddTraits(.isHeader)
    }
}

struct StudioResultHairline: View {
    var body: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }
}

/// Rows size to their content; more than `threshold` of them scroll at `height` rather than
/// growing, the same bound the panel puts on its own lists.
struct StudioResultBoundedRows<Content: View>: View {
    let count: Int
    var threshold = 7
    var height: CGFloat = 320
    @ViewBuilder let content: () -> Content

    var body: some View {
        if count > threshold {
            ScrollView {
                VStack(spacing: 0) { content() }
            }
            .frame(height: height)
        } else {
            VStack(spacing: 0) { content() }
        }
    }
}
