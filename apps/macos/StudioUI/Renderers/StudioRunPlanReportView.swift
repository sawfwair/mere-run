import AppKit
import StudioKit
import SwiftUI

/// Image ▸ Datasets ▸ Run plan's result: what the saved plan will do, read from the CLI's typed
/// preflight or materialize envelope and laid out as labelled facts instead of a path scrape.
struct StudioRunPlanReportView: View {
    let report: StudioRunPlanReport

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                ForEach(report.diagnostics) { diagnostic in
                    MereBanner(severity: severity(diagnostic.severity), text: "\(diagnostic.title): \(diagnostic.message)")
                }
                ForEach(report.sections) { section in
                    sectionView(section)
                }
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: statusSymbol)
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(statusColor)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(report.title)
                    .font(MereRunTheme.sectionFont)
                Text(report.summary)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(report.title), \(statusLabel). \(report.summary)")
    }

    private func sectionView(_ section: StudioRunPlanReport.Section) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(section.title)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(MereRunTheme.textMuted)
                .textCase(.uppercase)
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 7) {
                ForEach(section.rows) { row in
                    GridRow {
                        Text(row.label)
                            .font(MereRunTheme.captionFont)
                            .foregroundStyle(MereRunTheme.textMuted)
                            .frame(width: 168, alignment: .leading)
                            .gridColumnAlignment(.leading)
                        HStack(alignment: .firstTextBaseline, spacing: 6) {
                            // A row whose value is its path shows the path shortened; a row with
                            // more to say ("Missing: …") shows its words as written.
                            Text(row.path == row.value ? StudioOutputLocation.abbreviate(URL(fileURLWithPath: row.value)) : row.value)
                                .font(row.path == nil ? MereRunTheme.bodyFont : MereRunTheme.monoFont)
                                .textSelection(.enabled)
                                .lineLimit(row.path == nil ? 4 : 1)
                                .truncationMode(.middle)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            if let path = row.path {
                                Button {
                                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                                } label: {
                                    Image(systemName: "folder")
                                        .frame(width: 20, height: 20)
                                }
                                .buttonStyle(.mereIcon)
                                .help("Reveal in Finder")
                                .accessibilityLabel("Reveal \(row.label.lowercased()) in Finder")
                            }
                        }
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .merePanel()
    }

    private var statusSymbol: String {
        switch report.status {
        case "ok": return "checkmark.seal.fill"
        case "blocked": return "xmark.octagon.fill"
        default: return "exclamationmark.triangle.fill"
        }
    }

    private var statusColor: Color {
        switch report.status {
        case "ok": return MereRunTheme.green
        case "blocked": return MereRunTheme.red
        default: return MereRunTheme.yellow
        }
    }

    private var statusLabel: String {
        switch report.status {
        case "ok": return "ready"
        case "blocked": return "blocked"
        default: return "with warnings"
        }
    }

    private func severity(_ severity: StudioRunPlanReport.Severity) -> MereBanner.Severity {
        switch severity {
        case .blocker: return .error
        case .warning: return .warning
        case .note, .estimate, .unknown: return .info
        }
    }
}
