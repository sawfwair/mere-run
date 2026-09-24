import AppKit
import StudioKit
import SwiftUI

/// Image ▸ Datasets ▸ Run plan's result: what the saved plan will do, read from the CLI's typed
/// preflight or materialize envelope and laid out as labelled facts instead of a path scrape.
/// Drawn as the result panel's rows: the panel's header already names the report, so the first
/// row is its status and summary, then each diagnostic, then the sections.
struct StudioRunPlanReportView: View {
    let report: StudioRunPlanReport

    var body: some View {
        VStack(spacing: 0) {
            statusRow
            ForEach(report.diagnostics) { diagnostic in
                MereBanner(severity: severity(diagnostic.severity), text: "\(diagnostic.title): \(diagnostic.message)")
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
            }
            ForEach(report.sections) { section in
                sectionRows(section)
            }
        }
    }

    private var statusRow: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: statusSymbol)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(statusColor)
                    .accessibilityHidden(true)
                Text(report.summary)
                    .font(.system(size: 13))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(report.title), \(statusLabel). \(report.summary)")
            StudioResultHairline()
        }
    }

    private func sectionRows(_ section: StudioRunPlanReport.Section) -> some View {
        VStack(spacing: 0) {
            StudioResultCaptionRow(text: section.title)
            ForEach(section.rows) { row in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    Text(row.label)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .frame(width: 118, alignment: .leading)
                    // A row whose value is its path shows the path shortened; a row with more
                    // to say ("Missing: …") shows its words as written.
                    Text(row.path == row.value ? StudioOutputLocation.abbreviate(URL(fileURLWithPath: row.value)) : row.value)
                        .font(.system(size: 12.5, weight: .medium, design: row.path == nil ? .default : .monospaced))
                        .foregroundStyle(MereRunTheme.textPrimary)
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
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .accessibilityElement(children: .combine)
            }
            StudioResultHairline()
                .padding(.top, 4)
        }
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
