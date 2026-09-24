import StudioKit
import SwiftUI

/// Image ▸ Datasets ▸ Discover's result: the CLI's summary, how much it scanned, and one row per
/// candidate folder with its counts, what is wrong with it, and "Train on it", which opens Image
/// ▸ Train pointed at the folder. Training is a Project handoff, which an Analyze next step
/// (input-first siblings only) cannot be, so each row offers it instead.
struct StudioDatasetCandidates: View {
    let document: StudioDatasetDiscoveryDocument
    let onTrain: (StudioDatasetDiscoveryDocument.Candidate) -> Void

    var body: some View {
        VStack(spacing: 0) {
            StudioResultNoteRow(text: document.summary)
            StudioResultMetricRow([
                ("Scanned", String(document.scannedDirectories)),
                ("Candidates", String(document.candidates.count)),
                ("Trainable", String(document.trainableCount)),
            ])
            StudioResultBoundedRows(count: document.candidates.count, threshold: 4, height: 420) {
                ForEach(document.candidates) { candidate in
                    candidateRow(candidate)
                }
            }
            ForEach(Array(document.diagnostics.enumerated()), id: \.offset) { _, diagnostic in
                MereBanner(severity: .warning, text: diagnostic)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
            }
        }
    }

    private func candidateRow(_ candidate: StudioDatasetDiscoveryDocument.Candidate) -> some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Image(systemName: Self.glyph(for: candidate))
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Self.tint(for: candidate))
                        .accessibilityHidden(true)
                    Text(candidate.name)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(MereRunTheme.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 8)
                    Text(Self.statusTitle(candidate.status))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(Self.tint(for: candidate))
                    Button("Train on it") { onTrain(candidate) }
                        .buttonStyle(.mereSecondary)
                        .controlSize(.small)
                        .disabled(!candidate.trainable)
                        .help(candidate.trainable
                              ? "Open Image ▸ Train with this folder as the dataset"
                              : "Fix the problems below before training on this folder")
                        .accessibilityLabel("Train on \(candidate.name)")
                }
                Text(StudioOutputLocation.abbreviate(URL(fileURLWithPath: candidate.path)))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Text(Self.counts(for: candidate))
                    .font(.system(size: 12))
                    .foregroundStyle(MereRunTheme.textSecondary)
                ForEach(candidate.problems, id: \.self) { problem in
                    Label(problem, systemImage: "exclamationmark.circle")
                        .font(.system(size: 11.5))
                        .foregroundStyle(MereRunTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .accessibilityElement(children: .contain)
            .accessibilityLabel("\(candidate.name), \(Self.statusTitle(candidate.status)), \(Self.counts(for: candidate))")
            StudioResultHairline()
        }
    }

    /// "12 images · 11 captions · 10 usable pairs"
    private static func counts(for candidate: StudioDatasetDiscoveryDocument.Candidate) -> String {
        [
            candidate.images == 1 ? "1 image" : "\(candidate.images) images",
            candidate.captions == 1 ? "1 caption" : "\(candidate.captions) captions",
            candidate.usablePairs == 1 ? "1 usable pair" : "\(candidate.usablePairs) usable pairs",
        ].joined(separator: " · ")
    }

    private static func statusTitle(_ status: String) -> String {
        switch status {
        case "ok": return "Ready"
        case "warning": return "With warnings"
        case "blocked": return "Blocked"
        default: return status.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    private static func glyph(for candidate: StudioDatasetDiscoveryDocument.Candidate) -> String {
        switch candidate.status {
        case "ok": return "checkmark.circle.fill"
        case "blocked": return "xmark.octagon.fill"
        default: return "exclamationmark.triangle.fill"
        }
    }

    private static func tint(for candidate: StudioDatasetDiscoveryDocument.Candidate) -> Color {
        switch candidate.status {
        case "ok": return MereRunTheme.green
        case "blocked": return MereRunTheme.red
        default: return MereRunTheme.yellow
        }
    }
}

/// "Train on it": Image ▸ Train's parked task draft gets the folder in its dataset well
/// (`StudioTrainingRun.attachDataset`), then the task opens on it.
@MainActor
enum StudioDatasetTrainingHandoff {
    static func open(_ candidate: StudioDatasetDiscoveryDocument.Candidate, navigation: NavigationModel, sessions: StudioTaskSessions?) {
        if let sessions {
            StudioTrainingRun.attachDataset(candidate.path, to: .imageTrain, sessions: sessions)
        }
        navigation.open(task: .imageTrain)
    }
}
