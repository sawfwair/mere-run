import StudioKit
import SwiftUI

/// Image ▸ Datasets ▸ Validate's result: which family and suite ran, and the artifacts the run
/// left in its folder — the rendered checks and their reports — each revealable in Finder. The
/// folder is listed once per report, not on every draw.
struct StudioImageValidationArtifacts: View {
    let report: StudioImageValidationReport

    @State private var artifacts: [StudioImageValidationReport.Artifact] = []

    var body: some View {
        VStack(spacing: 0) {
            StudioResultFactRow(label: "Family", value: report.familyTitle, monospaced: false)
            StudioResultFactRow(label: "Suite", value: report.suite)
            StudioResultFileRow(url: report.artifactDirectory, detail: StudioOutputLocation.abbreviate(report.artifactDirectory), isDirectory: true)
            StudioResultCaptionRow(text: "Artifacts")
            if artifacts.isEmpty {
                StudioResultNoteRow(text: "No artifacts in the folder.")
            } else {
                StudioResultBoundedRows(count: artifacts.count) {
                    ForEach(artifacts) { artifact in
                        StudioResultFileRow(url: artifact.url, isDirectory: artifact.isDirectory)
                    }
                }
            }
        }
        .task(id: report) {
            let listed = await Task.detached(priority: .userInitiated) { [report] in report.artifacts() }.value
            guard !Task.isCancelled else { return }
            artifacts = listed
        }
    }
}
