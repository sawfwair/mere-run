import StudioKit
import SwiftUI

/// Image ▸ Datasets ▸ Validate's result: which family and suite ran, and the artifacts the run
/// left in its folder — the rendered checks and their reports — each revealable in Finder.
struct StudioImageValidationArtifacts: View {
    let report: StudioImageValidationReport

    private var artifacts: [URL] { report.artifacts() }

    var body: some View {
        VStack(spacing: 0) {
            StudioResultFactRow(label: "Family", value: report.familyTitle, monospaced: false)
            StudioResultFactRow(label: "Suite", value: report.suite)
            StudioResultFileRow(url: report.artifactDirectory, detail: StudioOutputLocation.abbreviate(report.artifactDirectory))
            StudioResultCaptionRow(text: "Artifacts")
            if artifacts.isEmpty {
                StudioResultNoteRow(text: "No artifacts in the folder.")
            } else {
                StudioResultBoundedRows(count: artifacts.count) {
                    ForEach(artifacts, id: \.self) { url in
                        StudioResultFileRow(url: url)
                    }
                }
            }
        }
    }
}
