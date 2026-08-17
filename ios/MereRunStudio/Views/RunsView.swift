import SwiftUI
import MereRunRelayKit

/// The run inbox: recent relay jobs, newest first.
struct RunsView: View {
    @EnvironmentObject private var relay: RelayStore
    @State private var jobs: [WorkflowRemoteJob] = []
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            List {
                ForEach(jobs, id: \.jobID) { job in
                    NavigationLink(value: job.jobID) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(job.jobID)
                                    .font(.system(.subheadline, design: .monospaced))
                                    .foregroundStyle(MereTheme.textPrimary)
                                    .lineLimit(1)
                                Spacer()
                                MereStatusLabel(
                                    text: job.state.rawValue,
                                    color: statusColor(job.state.rawValue)
                                )
                            }
                            if let updated = job.updatedAt {
                                Text(updated.formatted(.relative(presentation: .named)))
                                    .font(.footnote)
                                    .foregroundStyle(MereTheme.textMuted)
                            }
                        }
                    }
                }
                if jobs.isEmpty && errorMessage == nil {
                    Text("No runs yet. Submit a graph from the CLI or Studio and it appears here.")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.textMuted)
                }
                if let errorMessage {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(MereTheme.failure)
                }
            }
            .navigationTitle("Runs")
            .navigationDestination(for: String.self) { jobID in
                RunDetailView(jobID: jobID)
            }
            .refreshable { await load() }
            .task { await load() }
        }
    }

    private func load() async {
        guard let client = relay.client else { return }
        do {
            jobs = try await client.list(limit: 50)
            errorMessage = nil
        } catch let error as RelayClientError {
            errorMessage = error.message
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
