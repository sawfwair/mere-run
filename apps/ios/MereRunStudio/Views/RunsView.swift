import SwiftUI
import MereRunRelayKit

/// The run inbox: recent relay jobs, newest first.
struct RunsView: View {
    @EnvironmentObject private var relay: RelayStore
    @StateObject private var snapshots = RelaySnapshotStore<[WorkflowRemoteJob]>()
    @State private var path: [String] = []

    private var jobs: [WorkflowRemoteJob] { snapshots.value(for: relay.connectionID) ?? [] }
    private var errorMessage: String? { snapshots.errorMessage(for: relay.connectionID) }

    var body: some View {
        NavigationStack(path: $path) {
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
            .scrollContentBackground(.hidden)
            .background(MereTheme.background.ignoresSafeArea())
            .navigationTitle("Runs")
            .navigationDestination(for: String.self) { jobID in
                RunDetailView(jobID: jobID)
            }
            .refreshable { await load() }
            .task(id: relay.connectionID) { await load() }
        }
        .onChange(of: relay.connectionID) { path = [] }
    }

    private func load() async {
        guard let client = relay.client else { snapshots.reset(); return }
        let connectionID = relay.connectionID
        await snapshots.load(
            connectionID: connectionID,
            read: { try await client.list(limit: 50) },
            isCurrentConnection: { relay.connectionID == connectionID },
            presentError: { error in
                if let error = error as? RelayClientError { return AppErrorText.presentable(error.message) }
                return error.localizedDescription
            }
        )
    }
}
