import SwiftUI
import MereRunRelayKit

/// One run: live state, worker events with progress, cancel/retry, and
/// digest-verified artifact fetch once finished.
///
/// Events use the same snapshot-poll contract as `mere.run run watch`; the
/// relay serves the stream SSE-framed and `RelayEventText` normalizes either
/// framing, so this view can move to a held-open stream without changing the
/// parse path.
struct RunDetailView: View {
    let jobID: String

    @EnvironmentObject private var relay: RelayStore
    @State private var job: WorkflowRemoteJob?
    @State private var events: [GraphRunEvent] = []
    @State private var fetchedFiles: [URL] = []
    @State private var busy = false
    @State private var errorMessage: String?

    var body: some View {
        List {
            Section {
                HStack {
                    Text(jobID)
                        .font(.system(.subheadline, design: .monospaced))
                        .lineLimit(1)
                    Spacer()
                    MereStatusLabel(
                        text: job?.state.rawValue ?? "loading",
                        color: statusColor(job?.state.rawValue ?? "")
                    )
                }
                if let error = job?.error {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(MereTheme.failure)
                }
                if let placement = job?.placement, placement.eligibleNodes == 0 {
                    Text(placement.diagnostic ?? "No eligible nodes for this job.")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.caution)
                }
            }

            if let progress = latestProgress {
                Section("Progress") {
                    VStack(alignment: .leading, spacing: 6) {
                        if let fraction = progress.progress?.fraction {
                            ProgressView(value: fraction)
                        } else {
                            ProgressView()
                        }
                        Text(progressLine(progress))
                            .font(.footnote)
                            .foregroundStyle(MereTheme.textSecondary)
                    }
                }
            }

            Section("Events") {
                ForEach(events.suffix(30), id: \.sequence) { event in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text(event.type)
                                .font(.system(.footnote, design: .monospaced))
                                .foregroundStyle(MereTheme.textSecondary)
                            Spacer()
                            if let nodeID = event.nodeID {
                                Text(nodeID)
                                    .font(.footnote)
                                    .foregroundStyle(MereTheme.textMuted)
                            }
                        }
                        if let message = event.message, !message.isEmpty {
                            Text(message)
                                .font(.footnote)
                                .foregroundStyle(MereTheme.textPrimary)
                        }
                    }
                }
                if events.isEmpty {
                    Text("Waiting for events…")
                        .font(.footnote)
                        .foregroundStyle(MereTheme.textMuted)
                }
            }

            if !fetchedFiles.isEmpty {
                Section("Artifacts") {
                    ForEach(fetchedFiles, id: \.self) { file in
                        ShareLink(item: file) {
                            Label(file.lastPathComponent, systemImage: "square.and.arrow.up")
                        }
                    }
                }
            }

            if let errorMessage {
                Section {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(MereTheme.failure)
                }
            }
        }
        .navigationTitle("Run")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                if job?.state == .finished {
                    Button("Fetch") { Task { await fetchArtifacts() } }
                        .disabled(busy)
                } else if job?.state == .failed {
                    Button("Retry") { Task { await retry() } }
                        .disabled(busy)
                } else if isActive {
                    Button("Cancel", role: .destructive) { Task { await cancel() } }
                        .disabled(busy)
                }
            }
        }
        .task { await watch() }
    }

    private var isActive: Bool {
        guard let state = job?.state else { return false }
        return state != .finished && state != .failed && state != .cancelled
    }

    private var latestProgress: GraphRunEvent? {
        events.last { $0.progress != nil }
    }

    private func progressLine(_ event: GraphRunEvent) -> String {
        let phase = event.progress?.phase ?? event.type
        guard let current = event.progress?.current, let total = event.progress?.total else {
            return phase
        }
        let unit = event.progress?.unit.map { " \($0)" } ?? ""
        return "\(phase): \(Int(current))/\(Int(total))\(unit)"
    }

    private func watch() async {
        guard let client = relay.client else { return }
        while !Task.isCancelled {
            do {
                job = try await client.inspect(jobID: jobID)
                events = RelayEventText.decodedEvents(try await client.events(jobID: jobID))
                errorMessage = nil
            } catch let error as RelayClientError {
                errorMessage = error.message
            } catch {
                errorMessage = error.localizedDescription
            }
            if !isActive { return }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    private func cancel() async {
        guard let client = relay.client else { return }
        busy = true
        defer { busy = false }
        do {
            job = try await client.cancel(jobID: jobID)
        } catch let error as RelayClientError {
            errorMessage = error.message
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func retry() async {
        guard let client = relay.client else { return }
        busy = true
        defer { busy = false }
        do {
            job = try await client.retry(jobID: jobID)
        } catch let error as RelayClientError {
            errorMessage = error.message
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func fetchArtifacts() async {
        guard let client = relay.client else { return }
        busy = true
        defer { busy = false }
        let destination = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("runs", isDirectory: true)
            .appendingPathComponent(jobID, isDirectory: true)
        // Fetch validates a pre-existing destination as a materialized bundle,
        // which the phone never has; start each fetch from a clean directory.
        try? FileManager.default.removeItem(at: destination)
        do {
            let fetched = try await client.fetch(jobID: jobID, into: destination, allArtifacts: false)
            fetchedFiles = fetched.artifacts.compactMap { artifact in
                let url = destination.appendingPathComponent(artifact.path)
                return FileManager.default.fileExists(atPath: url.path) ? url : nil
            }
        } catch let error as RelayClientError {
            errorMessage = error.message
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
