import AVFoundation
import SwiftUI
import MereRunRelayKit
import UIKit

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
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var poller = RunSnapshotPoller()
    @State private var watchRevision = UUID()
    @State private var lastWatchTarget: WatchTarget?
    @State private var job: WorkflowRemoteJob?
    @State private var events: [GraphRunEvent] = []
    @State private var fetchedFiles: [URL] = []
    @State private var busy = false
    @State private var errorMessage: String?

    private struct WatchTarget: Equatable {
        let jobID: String
        let profile: WorkflowExecutorProfile?
        let connectionID: UUID
    }

    private struct WatchRequest: Equatable {
        let target: WatchTarget
        let revision: UUID
        let scenePhase: ScenePhase
    }

    private struct Snapshot {
        let job: WorkflowRemoteJob
        let events: [GraphRunEvent]

        var isTerminal: Bool {
            job.state == .finished || job.state == .failed || job.state == .cancelled
        }
    }

    private var watchTarget: WatchTarget {
        WatchTarget(jobID: jobID, profile: relay.profile, connectionID: relay.connectionID)
    }

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

            if isActive, let live = liveOutput {
                Section("Live output") {
                    Text(live)
                        .font(.body)
                        .foregroundStyle(MereTheme.textPrimary)
                        .textSelection(.enabled)
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
                        VStack(alignment: .leading, spacing: MereTheme.Spacing.s) {
                            if ["png", "jpg", "jpeg", "webp"].contains(file.pathExtension.lowercased()),
                               let image = UIImage(contentsOfFile: file.path) {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 320)
                                    .clipShape(RoundedRectangle(cornerRadius: MereTheme.Radius.panel))
                            }
                            if ["wav", "mp3", "m4a", "aiff"].contains(file.pathExtension.lowercased()) {
                                ArtifactAudioPlayer(url: file)
                            }
                            ShareLink(item: file) {
                                Label(file.lastPathComponent, systemImage: "square.and.arrow.up")
                            }
                        }
                    }
                }
            }

            if let errorMessage {
                Section {
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(MereTheme.failure)
                    if poller.phase == .paused {
                        Text("Updates paused after repeated connection failures.")
                            .font(.footnote)
                        Button("Resume updates") { watchRevision = UUID() }
                            .disabled(busy)
                    }
                }
            }
        }
        .scrollContentBackground(.hidden)
        .background(MereTheme.background.ignoresSafeArea())
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
        .task(id: WatchRequest(
            target: watchTarget,
            revision: watchRevision,
            scenePhase: scenePhase
        )) { await watch() }
    }

    private var isActive: Bool {
        guard let state = job?.state else { return false }
        return state != .finished && state != .failed && state != .cancelled
    }

    private var latestProgress: GraphRunEvent? {
        events.last { $0.progress != nil }
    }

    private var liveOutput: String? {
        guard let message = events.last(where: { $0.type == "node_output_delta" })?.message else {
            return nil
        }
        let visible = GeneratedTextFilters.strippingThinking(message, streaming: true)
        return visible.isEmpty ? nil : visible
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
        guard !Task.isCancelled else { return }
        poller.stop()
        let target = watchTarget
        if lastWatchTarget != target {
            job = nil
            events = []
            fetchedFiles = []
            errorMessage = nil
            lastWatchTarget = target
        }
        guard scenePhase == .active, !busy, let client = relay.client else { return }
        await poller.watch(
            read: {
                let current = try await client.inspect(jobID: jobID)
                try Task.checkCancellation()
                let currentEvents = RelayEventText.decodedEvents(try await client.events(jobID: jobID))
                return Snapshot(job: current, events: currentEvents)
            },
            isTerminal: { $0.isTerminal },
            receive: { snapshot in
                guard watchTarget == target else { return }
                job = snapshot.job
                events = snapshot.events
                errorMessage = nil
            },
            failed: { error in
                guard watchTarget == target else { return }
                if let error = error as? RelayClientError {
                    errorMessage = AppErrorText.presentable(error.message)
                } else {
                    errorMessage = error.localizedDescription
                }
            }
        )
    }

    private func cancel() async {
        guard !busy, let client = relay.client else { return }
        let target = watchTarget
        busy = true
        poller.stop()
        defer {
            busy = false
            watchRevision = UUID()
        }
        do {
            let current = try await client.cancel(jobID: jobID)
            guard watchTarget == target else { return }
            job = current
        } catch let error as RelayClientError {
            guard watchTarget == target else { return }
            errorMessage = AppErrorText.presentable(error.message)
        } catch {
            guard watchTarget == target else { return }
            errorMessage = error.localizedDescription
        }
    }

    private func retry() async {
        guard !busy, let client = relay.client else { return }
        let target = watchTarget
        busy = true
        poller.stop()
        defer {
            busy = false
            watchRevision = UUID()
        }
        do {
            let current = try await client.retry(jobID: jobID)
            guard watchTarget == target else { return }
            job = current
            events = []
            fetchedFiles = []
        } catch let error as RelayClientError {
            guard watchTarget == target else { return }
            errorMessage = AppErrorText.presentable(error.message)
        } catch {
            guard watchTarget == target else { return }
            errorMessage = error.localizedDescription
        }
    }

    private func fetchArtifacts() async {
        guard !busy, let client = relay.client else { return }
        let target = watchTarget
        busy = true
        defer {
            busy = false
            watchRevision = UUID()
        }
        let store = RunArtifactStore(
            runsRoot: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("runs", isDirectory: true)
        )
        do {
            let files = try await store.refresh(jobID: jobID) { staging in
                let fetched = try await client.fetch(jobID: jobID, into: staging, allArtifacts: false)
                return fetched.artifacts.map(\.path)
            }
            guard watchTarget == target else { return }
            fetchedFiles = files
        } catch let error as RelayClientError {
            guard watchTarget == target else { return }
            errorMessage = AppErrorText.presentable(error.message)
        } catch {
            guard watchTarget == target else { return }
            errorMessage = error.localizedDescription
        }
    }
}


/// Minimal transport for audio artifacts: play/pause and elapsed time.
struct ArtifactAudioPlayer: View {
    let url: URL

    @State private var player: AVAudioPlayer?
    @State private var playing = false

    var body: some View {
        HStack(spacing: MereTheme.Spacing.m) {
            Button {
                toggle()
            } label: {
                Image(systemName: playing ? "pause.circle.fill" : "play.circle.fill")
                    .font(.system(size: 34))
                    .foregroundStyle(MereTheme.accent)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(playing ? "Pause" : "Play")
            Text(url.lastPathComponent)
                .font(.footnote)
                .foregroundStyle(MereTheme.textSecondary)
                .lineLimit(1)
            Spacer()
        }
        .onDisappear {
            player?.stop()
        }
    }

    private func toggle() {
        if playing {
            player?.pause()
            playing = false
            return
        }
        if player == nil {
            try? AVAudioSession.sharedInstance().setCategory(.playback)
            try? AVAudioSession.sharedInstance().setActive(true)
            player = try? AVAudioPlayer(contentsOf: url)
        }
        player?.play()
        playing = true
    }
}
