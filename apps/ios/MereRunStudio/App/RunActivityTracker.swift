#if canImport(ActivityKit)
import ActivityKit
import Foundation
import MereRunRelayKit

/// Starts a Live Activity for a submitted job and follows it to a terminal
/// state, updating progress from the worker's event stream. Updates are
/// app-driven for now; push-updated activities need APNs support in the
/// relay and are a documented follow-up.
@MainActor
enum RunActivityTracker {
    private static let pollingPolicy = RunPollingPolicy()
    private static var tasks: [String: Task<Void, Never>] = [:]
    private static var taskTokens: [String: UUID] = [:]

    static func track(job: WorkflowRemoteJob, title: String, client: RelayWorkflowExecutor) {
        guard ActivityAuthorizationInfo().areActivitiesEnabled,
              tasks[job.jobID] == nil else { return }
        let attributes = RunActivityAttributes(jobID: job.jobID, title: title)
        let initial = RunActivityAttributes.ContentState(
            stateLabel: job.state.rawValue,
            fraction: nil,
            detail: "Queued on your fleet"
        )
        let jobID = job.jobID
        let startLabel = job.state.rawValue
        let token = UUID()
        taskTokens[jobID] = token
        tasks[jobID] = Task { @MainActor in
            defer { clear(jobID: jobID, token: token) }
            guard let activity = try? Activity.request(
                attributes: attributes,
                content: .init(state: initial, staleDate: nil)
            ) else { return }
            var lastLabel = startLabel
            var consecutiveFailures = 0
            while !Task.isCancelled {
                let current: WorkflowRemoteJob
                do {
                    current = try await client.inspect(jobID: jobID)
                    consecutiveFailures = 0
                } catch {
                    consecutiveFailures += 1
                    if pollingPolicy.shouldStop(afterConsecutiveFailures: consecutiveFailures) {
                        let paused = RunActivityAttributes.ContentState(
                            stateLabel: lastLabel,
                            fraction: nil,
                            detail: "Updates paused — open mere.run"
                        )
                        await activity.end(
                            .init(state: paused, staleDate: nil),
                            dismissalPolicy: .default
                        )
                        return
                    }
                    guard await sleep(
                        nanoseconds: pollingPolicy.delayNanoseconds(
                            afterConsecutiveFailures: consecutiveFailures
                        )
                    ) else {
                        let stopped = RunActivityAttributes.ContentState(
                            stateLabel: lastLabel,
                            fraction: nil,
                            detail: "Updates stopped"
                        )
                        await activity.end(
                            .init(state: stopped, staleDate: nil),
                            dismissalPolicy: .default
                        )
                        return
                    }
                    continue
                }
                var fraction: Double?
                var detail: String?
                if let raw = try? await client.events(jobID: jobID) {
                    let events = RelayEventText.decodedEvents(raw)
                    if let progress = events.last(where: { $0.progress?.fraction != nil })?.progress {
                        fraction = progress.fraction
                        detail = progress.phase
                    }
                    if let delta = events.last(where: { $0.type == "node_output_delta" })?.message {
                        let visible = GeneratedTextFilters.strippingThinking(delta, streaming: true)
                        if !visible.isEmpty { detail = String(visible.suffix(80)) }
                    }
                }
                lastLabel = current.state.rawValue
                let state = RunActivityAttributes.ContentState(
                    stateLabel: lastLabel,
                    fraction: fraction,
                    detail: detail
                )
                if current.state == .finished || current.state == .failed || current.state == .cancelled {
                    let final = RunActivityAttributes.ContentState(
                        stateLabel: lastLabel,
                        fraction: current.state == .finished ? 1 : fraction,
                        detail: current.state == .finished ? "Done — open mere.run to fetch" : current.error
                    )
                    await activity.end(
                        .init(state: final, staleDate: nil),
                        dismissalPolicy: .after(.now + 12)
                    )
                    return
                }
                await activity.update(.init(state: state, staleDate: nil))
                guard await sleep(nanoseconds: pollingPolicy.regularDelayNanoseconds) else {
                    let stopped = RunActivityAttributes.ContentState(
                        stateLabel: lastLabel,
                        fraction: nil,
                        detail: "Updates stopped"
                    )
                    await activity.end(
                        .init(state: stopped, staleDate: nil),
                        dismissalPolicy: .default
                    )
                    return
                }
            }
            let stopped = RunActivityAttributes.ContentState(
                stateLabel: lastLabel,
                fraction: nil,
                detail: "Updates stopped"
            )
            await activity.end(
                .init(state: stopped, staleDate: nil),
                dismissalPolicy: .default
            )
        }
    }

    static func cancelAll() {
        let running = Array(tasks.values)
        tasks.removeAll()
        taskTokens.removeAll()
        for task in running {
            task.cancel()
        }
    }

    private static func sleep(nanoseconds: UInt64) async -> Bool {
        do {
            try await Task.sleep(nanoseconds: nanoseconds)
            return !Task.isCancelled
        } catch {
            return false
        }
    }

    private static func clear(jobID: String, token: UUID) {
        guard taskTokens[jobID] == token else { return }
        tasks[jobID] = nil
        taskTokens[jobID] = nil
    }
}
#endif
