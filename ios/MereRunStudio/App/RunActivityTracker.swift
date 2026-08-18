#if canImport(ActivityKit)
import ActivityKit
import Foundation
import MereRunRelayKit

/// Starts a Live Activity for a submitted job and follows it to a terminal
/// state, updating progress from the worker's event stream. Updates are
/// app-driven for now; push-updated activities need APNs support in the
/// relay and are a documented follow-up.
enum RunActivityTracker {
    static func track(job: WorkflowRemoteJob, title: String, client: RelayWorkflowExecutor) {
        guard ActivityAuthorizationInfo().areActivitiesEnabled else { return }
        let attributes = RunActivityAttributes(jobID: job.jobID, title: title)
        let initial = RunActivityAttributes.ContentState(
            stateLabel: job.state.rawValue,
            fraction: nil,
            detail: "Queued on your fleet"
        )
        let jobID = job.jobID
        let startLabel = job.state.rawValue
        // The Activity handle is not Sendable; confining its whole lifetime to
        // one detached task keeps request, update, and end in a single region.
        Task.detached {
            guard let activity = try? Activity.request(
                attributes: attributes,
                content: .init(state: initial, staleDate: nil)
            ) else { return }
            var lastLabel = startLabel
            while true {
                try? await Task.sleep(for: .seconds(2))
                guard let current = try? await client.inspect(jobID: jobID) else { continue }
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
            }
        }
    }
}
#endif
