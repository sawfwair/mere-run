import Combine
import Foundation

/// The local API server — `mere.run api serve` — with one owner for the life of the app: the
/// phase it is in, whether Studio started it, and the start, stop, and restart that the Server
/// page and the menu bar extra both drive. Closing a window takes none of it away.
///
/// The process is a `StudioServiceProcess` in the `.service` lane, so the server never holds a
/// generation slot and is not a Library run. Reachability comes from `StudioServingMonitor`, which
/// polls the endpoint for the life of the app; ownership comes from the process.
@MainActor
package final class StudioLocalServer: ObservableObject {
    package enum Phase: Equatable {
        /// Nothing answers at the endpoint, and Studio has no server process.
        case stopped
        /// Studio launched a server and the endpoint has not answered yet.
        case starting
        /// Studio's server answers at the endpoint.
        case running
        /// Stop was requested and the process has not exited yet.
        case stopping
        /// A server Studio did not start answers at the endpoint: `api serve` in a terminal, or
        /// another app. Studio can watch it and manage its models, not stop it.
        case external
        /// Studio's server exited without being asked to; the message is why.
        case failed(String)

        /// Whether a server, Studio's or not, answers at the endpoint.
        package var isServing: Bool { self == .running || self == .external }

        /// Whether Studio owns a live server process that Stop and Restart can reach.
        package var isOwned: Bool { self == .starting || self == .running || self == .stopping }

        /// "Running", "Stopped": the state in one word or two, for a title or a menu bar row.
        package var title: String {
            switch self {
            case .stopped: return "Stopped"
            case .starting: return "Starting…"
            case .running: return "Running"
            case .stopping: return "Stopping…"
            case .external: return "Running outside Studio"
            case .failed: return "Stopped unexpectedly"
            }
        }
    }

    /// Where the `api serve` options persist in the task sessions store. The Server page kept
    /// them under its own view scope before the server had an app-wide owner; that value seeds
    /// this one once.
    static let optionsKey = "server.options"
    static let legacyOptionsKey = "server.serving.ServingConsole.draft"

    @Published package private(set) var phase: Phase = .stopped
    /// The `api serve` options: engine, default model, limits, memory guard, KV cache. The
    /// endpoint and API key are not taken from here — they are the controller's runtime
    /// connection, applied at launch, so the key is only ever read from its Keychain setting.
    @Published package var options: CommandDraft {
        didSet { controller?.taskSessions.set(options, for: Self.optionsKey) }
    }

    /// Studio's `api serve` process.
    package let process: StudioServiceProcess
    private weak var controller: MereRunController?
    private var subscriptions = Set<AnyCancellable>()

    package init(controller: MereRunController) {
        self.controller = controller
        process = StudioServiceProcess(templateID: .apiServe, controller: controller)
        let template = CommandCatalog.template(id: .apiServe)
        let fallback = controller.taskSessions.value(
            for: Self.legacyOptionsKey,
            default: template?.defaultDraft() ?? CommandDraft()
        )
        options = controller.taskSessions.value(for: Self.optionsKey, default: fallback)

        // `@Published` emits before it stores, so each sink passes on the value it received and
        // the property still holds the one before it.
        process.$state
            .sink { [weak self, weak controller] state in
                guard let self else { return }
                let exited = self.process.state.isRunning && !state.isRunning
                self.refreshPhase(state: state)
                // The endpoint stops answering the moment the process exits; ask now rather than
                // show the dying server as someone else's until the next poll.
                guard exited, let controller else { return }
                Task { await controller.servingMonitor.refreshRuntimeNow(controller: controller) }
            }
            .store(in: &subscriptions)
        controller.servingMonitor.$lastAnsweredAt
            .sink { [weak self] answeredAt in self?.refreshPhase(answeredAt: answeredAt) }
            .store(in: &subscriptions)
    }

    /// The configured host, or loopback when it is blank — where `runtimeURL` points too.
    private var host: String {
        let host = controller?.runtimeHost.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return host.isEmpty ? "127.0.0.1" : host
    }

    /// `http://127.0.0.1:8080`: where clients reach the server.
    package var endpoint: String {
        guard let controller else { return "" }
        return "http://\(host):\(controller.runtimePort)"
    }

    /// Whether the configured endpoint would be safe to serve on with the configured key.
    package var safety: StudioServingSafety {
        guard let controller else { return .loopback }
        return StudioServingSafety.evaluate(host: host, apiKey: controller.runtimeAPIKey)
    }

    /// `options` on the controller's endpoint, with its key: what Start launches.
    package var launchDraft: CommandDraft {
        var draft = options
        guard let controller else { return draft }
        draft.host = host
        draft.port = controller.runtimePort
        draft.apiKey = controller.runtimeAPIKey
        return draft
    }

    /// The endpoint the monitor polls. A host or port edited in the Command view would start a
    /// server nothing in Studio watches, so the server always binds where Settings says.
    private var pinnedEndpoint: [(flag: String, value: String)] {
        [(CommandFlags.APIServe.host, host), (CommandFlags.APIServe.port, String(controller?.runtimePort ?? 8_080))]
    }

    /// The job running (or last run for) Studio's server, for its log.
    package var ownedJob: Job? { process.job }

    /// Launches `api serve` on the controller's endpoint with `options`. Returns why it did not,
    /// or nil once the server is launching.
    @discardableResult
    package func start() -> String? {
        guard controller != nil else { return "The API server command is unavailable." }
        guard !phase.isOwned else { return nil }
        guard safety != .exposedWithoutAuthentication else {
            return "Add an API key before exposing the server beyond this Mac."
        }
        process.start(draft: launchDraft, pinned: pinnedEndpoint)
        controller?.servingMonitor.note("API server start requested", detail: endpoint)
        return nil
    }

    /// Asks Studio's server to exit (SIGTERM). Returns false when Studio does not own one.
    @discardableResult
    package func stop() -> Bool {
        guard process.stop() else { return false }
        controller?.servingMonitor.note("API server stop requested", detail: endpoint)
        return true
    }

    /// Stops Studio's server, waits for the process to exit so the port is free, then starts a new
    /// one with the current options and endpoint. Returns why it did not restart, or nil.
    package func restart() async -> String? {
        guard phase.isOwned else { return "Only a server Studio started can be restarted." }
        guard safety != .exposedWithoutAuthentication else {
            return "Add an API key before exposing the server beyond this Mac."
        }
        controller?.servingMonitor.note("API server restart requested", detail: endpoint)
        _ = await process.restart(draft: launchDraft, pinned: pinnedEndpoint)
        return nil
    }

    /// The phase a server process and the endpoint's reachability add up to. `answeredAt` is
    /// when the endpoint last answered, or nil while it does not; an answer from before Studio's
    /// server exited is that server's and does not make the endpoint someone else's.
    package static func phase(process: StudioServiceProcess.State, answeredAt: Date?) -> Phase {
        switch process {
        case .running(stopRequested: true, _):
            return .stopping
        case .running(stopRequested: false, let since):
            // Only an answer to a poll sent after this server launched is this server's.
            return answers(after: since, answeredAt) ? .running : .starting
        case .none(let exitedAt):
            return answers(after: exitedAt, answeredAt) ? .external : .stopped
        case .failed(let message, let exitedAt):
            // The usual reason a server fails to start is that one already holds the port.
            return answers(after: exitedAt, answeredAt) ? .external : .failed(message)
        }
    }

    private static func answers(after exitedAt: Date?, _ answeredAt: Date?) -> Bool {
        guard let answeredAt else { return false }
        guard let exitedAt else { return true }
        return answeredAt > exitedAt
    }

    private func refreshPhase(state: StudioServiceProcess.State? = nil, answeredAt: Date?? = nil) {
        guard let controller else { return }
        let next = Self.phase(
            process: state ?? process.state,
            answeredAt: answeredAt ?? controller.servingMonitor.lastAnsweredAt
        )
        if phase != next { phase = next }
    }
}
