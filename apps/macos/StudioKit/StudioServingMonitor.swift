import Combine
import Foundation

@MainActor
package final class StudioServingMonitor: ObservableObject {
    private let fetch: @Sendable (URLRequest) async throws -> (Data, URLResponse)

    package init(fetch: @escaping @Sendable (URLRequest) async throws -> (Data, URLResponse) = {
        try await URLSession.shared.data(for: $0)
    }) {
        self.fetch = fetch
    }
    @Published package private(set) var runtime: StudioRuntimeSnapshot?
    @Published package private(set) var agentStatus: StudioAgentStatus?
    @Published package private(set) var isReachable = false
    @Published package private(set) var isRefreshing = false
    @Published package private(set) var connectionDetail = "Waiting for runtime"
    @Published package private(set) var agentDetail = "Agent readiness has not been checked"
    @Published package private(set) var lastUpdated: Date?
    /// When the endpoint last answered a poll — with a snapshot, or with an error status such as
    /// 401 — or nil once a poll gets no answer. A server that rejects the key is still up.
    @Published package private(set) var lastAnsweredAt: Date?
    @Published package private(set) var activities: [StudioServiceActivity] = []
    /// Tokens generated per second between consecutive polls, oldest first, for the menu bar's
    /// sparkline. It reads 0 while the server is idle and empties when the server goes away.
    @Published package internal(set) var throughputHistory: [Double] = []
    private var lastTokenCount: (tokens: Int, at: Date)?
    /// Consecutive `/runtime/status` polls that timed out on a server whose `/health` answered.
    /// Each such request stays open on the server, so while this is nonzero the monitor checks
    /// liveness through `/health` and asks for status only every `slowStatusRetryInterval` polls.
    private(set) var slowStatusPolls = 0
    static let slowStatusRetryInterval = 15

    private var pollingTask: Task<Void, Never>?
    /// The `/runtime/status` request in flight. A poll that finds one waits for its answer
    /// rather than sending a second.
    private struct RuntimeEndpoint: Equatable {
        let statusURL: URL
        let healthURL: URL
        let authorization: String?
    }

    private struct RuntimePoll {
        let id: UUID
        let endpoint: RuntimeEndpoint
        let task: Task<Void, Never>
    }

    private var runtimePoll: RuntimePoll?
    private var currentEndpoint: RuntimeEndpoint?
    private var lifetime = UUID()
    private var agentRefreshID: UUID?

    package func start(controller: MereRunController) {
        guard pollingTask == nil else { return }
        // Agent readiness spawns a CLI process, so it is read when the Agents section asks for it
        // rather than on this loop, which runs for the life of the app.
        pollingTask = Task { @MainActor [weak self, weak controller] in
            guard let self, let controller else { return }
            while !Task.isCancelled {
                await refreshRuntime(controller: controller)
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    package func stop() {
        pollingTask?.cancel()
        pollingTask = nil
        runtimePoll?.task.cancel()
        runtimePoll = nil
        agentRefreshID = nil
        lifetime = UUID()
        isRefreshing = false
    }

    package func refreshNow(controller: MereRunController) async {
        let lifetime = lifetime
        await refreshRuntime(controller: controller)
        guard lifetime == self.lifetime, !Task.isCancelled else { return }
        await refreshAgent(controller: controller)
    }

    /// Polls the endpoint now instead of at the next tick: after a server exits, or after the
    /// endpoint changes. A poll already in flight was sent before whatever prompted this one, so
    /// this invalidates it and sends a fresh request. Superseded replies cannot publish.
    package func refreshRuntimeNow(controller: MereRunController) async {
        await startRuntimePoll(controller: controller)
    }

    package func refreshAgent(controller: MereRunController) async {
        let ticket = UUID()
        agentRefreshID = ticket
        let result = await controller.utilityCommandResult(args: ["agent", "status", "--json"])
        guard agentRefreshID == ticket, !Task.isCancelled else { return }
        guard result.exitCode == 0,
              let data = Self.jsonObjectData(in: result.stdout) else {
            agentDetail = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "Agent readiness is unavailable"
                : StudioActivitySanitizer.sanitize(result.stderr)
            return
        }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            agentStatus = try decoder.decode(StudioAgentStatus.self, from: data)
            agentDetail = "Readiness checked"
        } catch {
            agentDetail = "This CLI does not expose typed agent readiness yet"
        }
    }

    /// Loads or unloads one text model on the runtime (`POST /runtime/models/<id>/load|unload`),
    /// records the outcome in the activity feed, and re-reads the pool. Returns the error to show,
    /// or nil once the runtime accepted the request.
    package func setModel(_ id: String, loaded: Bool, controller: MereRunController) async -> String? {
        let endpoint = endpoint(controller)
        let lifetime = lifetime
        let action = loaded ? "load" : "unload"
        let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
        var request = URLRequest(url: controller.runtimeURL(path: "/runtime/models/\(encoded)/\(action)"))
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        if let authorization = controller.runtimeAuthorizationHeader {
            request.setValue(authorization, forHTTPHeaderField: "Authorization")
        }
        let failure: String?
        do {
            let (data, response) = try await fetch(request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            failure = (200..<300).contains(code) ? nil : String(data: data, encoding: .utf8) ?? "HTTP \(code)"
        } catch {
            failure = error.localizedDescription
        }
        guard lifetime == self.lifetime, endpoint == self.endpoint(controller), !Task.isCancelled else {
            return "The runtime connection changed before the request completed."
        }
        if let failure {
            note("Model \(action) failed", detail: failure, level: .error)
            return StudioActivitySanitizer.sanitize(failure)
        }
        note(loaded ? "Model load requested" : "Model unload requested", detail: id, level: .success)
        await refreshRuntime(controller: controller)
        return nil
    }

    package func note(
        _ title: String,
        detail: String? = nil,
        level: StudioServiceActivity.Level = .info
    ) {
        append([StudioServiceActivity(level: level, title: title, detail: detail)])
    }

    private func endpoint(_ controller: MereRunController) -> RuntimeEndpoint {
        RuntimeEndpoint(statusURL: controller.runtimeURL(path: "/runtime/status"),
                        healthURL: controller.runtimeURL(path: "/health"),
                        authorization: controller.runtimeAuthorizationHeader)
    }

    private func refreshRuntime(controller: MereRunController) async {
        if let poll = runtimePoll, poll.endpoint == endpoint(controller) { return await poll.task.value }
        await startRuntimePoll(controller: controller)
    }

    private func startRuntimePoll(controller: MereRunController) async {
        guard !Task.isCancelled else { return }
        let endpoint = endpoint(controller)
        runtimePoll?.task.cancel()
        if currentEndpoint != endpoint {
            currentEndpoint = endpoint
            runtime = nil
            lastAnsweredAt = nil
            lastUpdated = nil
            isReachable = false
            lastTokenCount = nil
            throughputHistory = []
            slowStatusPolls = 0
            activities = []
            connectionDetail = "Waiting for runtime"
        }
        let ticket = UUID()
        let task = Task { @MainActor [weak self, weak controller] in
            guard let self, let controller else { return }
            await self.pollRuntime(controller: controller, endpoint: endpoint, ticket: ticket)
        }
        runtimePoll = RuntimePoll(id: ticket, endpoint: endpoint, task: task)
        isRefreshing = true
        await task.value
        if runtimePoll?.id == ticket {
            runtimePoll = nil
            isRefreshing = false
        }
    }

    private func isCurrent(_ ticket: UUID, endpoint: RuntimeEndpoint, controller: MereRunController) -> Bool {
        runtimePoll?.id == ticket && endpoint == self.endpoint(controller) && !Task.isCancelled
    }

    private func pollRuntime(controller: MereRunController, endpoint: RuntimeEndpoint, ticket: UUID) async {
        guard isCurrent(ticket, endpoint: endpoint, controller: controller) else { return }
        if slowStatusPolls > 0, !slowStatusPolls.isMultiple(of: Self.slowStatusRetryInterval) {
            let sentAt = Date()
            let healthy = await healthAnswers(endpoint: endpoint)
            guard isCurrent(ticket, endpoint: endpoint, controller: controller) else { return }
            slowStatusPolls += 1
            if healthy {
                lastAnsweredAt = sentAt
            } else {
                markUnreachable(detail: "Runtime is not reachable")
            }
            return
        }
        var request = URLRequest(url: endpoint.statusURL)
        request.timeoutInterval = 3
        if let authorization = endpoint.authorization {
            request.setValue(authorization, forHTTPHeaderField: "Authorization")
        }
        do {
            let sentAt = Date()
            let (data, response) = try await fetch(request)
            guard isCurrent(ticket, endpoint: endpoint, controller: controller) else { return }
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            lastAnsweredAt = sentAt
            guard (200..<300).contains(code) else {
                let wasReachable = isReachable
                isReachable = false
                connectionDetail = code == 401
                    ? "Authentication failed — check the API key"
                    : "Runtime returned HTTP \(code)"
                if wasReachable {
                    append([.init(level: .warning, title: "Runtime disconnected", detail: connectionDetail)])
                }
                return
            }
            let decoded = try JSONDecoder().decode(StudioRuntimeSnapshot.self, from: data)
            slowStatusPolls = 0
            let previous = runtime
            let wasReachable = isReachable
            runtime = decoded
            isReachable = true
            connectionDetail = "Connected"
            lastUpdated = Date()
            recordThroughput(decoded, at: sentAt)
            append(StudioServiceActivityDiff.events(previous: wasReachable ? previous : nil, current: decoded))
        } catch {
            guard isCurrent(ticket, endpoint: endpoint, controller: controller) else { return }
            if (error as? URLError)?.code == .timedOut {
                let healthy = await healthAnswers(endpoint: endpoint)
                guard isCurrent(ticket, endpoint: endpoint, controller: controller) else { return }
                if healthy {
                    if slowStatusPolls == 0 {
                        note("Runtime is slow to report status", detail: "The server answers /health but not /runtime/status", level: .warning)
                    }
                    slowStatusPolls += 1
                    lastAnsweredAt = Date()
                    isReachable = false
                    connectionDetail = "Up, but not reporting its status"
                    return
                }
            }
            markUnreachable(detail: error.localizedDescription)
        }
    }

    private func markUnreachable(detail: String) {
        let wasReachable = isReachable || slowStatusPolls > 0
        isReachable = false
        lastAnsweredAt = nil
        lastTokenCount = nil
        throughputHistory = []
        slowStatusPolls = 0
        connectionDetail = "Runtime is not reachable"
        if wasReachable {
            append([.init(level: .warning, title: "Runtime disconnected", detail: detail)])
        }
    }

    /// Whether `/health` answers within two seconds: the server is up, whatever its status says.
    private func healthAnswers(endpoint: RuntimeEndpoint) async -> Bool {
        var request = URLRequest(url: endpoint.healthURL)
        request.timeoutInterval = 2
        guard let (_, response) = try? await fetch(request) else { return false }
        return ((response as? HTTPURLResponse)?.statusCode ?? 0) > 0
    }

    /// Appends the generation rate since the previous poll. A counter that went backwards is a
    /// restarted server: that poll starts a new baseline rather than reading a negative rate.
    func recordThroughput(_ snapshot: StudioRuntimeSnapshot, at date: Date) {
        let perModel = snapshot.textModels.compactMap(\.benchmarkStats?.generatedTokens)
        // A runtime that reports no token counts has no rate to show; "Idle" would be a guess.
        guard let tokens = snapshot.benchmarkStats?.generatedTokens
            ?? (perModel.isEmpty ? nil : perModel.reduce(0, +)) else { return }
        defer { lastTokenCount = (tokens, date) }
        guard let last = lastTokenCount, tokens >= last.tokens, date > last.at else { return }
        throughputHistory.append(Double(tokens - last.tokens) / date.timeIntervalSince(last.at))
        if throughputHistory.count > StudioMachineMonitor.historyLength {
            throughputHistory.removeFirst(throughputHistory.count - StudioMachineMonitor.historyLength)
        }
    }

    private func append(_ events: [StudioServiceActivity]) {
        guard !events.isEmpty else { return }
        activities.insert(contentsOf: events.reversed(), at: 0)
        if activities.count > 200 {
            activities.removeLast(activities.count - 200)
        }
    }

    private static func jsonObjectData(in text: String) -> Data? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}"),
              start <= end else {
            return nil
        }
        return String(text[start...end]).data(using: .utf8)
    }
}
