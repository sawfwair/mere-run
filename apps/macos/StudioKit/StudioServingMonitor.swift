import Combine
import Foundation

@MainActor
package final class StudioServingMonitor: ObservableObject {
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

    private var pollingTask: Task<Void, Never>?
    /// The `/runtime/status` request in flight. A poll that finds one waits for its answer
    /// rather than sending a second.
    private var runtimePoll: Task<Void, Never>?

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
    }

    package func refreshNow(controller: MereRunController) async {
        await refreshRuntime(controller: controller)
        await refreshAgent(controller: controller)
    }

    /// Polls the endpoint now instead of at the next tick: after a server exits, or after the
    /// endpoint changes. A poll already in flight was sent before whatever prompted this one, so
    /// this waits for it and then sends its own.
    package func refreshRuntimeNow(controller: MereRunController) async {
        if let runtimePoll { await runtimePoll.value }
        await startRuntimePoll(controller: controller)
    }

    package func refreshAgent(controller: MereRunController) async {
        let result = await controller.utilityCommandResult(args: ["agent", "status", "--json"])
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
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            failure = (200..<300).contains(code) ? nil : String(data: data, encoding: .utf8) ?? "HTTP \(code)"
        } catch {
            failure = error.localizedDescription
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

    private func refreshRuntime(controller: MereRunController) async {
        if let runtimePoll { return await runtimePoll.value }
        await startRuntimePoll(controller: controller)
    }

    private func startRuntimePoll(controller: MereRunController) async {
        let poll = Task { @MainActor [weak self, weak controller] in
            guard let self, let controller else { return }
            await self.pollRuntime(controller: controller)
        }
        runtimePoll = poll
        isRefreshing = true
        await poll.value
        if runtimePoll == poll {
            runtimePoll = nil
            isRefreshing = false
        }
    }

    private func pollRuntime(controller: MereRunController) async {
        var request = URLRequest(url: controller.runtimeURL(path: "/runtime/status"))
        request.timeoutInterval = 3
        if let authorization = controller.runtimeAuthorizationHeader {
            request.setValue(authorization, forHTTPHeaderField: "Authorization")
        }

        do {
            // An answer is stamped with when it was asked for: a reply that crosses a server's exit
            // was the exiting server's, not a sign that another one took the port.
            let sentAt = Date()
            let (data, response) = try await URLSession.shared.data(for: request)
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
            let previous = runtime
            let wasReachable = isReachable
            runtime = decoded
            isReachable = true
            connectionDetail = "Connected"
            lastUpdated = Date()
            if wasReachable {
                append(StudioServiceActivityDiff.events(previous: previous, current: decoded))
            } else {
                append(StudioServiceActivityDiff.events(previous: nil, current: decoded))
            }
        } catch {
            let wasReachable = isReachable
            isReachable = false
            lastAnsweredAt = nil
            connectionDetail = "Runtime is not reachable"
            if wasReachable {
                append([.init(level: .warning, title: "Runtime disconnected", detail: error.localizedDescription)])
            }
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
