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
    @Published package private(set) var activities: [StudioServiceActivity] = []

    private var pollingTask: Task<Void, Never>?

    package func start(controller: MereRunController) {
        guard pollingTask == nil else { return }
        pollingTask = Task { @MainActor [weak self, weak controller] in
            guard let self, let controller else { return }
            await refreshAgent(controller: controller)
            var iteration = 0
            while !Task.isCancelled {
                await refreshRuntime(controller: controller)
                if iteration > 0, iteration.isMultiple(of: 8) {
                    await refreshAgent(controller: controller)
                }
                iteration += 1
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

    package func note(
        _ title: String,
        detail: String? = nil,
        level: StudioServiceActivity.Level = .info
    ) {
        append([StudioServiceActivity(level: level, title: title, detail: detail)])
    }

    private func refreshRuntime(controller: MereRunController) async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        var request = URLRequest(url: controller.runtimeURL(path: "/runtime/status"))
        request.timeoutInterval = 3
        if let authorization = controller.runtimeAuthorizationHeader {
            request.setValue(authorization, forHTTPHeaderField: "Authorization")
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
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
