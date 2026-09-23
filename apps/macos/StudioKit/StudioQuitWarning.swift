import Foundation

/// What quitting the app would stop. Every child process the app launched ends with it, so Quit
/// asks first when that includes a server Studio started or work the user is waiting on.
package enum StudioQuitWarning {
    /// The alert's explanation, or nil when nothing the user started is in flight. `servers` names
    /// the servers Studio started that are still running: "API server", "vision server".
    package static func message(servers: [String], runningJobs: Int, queuedJobs: Int) -> String? {
        var stops: [String] = []
        if !servers.isEmpty { stops.append("the \(list(servers)) Studio started") }
        if runningJobs > 0 { stops.append(count(runningJobs, "running job")) }
        var sentences: [String] = []
        if !stops.isEmpty {
            sentences.append("Quitting stops \(stops.joined(separator: servers.count > 1 ? ", and " : " and ")).")
        }
        if queuedJobs > 0 {
            let queued = count(queuedJobs, "queued job")
            sentences.append(sentences.isEmpty ? "Quitting drops \(queued)." : "\(queued.prefix(1).uppercased())\(queued.dropFirst()) will not start.")
        }
        return sentences.isEmpty ? nil : sentences.joined(separator: " ")
    }

    /// The warning for `controller`'s current state: the servers it started and its inference
    /// work. Utility reads and readiness probes are not the user's work and never hold Quit back.
    @MainActor
    package static func message(for controller: MereRunController) -> String? {
        var servers: [String] = []
        if controller.localServer.phase.isOwned { servers.append("API server") }
        for server in controller.residentServers where server.state.isRunning {
            servers.append(server.title.prefix(1).lowercased() + server.title.dropFirst())
        }
        return message(
            servers: servers,
            runningJobs: controller.jobs.running(in: .inference).count,
            queuedJobs: controller.jobs.queued(in: .inference).count
        )
    }

    /// "API server", "API server and vision server", "API server, vision server, and music server".
    private static func list(_ items: [String]) -> String {
        switch items.count {
        case 0, 1: return items.first ?? ""
        case 2: return "\(items[0]) and \(items[1])"
        default: return items.dropLast().joined(separator: ", ") + ", and " + items[items.count - 1]
        }
    }

    private static func count(_ value: Int, _ noun: String) -> String {
        value == 1 ? "1 \(noun)" : "\(value) \(noun)s"
    }
}
