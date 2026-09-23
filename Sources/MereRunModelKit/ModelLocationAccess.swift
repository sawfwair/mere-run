import Foundation

/// Why lookup could not read a model location.
public enum ModelLocationProblem: String, Codable, Equatable, Sendable {
    /// The directory read failed with a permission error.
    case denied
    /// The read did not return before the deadline. macOS holds file access on a removable
    /// or network volume while its consent prompt is unanswered, and a sleeping or
    /// disconnected volume can block the same way.
    case unresponsive
}

/// A model location that lookup could not read.
public struct ModelLocationIssue: Codable, Equatable, Sendable {
    public let path: String
    public let problem: ModelLocationProblem

    public init(path: String, problem: ModelLocationProblem) {
        self.path = path
        self.problem = problem
    }
}

/// Checks model locations off-thread, with a deadline, before lookup touches paths in them.
///
/// A blocked filesystem call cannot be cancelled, so each check runs on its own thread and
/// callers stop waiting at the deadline. While that check stays blocked, its location is
/// reported unresponsive at once and no second check starts.
public final class ModelLocationAccess: @unchecked Sendable {
    public static let shared = ModelLocationAccess()

    private struct Entry {
        var problem: ModelLocationProblem?
        var checkedAt: Date?
        var checkStartedAt: Date?
    }

    private let deadline: TimeInterval
    private let recheckInterval: TimeInterval
    private let read: @Sendable (URL) -> ModelLocationProblem?
    private let condition = NSCondition()
    private var entries: [String: Entry] = [:]

    public init(
        deadline: TimeInterval = 3,
        recheckInterval: TimeInterval = 30,
        read: @escaping @Sendable (URL) -> ModelLocationProblem? = ModelLocationAccess.readDirectory
    ) {
        self.deadline = deadline
        self.recheckInterval = recheckInterval
        self.read = read
    }

    /// Configured locations that could not be read, checked in parallel.
    public func issues(in locations: ModelLocationSnapshot) -> [ModelLocationIssue] {
        issues(for: locations.locationRoots)
    }

    /// Paths of locations whose check is blocked; lookup skips candidates under them.
    public func unresponsivePaths(in locations: ModelLocationSnapshot) -> Set<String> {
        Set(issues(in: locations).filter { $0.problem == .unresponsive }.map(\.path))
    }

    public func issues(for roots: [URL]) -> [ModelLocationIssue] {
        var seen: Set<String> = []
        let paths = roots.map(\.standardizedFileURL.path).filter { seen.insert($0).inserted }

        condition.lock()
        defer { condition.unlock() }

        let now = Date()
        for path in paths {
            var entry = entries[path] ?? Entry()
            let isFresh = entry.checkedAt.map { now.timeIntervalSince($0) < recheckInterval } ?? false
            guard entry.checkStartedAt == nil, !isFresh else { continue }
            entry.checkStartedAt = now
            entries[path] = entry
            startCheck(path)
        }

        while let wait = paths.compactMap({ entries[$0]?.checkStartedAt?.addingTimeInterval(deadline) })
            .filter({ $0 > Date() })
            .max() {
            condition.wait(until: wait)
        }

        return paths.compactMap { path in
            let entry = entries[path] ?? Entry()
            let problem = entry.checkStartedAt == nil ? entry.problem : .unresponsive
            return problem.map { ModelLocationIssue(path: path, problem: $0) }
        }
    }

    private func startCheck(_ path: String) {
        let thread = Thread { [self] in
            let problem = read(URL(fileURLWithPath: path, isDirectory: true))
            condition.lock()
            entries[path] = Entry(problem: problem, checkedAt: Date())
            condition.broadcast()
            condition.unlock()
        }
        thread.name = "run.mere.model-location-check"
        thread.start()
    }

    /// Lists the directory. Only a permission error is a problem; lookup reports a missing directory itself.
    public static func readDirectory(_ root: URL) -> ModelLocationProblem? {
        do {
            _ = try FileManager().contentsOfDirectory(atPath: root.resolvingSymlinksInPath().path)
            return nil
        } catch CocoaError.fileReadNoPermission {
            return .denied
        } catch {
            return nil
        }
    }
}
