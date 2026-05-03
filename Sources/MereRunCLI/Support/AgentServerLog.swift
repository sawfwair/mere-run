import Foundation
import MereRunCore

enum AgentServerLog {
    static func makeLogHandle(prefix: String) throws -> (url: URL, handle: FileHandle) {
        let directory = MereRunModelPaths.outputDir(for: "agent")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let timestamp = ISO8601DateFormatter()
            .string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let url = directory.appendingPathComponent("\(prefix)-\(timestamp).log", isDirectory: false)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: Data("mere.run agent server log\n".utf8))
        return (url, handle)
    }

    static func nullInputHandle() -> FileHandle? {
        FileHandle(forReadingAtPath: "/dev/null")
    }
}
