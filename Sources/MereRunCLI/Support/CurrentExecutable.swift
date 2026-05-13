import Foundation

enum CurrentExecutable {
    /// Resolve the running `mere.run` executable to an absolute path so child
    /// processes can re-spawn the CLI even when it was invoked from $PATH.
    static func url() -> URL {
        if let path = Bundle.main.executablePath {
            return URL(fileURLWithPath: path)
        }

        let argv0 = CommandLine.arguments.first ?? "mere.run"
        if !argv0.hasPrefix("/"),
           let resolved = which(argv0) {
            return resolved
        }
        return URL(fileURLWithPath: argv0)
    }

    private static func which(_ name: String) -> URL? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [name]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        try? process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let raw = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty else { return nil }
        return URL(fileURLWithPath: raw)
    }
}
