import Foundation
#if canImport(Darwin)
import Darwin
#endif

public enum MereRunModelSourceConfiguration {
    public static let baseURLEnvironmentKey = "MERERUN_MODEL_SOURCE_BASE_URL"
    public static let packagedBaseURLFilename = "mererun-model-source-base-url.txt"

    public static func publicBaseURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        currentDirectoryPath: String = FileManager.default.currentDirectoryPath,
        commandPath: String = CommandLine.arguments.first ?? "",
        pathEnvironment: String? = ProcessInfo.processInfo.environment["PATH"],
        processExecutablePath: String? = nil
    ) -> URL? {
        let effectiveProcessExecutablePath = processExecutablePath ?? self.processExecutablePath()
        guard let raw = environment[baseURLEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return packagedPublicBaseURL(
                currentDirectoryPath: currentDirectoryPath,
                commandPath: commandPath,
                pathEnvironment: pathEnvironment,
                processExecutablePath: effectiveProcessExecutablePath
            )
        }
        return URL(string: raw)
    }

    public static func publicArchiveURL(
        for key: String,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        currentDirectoryPath: String = FileManager.default.currentDirectoryPath,
        commandPath: String = CommandLine.arguments.first ?? "",
        pathEnvironment: String? = ProcessInfo.processInfo.environment["PATH"],
        processExecutablePath: String? = nil
    ) -> URL? {
        guard let baseURL = publicBaseURL(
            environment: environment,
            currentDirectoryPath: currentDirectoryPath,
            commandPath: commandPath,
            pathEnvironment: pathEnvironment,
            processExecutablePath: processExecutablePath
        ) else {
            return nil
        }
        let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedKey = trimmedKey.hasPrefix("/") ? String(trimmedKey.dropFirst()) : trimmedKey
        guard !normalizedKey.isEmpty else {
            return nil
        }
        return URL(string: normalizedKey, relativeTo: baseURL.appendingPathComponent(""))?.absoluteURL
    }

    public static func missingConfigurationMessage(
        purpose: String = "Managed model downloads"
    ) -> String {
        """
        \(purpose) require explicit model-source configuration. Set \(baseURLEnvironmentKey)=https://your-host.example/models/ for unsigned archives, or configure MERERUN_R2_SIGNED_URL_ENDPOINT / MERERUN_R2_ACCOUNT_ID + MERERUN_R2_ACCESS_KEY_ID + MERERUN_R2_SECRET_ACCESS_KEY for signed downloads.
        """
    }

    public static func hasExplicitDownloadConfiguration(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let keys = [
            baseURLEnvironmentKey,
            "MERERUN_R2_SIGNED_URL_ENDPOINT",
            "R2_SIGNED_URL_ENDPOINT",
            "MERERUN_R2_ACCOUNT_ID",
            "R2_ACCOUNT_ID",
            "MERERUN_R2_ACCESS_KEY_ID",
            "R2_ACCESS_KEY_ID",
            "AWS_ACCESS_KEY_ID",
            "MERERUN_R2_SECRET_ACCESS_KEY",
            "R2_SECRET_ACCESS_KEY",
            "AWS_SECRET_ACCESS_KEY",
        ]

        return keys.contains { key in
            guard let value = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines) else {
                return false
            }
            return !value.isEmpty
        }
    }

    private static func packagedPublicBaseURL(
        currentDirectoryPath: String,
        commandPath: String,
        pathEnvironment: String?,
        processExecutablePath: String?
    ) -> URL? {
        for directory in packagedBaseURLSearchDirectories(
            currentDirectoryPath: currentDirectoryPath,
            commandPath: commandPath,
            pathEnvironment: pathEnvironment,
            processExecutablePath: processExecutablePath
        ) {
            let configURL = directory.appendingPathComponent(packagedBaseURLFilename, isDirectory: false)
            guard let raw = try? String(contentsOf: configURL, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty,
                  let url = URL(string: raw) else {
                continue
            }
            return url
        }

        return nil
    }

    private static func packagedBaseURLSearchDirectories(
        currentDirectoryPath: String,
        commandPath: String,
        pathEnvironment: String?,
        processExecutablePath: String?
    ) -> [URL] {
        var candidates: [URL] = []

        if let processExecutablePath {
            candidates.append(URL(fileURLWithPath: processExecutablePath).resolvingSymlinksInPath().deletingLastPathComponent())
        }

        if let invocationPath = apparentInvocationPath(
            currentDirectoryPath: currentDirectoryPath,
            commandPath: commandPath,
            pathEnvironment: pathEnvironment
        ) {
            candidates.append(invocationPath.deletingLastPathComponent())
        }

        var seen = Set<String>()
        return candidates.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    private static func apparentInvocationPath(
        currentDirectoryPath: String,
        commandPath: String,
        pathEnvironment: String?
    ) -> URL? {
        guard !commandPath.isEmpty else {
            return nil
        }

        let fm = FileManager.default
        if commandPath.contains("/") {
            let cwd = URL(fileURLWithPath: currentDirectoryPath, isDirectory: true)
            return URL(fileURLWithPath: commandPath, relativeTo: cwd).standardizedFileURL
        }

        let pathEntries = pathEnvironment?.split(separator: ":") ?? []
        for entry in pathEntries {
            let candidate = URL(fileURLWithPath: String(entry), isDirectory: true)
                .appendingPathComponent(commandPath, isDirectory: false)
            if fm.isExecutableFile(atPath: candidate.path) {
                return candidate.standardizedFileURL
            }
        }

        return nil
    }

#if canImport(Darwin)
    private static func processExecutablePath() -> String? {
        var buffer = [CChar](repeating: 0, count: 4096)
        let result = proc_pidpath(getpid(), &buffer, UInt32(buffer.count))
        guard result > 0 else {
            return nil
        }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
#else
    private static func processExecutablePath() -> String? { nil }
#endif
}
