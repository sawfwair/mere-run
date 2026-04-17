import Foundation

public enum MereRunRuntimeDebug {
    public static func isEnabled(
        _ keys: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        for key in keys {
            guard let raw = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
                  !raw.isEmpty else {
                continue
            }
            if raw == "1" || raw == "true" || raw == "yes" || raw == "on" {
                return true
            }
        }
        return false
    }

    public static func write(
        _ message: @autoclosure () -> String,
        enabled: Bool
    ) {
        guard enabled else { return }
        let line = message() + "\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    public static func logger(
        keys: [String],
        prefix: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> (@Sendable (String) -> Void)? {
        guard isEnabled(keys, environment: environment) else {
            return nil
        }
        return { message in
            let rendered: String
            if let prefix, !prefix.isEmpty {
                rendered = "\(prefix) \(message)"
            } else {
                rendered = message
            }
            FileHandle.standardError.write(Data((rendered + "\n").utf8))
        }
    }
}
