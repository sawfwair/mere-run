import Foundation

/// The single parser for `MERERUN_*` on/off switches.
///
/// Every runtime switch reads through this type so a value means the same thing everywhere:
/// `1`, `true`, `yes`, and `on` enable; `0`, `false`, `no`, and `off` disable. Matching ignores
/// case and surrounding whitespace. An unset, empty, or unrecognized value leaves the caller's
/// default in place. `MereRunEnvironmentRegistry` validates the same vocabulary.
public enum MereRunEnvironmentSwitch {
    /// Parses a raw switch value. Returns `nil` for `nil`, empty, or unrecognized text.
    public static func parse(_ rawValue: String?) -> Bool? {
        guard let rawValue else { return nil }
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return nil
        }
    }

    /// The explicit setting of `key`, or `nil` when it is unset or unrecognized.
    public static func value(
        _ key: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool? {
        parse(environment[key])
    }

    /// Whether `key` is enabled, falling back to `defaultValue` when it is unset or unrecognized.
    public static func isEnabled(
        _ key: String,
        default defaultValue: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        value(key, environment: environment) ?? defaultValue
    }
}
