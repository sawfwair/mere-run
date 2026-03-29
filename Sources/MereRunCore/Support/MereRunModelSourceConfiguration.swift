import Foundation

public enum MereRunModelSourceConfiguration {
    public static let baseURLEnvironmentKey = "MERERUN_MODEL_SOURCE_BASE_URL"

    public static func publicBaseURL(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        guard let raw = environment[baseURLEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return nil
        }
        return URL(string: raw)
    }

    public static func publicArchiveURL(
        for key: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        guard let baseURL = publicBaseURL(environment: environment) else {
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
}
