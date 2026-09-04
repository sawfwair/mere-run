import Foundation
import MetricKit
import os

/// Opt-in local crash and hang capture.
///
/// MetricKit hands the app a daily payload describing crashes, hangs, disk writes, and
/// CPU exceptions that already happened. Nothing is transmitted: payloads are written to
/// the user's Application Support directory so they can be attached to an issue through
/// **Export Diagnostics**. Capture is off until the user turns it on in Settings.
@MainActor
package final class StudioCrashReporter: NSObject, ObservableObject {
    /// Mirrors the Settings toggle so the app can start and stop capture without a relaunch.
    package static let optInDefaultsKey = "studio.diagnostics.metricKitOptIn"

    @Published package private(set) var isCapturing = false
    @Published package private(set) var storedPayloadCount = 0

    private let logger = Logger(subsystem: "run.mere.studio", category: "diagnostics")

    package static func payloadDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("Diagnostics", isDirectory: true)
    }

    /// Applies the stored preference. Safe to call repeatedly.
    package func applyStoredPreference() {
        setCapturing(UserDefaults.standard.bool(forKey: Self.optInDefaultsKey))
    }

    package func setCapturing(_ enabled: Bool) {
        guard enabled != isCapturing else {
            refreshStoredPayloadCount()
            return
        }
        if enabled {
            MXMetricManager.shared.add(self)
        } else {
            MXMetricManager.shared.remove(self)
        }
        isCapturing = enabled
        UserDefaults.standard.set(enabled, forKey: Self.optInDefaultsKey)
        refreshStoredPayloadCount()
    }

    package func refreshStoredPayloadCount() {
        let contents = try? FileManager.default.contentsOfDirectory(
            at: Self.payloadDirectory(),
            includingPropertiesForKeys: nil
        )
        storedPayloadCount = contents?.filter { $0.pathExtension == "json" }.count ?? 0
    }

    /// Deletes every captured payload. The user owns this data, so removing it is one action.
    package func deleteStoredPayloads() {
        let directory = Self.payloadDirectory()
        if let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ) {
            for url in contents where url.pathExtension == "json" {
                try? FileManager.default.removeItem(at: url)
            }
        }
        refreshStoredPayloadCount()
    }

    private func write(_ data: Data, prefix: String) {
        let directory = Self.payloadDirectory()
        do {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
            let url = directory.appendingPathComponent(
                "\(prefix)-\(formatter.string(from: Date())).json",
                isDirectory: false
            )
            try data.write(to: url, options: .atomic)
            refreshStoredPayloadCount()
        } catch {
            logger.error("Could not write \(prefix, privacy: .public) payload: \(error)")
        }
    }
}

extension StudioCrashReporter: MXMetricManagerSubscriber {
    package nonisolated func didReceive(_ payloads: [MXMetricPayload]) {
        let encoded = payloads.map { $0.jsonRepresentation() }
        Task { @MainActor in
            for data in encoded {
                write(data, prefix: "metrics")
            }
        }
    }

    package nonisolated func didReceive(_ payloads: [MXDiagnosticPayload]) {
        let encoded = payloads.map { $0.jsonRepresentation() }
        Task { @MainActor in
            for data in encoded {
                write(data, prefix: "diagnostics")
            }
        }
    }
}
