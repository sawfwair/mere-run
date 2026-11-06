import Combine
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
package final class StudioCrashReporter: @preconcurrency ObservableObject {
    /// Mirrors the Settings toggle so the app can start and stop capture without a relaunch.
    package static let optInDefaultsKey = "studio.diagnostics.metricKitOptIn"

    package let objectWillChange = ObservableObjectPublisher()
    package var isCapturing: Bool { subscription != nil }
    package private(set) var storedPayloadCount = 0

    private let logger = Logger(subsystem: "run.mere.studio", category: "diagnostics")
    private let directory: URL
    private let defaults: UserDefaults
    private let now: @MainActor () -> Date
    private let subscribe: @MainActor (StudioMetricSubscription) -> Void
    private let unsubscribe: @MainActor (StudioMetricSubscription) -> Void
    private var subscription: StudioMetricSubscription?

    package convenience init() {
        self.init(directory: Self.payloadDirectory(), defaults: .standard,
                  subscribe: { MXMetricManager.shared.add($0) }, unsubscribe: { MXMetricManager.shared.remove($0) })
    }

    init(
        directory: URL,
        defaults: UserDefaults,
        now: @escaping @MainActor () -> Date = { Date() },
        subscribe: @escaping @MainActor (StudioMetricSubscription) -> Void,
        unsubscribe: @escaping @MainActor (StudioMetricSubscription) -> Void
    ) {
        self.directory = directory
        self.defaults = defaults
        self.now = now
        self.subscribe = subscribe
        self.unsubscribe = unsubscribe
    }

    package static func payloadDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("Diagnostics", isDirectory: true)
    }

    /// Applies the stored preference. Safe to call repeatedly.
    package func applyStoredPreference() {
        setCapturing(defaults.bool(forKey: Self.optInDefaultsKey))
    }

    package func setCapturing(_ enabled: Bool) {
        guard enabled != isCapturing else {
            refreshStoredPayloadCount()
            return
        }
        if enabled {
            let subscription = StudioMetricSubscription { [weak self] source, ticket, payloads, kind in
                Task { @MainActor in
                    guard let self else { return }
                    for data in payloads {
                        guard self.subscription === source, source.isCurrent(ticket) else { return }
                        self.write(data, prefix: kind.rawValue)
                    }
                }
            }
            self.subscription = subscription
            subscribe(subscription)
        } else if let current = subscription {
            subscription = nil
            current.close()
            unsubscribe(current)
        }
        defaults.set(enabled, forKey: Self.optInDefaultsKey)
        refreshStoredPayloadCount()
    }

    package func refreshStoredPayloadCount() {
        let contents = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        storedPayloadCount = contents?.filter { $0.pathExtension == "json" }.count ?? 0
        objectWillChange.send()
    }

    /// Deletes every captured payload. The user owns this data, so removing it is one action.
    package func deleteStoredPayloads() {
        subscription?.discardPending()
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
        do {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
            let url = directory.appendingPathComponent(
                "\(prefix)-\(formatter.string(from: now()))-\(UUID()).json",
                isDirectory: false
            )
            try data.write(to: url, options: .atomic)
            refreshStoredPayloadCount()
        } catch {
            logger.error("Could not write \(prefix, privacy: .public) payload: \(error)")
        }
    }
}

/// One opt-in lifetime. The lock protects delivery identity while native callbacks
/// encode off the main actor. Closing is permanent; deletion advances the identity
/// so queued reports cannot restore files the user just deleted.
final class StudioMetricSubscription: NSObject, MXMetricManagerSubscriber, @unchecked Sendable {
    enum Kind: String, Sendable { case metrics, diagnostics }

    private let handler: @Sendable (StudioMetricSubscription, UUID, [Data], Kind) -> Task<Void, Never>
    private let lock = NSLock()
    private var generation: UUID? = UUID()

    init(handler: @escaping @Sendable (StudioMetricSubscription, UUID, [Data], Kind) -> Task<Void, Never>) {
        self.handler = handler
    }

    @MainActor
    func close() { lock.withLock { generation = nil } }

    @MainActor
    func discardPending() {
        lock.withLock {
            if generation != nil { generation = UUID() }
        }
    }

    func isCurrent(_ ticket: UUID) -> Bool { lock.withLock { generation == ticket } }

    func didReceive(_ payloads: [MXMetricPayload]) {
        receive(kind: .metrics) { payloads.map { $0.jsonRepresentation() } }
    }

    func didReceive(_ payloads: [MXDiagnosticPayload]) {
        receive(kind: .diagnostics) { payloads.map { $0.jsonRepresentation() } }
    }

    @discardableResult
    func receiveEncoded(_ payloads: [Data], kind: Kind) -> Task<Void, Never>? {
        receive(kind: kind) { payloads }
    }

    @discardableResult
    func receive(kind: Kind, encode: () -> [Data]) -> Task<Void, Never>? {
        guard let ticket = lock.withLock({ generation }) else { return nil }
        return handler(self, ticket, encode(), kind)
    }
}
