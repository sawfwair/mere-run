import AuthenticationServices
import Foundation
import MereRunRelayKit
import UIKit

/// Owns the relay profile, authentication state, and client access for the
/// app. Storage mirrors the CLI's JSON shapes inside the app sandbox's
/// Application Support directory, protected with complete file protection, so
/// a profile paired on the phone is the same document the CLI would write.
@MainActor
public final class RelayStore: ObservableObject {
    public typealias DirectAuthorization = @MainActor (
        String, String, @escaping @MainActor () throws -> Void
    ) async throws -> RelayOAuthTokenSet
    public enum PairingState: Equatable {
        case unpaired
        case discovering
        case awaitingApproval(verificationURL: String, userCode: String)
        case paired
        case failed(String)
    }

    @Published public private(set) var profile: WorkflowExecutorProfile?
    @Published public private(set) var connectionID = UUID()
    @Published public private(set) var pairing: PairingState = .unpaired
    @Published private(set) var authStatus: RelayAuthStatusResult?

    static let iosClientID = "mererun-ios"
    static let iosRedirectURI = "https://mere.world/oauth/ios-done"

    private let supportBase: URL
    private let credentialStorageFactory: (String) -> any RelayCredentialStorage
    private let directAuthorization: DirectAuthorization
    private var pairingID: UUID?
    private struct Connection {
        let profile: WorkflowExecutorProfile
        let credentials: RelayCredentialScope
    }
    private var connection: Connection?
    private var profilesURL: URL { supportBase.appendingPathComponent("executors.json") }

    /// Keychain is the credential's home on iOS; the legacy protected file
    /// migrates in on first load and is removed.
    private func credentialStorage(for profileName: String) -> any RelayCredentialStorage {
        credentialStorageFactory(profileName)
    }

    public init(
        supportBase requestedSupportBase: URL? = nil,
        credentialStorageFactory: @escaping (String) -> any RelayCredentialStorage = { profileName in
            KeychainCredentialStorage(service: "run.mere.studio.relay", account: profileName)
        },
        directAuthorization: DirectAuthorization? = nil
    ) {
        let base = requestedSupportBase
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("MereRun", isDirectory: true)
        supportBase = base
        self.credentialStorageFactory = credentialStorageFactory
        self.directAuthorization = directAuthorization ?? Self.authorizeDirect
        try? FileManager.default.createDirectory(
            at: base,
            withIntermediateDirectories: true,
            attributes: [.protectionKey: FileProtectionType.complete]
        )
        // Credentials now live in the Keychain. Recompute the old sandbox path
        // only to migrate installations that predate Keychain storage.
        if let stored = (try? WorkflowExecutorProfileStore.load(from: profilesURL))?
            .profiles.first(where: { $0.kind == .relay }) {
            let legacyTokenFile = RelayAuthentication.defaultTokenFile(
                profileName: stored.name,
                applicationSupportBase: base
            )
            let restoredProfile = WorkflowExecutorProfile(
                name: stored.name,
                kind: stored.kind,
                destination: stored.destination,
                remoteRoot: stored.remoteRoot,
                port: stored.port,
                identityFile: stored.identityFile,
                mereRunPath: stored.mereRunPath,
                url: stored.url,
                tokenFile: nil
            )
            let storage = credentialStorage(for: stored.name)
            if (try? storage.load()) == nil,
               let legacy = try? FileCredentialStorage(url: legacyTokenFile).load() {
                try? storage.save(legacy)
                try? FileCredentialStorage(url: legacyTokenFile).clear()
            }
            connection = Connection(profile: restoredProfile, credentials: RelayCredentialScope(storage: storage))
            profile = restoredProfile
            pairing = .paired
            refreshAuthStatus()
        }
        #if DEBUG
        // UI preview without a paired relay (simulator screenshots, design
        // iteration): `simctl launch <sim> run.mere.studio.ios MERERUN_UI_PREVIEW`
        if ProcessInfo.processInfo.arguments.contains("MERERUN_UI_PREVIEW") {
            pairing = .paired
        }
        #endif
    }

    public var client: RelayWorkflowExecutor? {
        connection.map { RelayWorkflowExecutor(profile: $0.profile, credentialStorage: $0.credentials) }
    }

    var executionPrivacyLane: ExecutionPrivacyLane {
        profile?.name == "direct" ? .directMachine : .hostedRelay
    }

    func refreshAuthStatus() {
        guard let profile else { return }
        if let tokenSet = try? credentialStorage(for: profile.name).load() {
            authStatus = RelayAuthStatusResult(
                executor: "relay:\(profile.name)",
                credentialKind: "keychain-token-set",
                authenticated: tokenSet.isFresh(now: Int64(Date().timeIntervalSince1970), skewSeconds: 0),
                refreshable: tokenSet.refreshToken?.isEmpty == false,
                expiresAtEpochSeconds: tokenSet.expiresAtEpochSeconds(),
                tokenFile: nil
            )
            return
        }
        authStatus = RelayAuthentication.status(profile: profile)
    }

    /// The direct lane: pair straight to a machine running
    /// `mere.run relay serve` over the LAN or a tailnet, exchanging the
    /// terminal's pairing code for a long-lived bearer token in the Keychain.
    /// No broker, no cloud hop — prompts and outputs stay on your network.
    public func pairDirect(urlString: String, code: String, profileName: String = "direct") async {
        let attempt = beginPairing()
        var trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if !trimmed.contains("://") {
            trimmed = "http://\(trimmed)"
        }
        guard let parsed = URL(string: trimmed), parsed.scheme == "http" || parsed.scheme == "https" else {
            pairing = .failed("Enter the machine's address, like lab.local:6373 or a tailnet name.")
            return
        }
        pairing = .discovering
        do {
            let tokenSet = try await directAuthorization(trimmed, code, { try self.checkPairing(attempt) })
            try checkPairing(attempt)
            let candidate = WorkflowExecutorProfile(
                name: profileName,
                kind: .relay,
                destination: nil,
                remoteRoot: nil,
                port: nil,
                identityFile: nil,
                mereRunPath: nil,
                url: trimmed,
                tokenFile: nil
            )
            try commitPairing(tokenSet, profile: candidate, attempt: attempt)
        } catch let error as RelayClientError {
            guard pairingIsCurrent(attempt) else { return }
            pairing = .failed(AppErrorText.presentable(error.message))
        } catch {
            guard pairingIsCurrent(attempt) else { return }
            pairing = .failed(error.localizedDescription)
        }
    }

    private static func authorizeDirect(
        url: String, code: String, checkCurrent: @escaping @MainActor () throws -> Void
    ) async throws -> RelayOAuthTokenSet {
        _ = try await RelayAuthentication.discoverLocalRelay(url: url)
        try checkCurrent()
        let paired = try await RelayAuthentication.pairLocalRelay(url: url, code: code, deviceName: UIDevice.current.name)
        return RelayOAuthTokenSet(accessToken: paired.token, refreshToken: nil, tokenType: "Bearer", expiresIn: nil,
                                 obtainedAtEpochSeconds: Int64(Date().timeIntervalSince1970))
    }

    /// Browser sign-in: Authorization Code + PKCE through an in-app sheet,
    /// intercepted at the universal-link callback. Falls back to the device
    /// grant when the relay's broker does not advertise an authorize URL.
    func signIn(urlString: String, profileName: String = "phone") async {
        let attempt = beginPairing()
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let parsed = URL(string: trimmed), parsed.scheme == "https" || isLoopback(parsed) else {
            pairing = .failed("Relay URL must use HTTPS.")
            return
        }
        let candidate = WorkflowExecutorProfile(
            name: profileName,
            kind: .relay,
            destination: nil,
            remoteRoot: nil,
            port: nil,
            identityFile: nil,
            mereRunPath: nil,
            url: trimmed,
            tokenFile: nil
        )
        pairing = .discovering
        do {
            let configuration = try await RelayAuthentication.discover(profile: candidate)
            try checkPairing(attempt)
            guard configuration.authorizationEndpoint != nil else {
                await pair(urlString: urlString, profileName: profileName)
                return
            }
            let started = try RelayAuthorizationCodeFlow.begin(
                configuration: configuration,
                clientID: Self.iosClientID,
                redirectURI: Self.iosRedirectURI
            )
            let callback = try await WebSignIn.present(url: started.authorizationURL)
            try checkPairing(attempt)
            let code = try RelayAuthorizationCodeFlow.code(fromCallback: callback, started: started)
            let tokenSet = try await RelayAuthorizationCodeFlow.exchange(
                code: code,
                started: started,
                configuration: configuration,
                clientID: Self.iosClientID,
                redirectURI: Self.iosRedirectURI
            )
            try checkPairing(attempt)
            try commitPairing(tokenSet, profile: candidate, attempt: attempt)
        } catch let error as RelayClientError {
            guard pairingIsCurrent(attempt) else { return }
            pairing = .failed(AppErrorText.presentable(error.message))
        } catch let error as ASWebAuthenticationSessionError where error.code == .canceledLogin {
            guard pairingIsCurrent(attempt) else { return }
            pairing = .unpaired
        } catch {
            guard pairingIsCurrent(attempt) else { return }
            pairing = .failed(error.localizedDescription)
        }
    }

    /// The mere.world account this pairing belongs to, from the token's
    /// email claim. Fleet visibility is scoped by account.
    var accountEmail: String? {
        guard let name = profile?.name, let tokenSet = try? credentialStorage(for: name).load() else {
            return nil
        }
        return tokenSet.accountEmail
    }

    /// Runs discovery and the OAuth device grant against a relay base URL,
    /// then persists the profile and token set on approval.
    func pair(urlString: String, profileName: String = "phone") async {
        let attempt = beginPairing()
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let parsed = URL(string: trimmed), parsed.scheme == "https" || isLoopback(parsed) else {
            pairing = .failed("Relay URL must use HTTPS.")
            return
        }
        let candidate = WorkflowExecutorProfile(
            name: profileName,
            kind: .relay,
            destination: nil,
            remoteRoot: nil,
            port: nil,
            identityFile: nil,
            mereRunPath: nil,
            url: trimmed,
            tokenFile: nil
        )
        pairing = .discovering
        do {
            let configuration = try await RelayAuthentication.discover(profile: candidate)
            try checkPairing(attempt)
            let authorization = try await RelayAuthentication.beginDeviceAuthorization(
                configuration: configuration
            )
            try checkPairing(attempt)
            pairing = .awaitingApproval(
                verificationURL: authorization.verificationURIComplete ?? authorization.verificationURI,
                userCode: authorization.userCode
            )
            let tokenSet = try await RelayAuthentication.pollDeviceAuthorization(
                authorization,
                configuration: configuration
            )
            try checkPairing(attempt)
            try commitPairing(tokenSet, profile: candidate, attempt: attempt)
        } catch let error as RelayClientError {
            guard pairingIsCurrent(attempt) else { return }
            pairing = .failed(error.message)
        } catch {
            guard pairingIsCurrent(attempt) else { return }
            pairing = .failed(error.localizedDescription)
        }
    }

    /// Cached worker capabilities for the paired fleet: node kinds and
    /// installed models drive the Create form.
    @Published private(set) var workerProbe: WorkflowExecutorProbe?

    func refreshWorkerProbe() async {
        guard let client else { return }
        let requestedConnection = connectionID
        let probe = try? await client.probe()
        guard requestedConnection == connectionID, !Task.isCancelled else { return }
        workerProbe = probe
    }

    /// Builds a single-node graph from the Create form, materializes it with
    /// the portable environment (models are always pinned explicitly on the
    /// phone), and submits it through relay. Returns the created job.
    func submit(
        kind: String, arguments: [String: WorkflowValue], using suppliedClient: RelayWorkflowExecutor? = nil
    ) async throws -> WorkflowRemoteJob {
        guard let client = suppliedClient ?? client else {
            throw RelayClientError("Pair with a relay before submitting.")
        }
        guard let entry = WorkflowNodeRegistry.entry(for: kind) else {
            throw RelayClientError("Unknown node kind '\(kind)'.")
        }
        let nodeID = "generate"
        let outputName = entry.outputs.first?.name ?? "output"
        let graph = WorkflowGraphDocument(
            schemaVersion: WorkflowGraphDocument.schemaVersion,
            kind: WorkflowGraphDocument.kind,
            name: "phone-\(kind.replacingOccurrences(of: ".", with: "-"))",
            inputs: [:],
            nodes: [
                WorkflowNode(
                    id: nodeID,
                    kind: kind,
                    arguments: arguments,
                    dependsOn: nil
                )
            ],
            outputs: [outputName: .reference("nodes.\(nodeID).outputs.\(outputName)")],
            metadata: nil
        )
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("bundles", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let bundle = try WorkflowBundleMaterializer(
            graph: graph,
            suppliedInputs: WorkflowInputsDocument(values: [:]),
            destination: scratch,
            environment: .portable
        ).materialize()
        let runDirectory = supportBase
            .appendingPathComponent("submitted", isDirectory: true)
            .appendingPathComponent(bundle.job.jobID, isDirectory: true)
        try FileManager.default.createDirectory(at: runDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let job = try await client.submit(
            bundleDirectory: bundle.directory,
            localRunDirectory: runDirectory,
            pluginNodes: []
        )
        #if canImport(ActivityKit)
        RunActivityTracker.track(job: job, title: entry.title, client: client)
        #endif
        return job
    }

    public func unpair() {
        pairingID = nil
        retireConnection()
        if let profile {
            try? credentialStorage(for: profile.name).clear()
            if let tokenFile = profile.tokenFile {
                try? FileManager.default.removeItem(atPath: tokenFile)
            }
        }
        try? FileManager.default.removeItem(at: profilesURL)
        profile = nil
        authStatus = nil
        pairing = .unpaired
    }

    private func retireConnection() {
        connection?.credentials.invalidate()
        connection = nil
        workerProbe = nil
        connectionID = UUID()
        #if canImport(ActivityKit)
        RunActivityTracker.cancelAll()
        #endif
    }

    private func commitPairing(_ tokenSet: RelayOAuthTokenSet, profile candidate: WorkflowExecutorProfile, attempt: UUID) throws {
        try checkPairing(attempt)
        retireConnection()
        // The old connection is gone from here on. Until both durable writes succeed nothing is
        // paired, so a failed write cannot leave the old endpoint presented without a client.
        profile = nil
        authStatus = nil
        try checkPairing(attempt)
        let storage = credentialStorage(for: candidate.name)
        // No token may survive paired with a different endpoint if either durable write fails.
        try storage.clear()
        try WorkflowExecutorProfileStore.save(WorkflowExecutorProfiles(schemaVersion: 1, profiles: [candidate]), to: profilesURL)
        try storage.save(tokenSet)
        connection = Connection(profile: candidate, credentials: RelayCredentialScope(storage: storage))
        profile = candidate
        pairing = .paired
        refreshAuthStatus()
    }

    private func beginPairing() -> UUID {
        let attempt = UUID()
        pairingID = attempt
        return attempt
    }

    private func pairingIsCurrent(_ attempt: UUID) -> Bool {
        pairingID == attempt && !Task.isCancelled
    }

    private func checkPairing(_ attempt: UUID) throws {
        guard pairingIsCurrent(attempt) else { throw CancellationError() }
    }

    private func isLoopback(_ url: URL) -> Bool {
        url.scheme == "http" && ["127.0.0.1", "localhost", "::1"].contains(url.host ?? "")
    }
}

/// Presents the broker sign-in sheet and resolves with the intercepted
/// universal-link callback.
@MainActor
enum WebSignIn {
    private final class ContextProvider: NSObject, ASWebAuthenticationPresentationContextProviding {
        func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
            ASPresentationAnchor()
        }
    }

    private static let contextProvider = ContextProvider()

    static func present(url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            let session = ASWebAuthenticationSession(
                url: url,
                callback: .https(host: "mere.world", path: "/oauth/ios-done")
            ) { callbackURL, error in
                if let error {
                    continuation.resume(throwing: error)
                } else if let callbackURL {
                    continuation.resume(returning: callbackURL)
                } else {
                    continuation.resume(throwing: RelayClientError("Sign-in did not complete."))
                }
            }
            session.presentationContextProvider = contextProvider
            session.start()
        }
    }
}


/// Translates client errors into the app's voice. RelayKit speaks the CLI's
/// language ("executor login", flag names); the phone never should.
enum AppErrorText {
    /// Relay failures carry server text worth translating; anything else shows its own description.
    static func presentable(_ error: any Error) -> String {
        guard let error = error as? RelayClientError else { return error.localizedDescription }
        return presentable(error.message)
    }

    static func presentable(_ message: String) -> String {
        if message.contains("executor login") || message.contains("MERERUN_RELAY_TOKEN") {
            return "Your session expired. Sign out in Settings and pair again."
        }
        if message.contains("Relay session expired") {
            return "Your session expired. Sign out in Settings and pair again."
        }
        if message.contains("missing node kinds") {
            return "Your fleet's nodes don't support this yet. Update mere.run on your nodes."
        }
        if message.contains("missing models") {
            return "The selected model isn't installed on any online node."
        }
        return message
    }
}
