import AuthenticationServices
import Foundation
import MereRunRelayKit
import UIKit

/// Owns the relay profile, authentication state, and client access for the
/// app. Storage mirrors the CLI's JSON shapes inside the app sandbox's
/// Application Support directory, protected with complete file protection, so
/// a profile paired on the phone is the same document the CLI would write.
@MainActor
final class RelayStore: ObservableObject {
    enum PairingState: Equatable {
        case unpaired
        case discovering
        case awaitingApproval(verificationURL: String, userCode: String)
        case paired
        case failed(String)
    }

    @Published private(set) var profile: WorkflowExecutorProfile?
    @Published private(set) var pairing: PairingState = .unpaired
    @Published private(set) var authStatus: RelayAuthStatusResult?

    static let iosClientID = "mererun-ios"
    static let iosRedirectURI = "https://mere.world/oauth/ios-done"

    private let supportBase: URL
    private var profilesURL: URL { supportBase.appendingPathComponent("executors.json") }

    /// Keychain is the credential's home on iOS; the legacy protected file
    /// migrates in on first load and is removed.
    private func keychain(for profileName: String) -> KeychainCredentialStorage {
        KeychainCredentialStorage(service: "run.mere.studio.relay", account: profileName)
    }

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
        supportBase = base
        try? FileManager.default.createDirectory(
            at: base,
            withIntermediateDirectories: true,
            attributes: [.protectionKey: FileProtectionType.complete]
        )
        // iOS moves the data container to a new absolute path on app updates,
        // so the persisted token-file path is advisory only: recompute it from
        // the current container (the file itself migrates with the app).
        if let stored = (try? WorkflowExecutorProfileStore.load(from: profilesURL))?
            .profiles.first(where: { $0.kind == .relay }) {
            let tokenFile = RelayAuthentication.defaultTokenFile(
                profileName: stored.name,
                applicationSupportBase: base
            )
            profile = WorkflowExecutorProfile(
                name: stored.name,
                kind: stored.kind,
                destination: stored.destination,
                remoteRoot: stored.remoteRoot,
                port: stored.port,
                identityFile: stored.identityFile,
                mereRunPath: stored.mereRunPath,
                url: stored.url,
                tokenFile: tokenFile.path
            )
            let storage = keychain(for: stored.name)
            if (try? storage.load()) == nil,
               let legacy = try? FileCredentialStorage(url: tokenFile).load() {
                try? storage.save(legacy)
                try? FileCredentialStorage(url: tokenFile).clear()
            }
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

    var client: RelayWorkflowExecutor? {
        profile.map { RelayWorkflowExecutor(profile: $0, credentialStorage: keychain(for: $0.name)) }
    }

    func refreshAuthStatus() {
        guard let profile else { return }
        if let tokenSet = try? keychain(for: profile.name).load() {
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
    func pairDirect(urlString: String, code: String, profileName: String = "direct") async {
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
            _ = try await RelayAuthentication.discoverLocalRelay(url: trimmed)
            let paired = try await RelayAuthentication.pairLocalRelay(
                url: trimmed,
                code: code,
                deviceName: UIDevice.current.name
            )
            let tokenSet = RelayOAuthTokenSet(
                accessToken: paired.token,
                refreshToken: nil,
                tokenType: "Bearer",
                expiresIn: nil,
                obtainedAtEpochSeconds: Int64(Date().timeIntervalSince1970)
            )
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
            try keychain(for: profileName).save(tokenSet)
            try WorkflowExecutorProfileStore.save(
                WorkflowExecutorProfiles(schemaVersion: 1, profiles: [candidate]),
                to: profilesURL
            )
            profile = candidate
            pairing = .paired
            refreshAuthStatus()
        } catch let error as RelayClientError {
            pairing = .failed(AppErrorText.presentable(error.message))
        } catch {
            pairing = .failed(error.localizedDescription)
        }
    }

    /// Browser sign-in: Authorization Code + PKCE through an in-app sheet,
    /// intercepted at the universal-link callback. Falls back to the device
    /// grant when the relay's broker does not advertise an authorize URL.
    func signIn(urlString: String, profileName: String = "phone") async {
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
            let code = try RelayAuthorizationCodeFlow.code(fromCallback: callback, started: started)
            let tokenSet = try await RelayAuthorizationCodeFlow.exchange(
                code: code,
                started: started,
                configuration: configuration,
                clientID: Self.iosClientID,
                redirectURI: Self.iosRedirectURI
            )
            try keychain(for: profileName).save(tokenSet)
            try WorkflowExecutorProfileStore.save(
                WorkflowExecutorProfiles(schemaVersion: 1, profiles: [candidate]),
                to: profilesURL
            )
            profile = candidate
            pairing = .paired
            refreshAuthStatus()
        } catch let error as RelayClientError {
            pairing = .failed(AppErrorText.presentable(error.message))
        } catch let error as ASWebAuthenticationSessionError where error.code == .canceledLogin {
            pairing = .unpaired
        } catch {
            pairing = .failed(error.localizedDescription)
        }
    }

    /// The mere.world account this pairing belongs to, from the token's
    /// email claim. Fleet visibility is scoped by account.
    var accountEmail: String? {
        guard let name = profile?.name, let tokenSet = try? keychain(for: name).load() else {
            return nil
        }
        return tokenSet.accountEmail
    }

    /// Runs discovery and the OAuth device grant against a relay base URL,
    /// then persists the profile and token set on approval.
    func pair(urlString: String, profileName: String = "phone") async {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard let parsed = URL(string: trimmed), parsed.scheme == "https" || isLoopback(parsed) else {
            pairing = .failed("Relay URL must use HTTPS.")
            return
        }
        let tokenFile = RelayAuthentication.defaultTokenFile(
            profileName: profileName,
            applicationSupportBase: supportBase
        )
        let candidate = WorkflowExecutorProfile(
            name: profileName,
            kind: .relay,
            destination: nil,
            remoteRoot: nil,
            port: nil,
            identityFile: nil,
            mereRunPath: nil,
            url: trimmed,
            tokenFile: tokenFile.path
        )
        pairing = .discovering
        do {
            let configuration = try await RelayAuthentication.discover(profile: candidate)
            let authorization = try await RelayAuthentication.beginDeviceAuthorization(
                configuration: configuration
            )
            pairing = .awaitingApproval(
                verificationURL: authorization.verificationURIComplete ?? authorization.verificationURI,
                userCode: authorization.userCode
            )
            let tokenSet = try await RelayAuthentication.pollDeviceAuthorization(
                authorization,
                configuration: configuration
            )
            try RelayAuthentication.save(tokenSet, to: tokenFile)
            try WorkflowExecutorProfileStore.save(
                WorkflowExecutorProfiles(schemaVersion: 1, profiles: [candidate]),
                to: profilesURL
            )
            profile = candidate
            pairing = .paired
            refreshAuthStatus()
        } catch let error as RelayClientError {
            pairing = .failed(error.message)
        } catch {
            pairing = .failed(error.localizedDescription)
        }
    }

    /// Cached worker capabilities for the paired fleet: node kinds and
    /// installed models drive the Create form.
    @Published private(set) var workerProbe: WorkflowExecutorProbe?

    func refreshWorkerProbe() async {
        guard let client else { return }
        workerProbe = try? await client.probe()
    }

    /// Builds a single-node graph from the Create form, materializes it with
    /// the portable environment (models are always pinned explicitly on the
    /// phone), and submits it through relay. Returns the created job.
    func submit(kind: String, arguments: [String: WorkflowValue]) async throws -> WorkflowRemoteJob {
        guard let client else {
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

    func unpair() {
        if let profile {
            try? keychain(for: profile.name).clear()
            if let tokenFile = profile.tokenFile {
                try? FileManager.default.removeItem(atPath: tokenFile)
            }
        }
        try? FileManager.default.removeItem(at: profilesURL)
        profile = nil
        authStatus = nil
        pairing = .unpaired
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
