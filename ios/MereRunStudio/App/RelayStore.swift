import Foundation
import MereRunRelayKit

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

    private let supportBase: URL
    private var profilesURL: URL { supportBase.appendingPathComponent("executors.json") }

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
        profile.map(RelayWorkflowExecutor.init(profile:))
    }

    func refreshAuthStatus() {
        guard let profile else { return }
        authStatus = RelayAuthentication.status(profile: profile)
    }

    /// The mere.world account this pairing belongs to, from the token's
    /// email claim. Fleet visibility is scoped by account.
    var accountEmail: String? {
        guard let tokenFile = profile?.tokenFile,
              let data = try? Data(contentsOf: URL(fileURLWithPath: tokenFile)),
              let tokenSet = try? WorkflowBundleCodec.decoder().decode(RelayOAuthTokenSet.self, from: data) else {
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
        return try await client.submit(
            bundleDirectory: bundle.directory,
            localRunDirectory: runDirectory,
            pluginNodes: []
        )
    }

    func unpair() {
        if let tokenFile = profile?.tokenFile {
            try? FileManager.default.removeItem(atPath: tokenFile)
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
