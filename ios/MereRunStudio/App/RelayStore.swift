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
        profile = (try? WorkflowExecutorProfileStore.load(from: profilesURL))?
            .profiles.first { $0.kind == .relay }
        if profile != nil {
            pairing = .paired
            refreshAuthStatus()
        }
    }

    var client: RelayWorkflowExecutor? {
        profile.map(RelayWorkflowExecutor.init(profile:))
    }

    func refreshAuthStatus() {
        guard let profile else { return }
        authStatus = RelayAuthentication.status(profile: profile)
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
