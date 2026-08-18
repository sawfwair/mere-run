import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public struct RelayDiscoveryDocument: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let kind: String
    public let auth: RelayAuthConfiguration

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case auth
    }

    public init(schemaVersion: Int, kind: String, auth: RelayAuthConfiguration) {
        self.schemaVersion = schemaVersion
        self.kind = kind
        self.auth = auth
    }
}

public struct RelayAuthConfiguration: Codable, Equatable, Sendable {
    public let issuer: String
    public let deviceAuthorizationEndpoint: String
    public let tokenEndpoint: String
    public let clientID: String
    public let scope: String
    public let authorizationEndpoint: String?

    enum CodingKeys: String, CodingKey {
        case issuer
        case deviceAuthorizationEndpoint = "device_authorization_endpoint"
        case tokenEndpoint = "token_endpoint"
        case clientID = "client_id"
        case scope
        case authorizationEndpoint = "authorization_endpoint"
    }

    public init(
        issuer: String,
        deviceAuthorizationEndpoint: String,
        tokenEndpoint: String,
        clientID: String,
        scope: String,
        authorizationEndpoint: String? = nil
    ) {
        self.issuer = issuer
        self.deviceAuthorizationEndpoint = deviceAuthorizationEndpoint
        self.tokenEndpoint = tokenEndpoint
        self.clientID = clientID
        self.scope = scope
        self.authorizationEndpoint = authorizationEndpoint
    }
}

public struct RelayOAuthTokenSet: Codable, Equatable, Sendable {
    public let accessToken: String
    public let refreshToken: String?
    public let tokenType: String?
    public let expiresIn: Int64?
    public let obtainedAtEpochSeconds: Int64?
    public let issuer: String?
    public let tokenEndpoint: String?
    public let clientID: String?
    public let scope: String?

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case tokenType = "token_type"
        case expiresIn = "expires_in"
        case obtainedAtEpochSeconds = "obtained_at_epoch_seconds"
        case issuer
        case tokenEndpoint = "token_endpoint"
        case clientID = "client_id"
        case scope
    }

    public init(
        accessToken: String,
        refreshToken: String?,
        tokenType: String?,
        expiresIn: Int64?,
        obtainedAtEpochSeconds: Int64?,
        issuer: String? = nil,
        tokenEndpoint: String? = nil,
        clientID: String? = nil,
        scope: String? = nil
    ) {
        self.accessToken = accessToken
        self.refreshToken = refreshToken
        self.tokenType = tokenType
        self.expiresIn = expiresIn
        self.obtainedAtEpochSeconds = obtainedAtEpochSeconds
        self.issuer = issuer
        self.tokenEndpoint = tokenEndpoint
        self.clientID = clientID
        self.scope = scope
    }

    public func expiresAtEpochSeconds() -> Int64? {
        Self.jwtExpiration(accessToken) ?? obtainedAtEpochSeconds.flatMap { obtained in
            expiresIn.map { obtained + $0 }
        }
    }

    public func isFresh(now: Int64, skewSeconds: Int64 = 60) -> Bool {
        expiresAtEpochSeconds().map { $0 > now + skewSeconds } ?? true
    }

    public func withConfiguration(_ configuration: RelayAuthConfiguration, obtainedAt: Int64) -> RelayOAuthTokenSet {
        RelayOAuthTokenSet(
            accessToken: accessToken,
            refreshToken: refreshToken,
            tokenType: tokenType,
            expiresIn: expiresIn,
            obtainedAtEpochSeconds: obtainedAt,
            issuer: configuration.issuer,
            tokenEndpoint: configuration.tokenEndpoint,
            clientID: configuration.clientID,
            scope: configuration.scope
        )
    }

    /// The `email` claim of the access token, when it is a JWT. Lets clients
    /// show which account a pairing belongs to — fleet visibility is scoped
    /// by account, so a surprising empty fleet is usually a wrong account.
    public var accountEmail: String? {
        let pieces = accessToken.split(separator: ".", omittingEmptySubsequences: false)
        guard pieces.count == 3 else { return nil }
        var payload = String(pieces[1]).replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        payload.append(String(repeating: "=", count: (4 - payload.count % 4) % 4))
        guard let data = Data(base64Encoded: payload),
              let claims = try? JSONDecoder().decode(RelayJWTEmailPayload.self, from: data) else {
            return nil
        }
        return claims.email
    }

    private static func jwtExpiration(_ token: String) -> Int64? {
        let pieces = token.split(separator: ".", omittingEmptySubsequences: false)
        guard pieces.count == 3 else { return nil }
        var payload = String(pieces[1]).replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        payload.append(String(repeating: "=", count: (4 - payload.count % 4) % 4))
        guard let data = Data(base64Encoded: payload),
              let expiration = try? JSONDecoder().decode(RelayJWTPayload.self, from: data).exp else {
            return nil
        }
        return expiration
    }
}

public struct RelayAuthStatusResult: Codable, Equatable, Sendable {
    public let executor: String
    public let credentialKind: String
    public let authenticated: Bool
    public let refreshable: Bool
    public let expiresAtEpochSeconds: Int64?
    public let tokenFile: String?

    enum CodingKeys: String, CodingKey {
        case executor
        case credentialKind = "credential_kind"
        case authenticated
        case refreshable
        case expiresAtEpochSeconds = "expires_at_epoch_seconds"
        case tokenFile = "token_file"
    }

    public init(
        executor: String,
        credentialKind: String,
        authenticated: Bool,
        refreshable: Bool,
        expiresAtEpochSeconds: Int64?,
        tokenFile: String?
    ) {
        self.executor = executor
        self.credentialKind = credentialKind
        self.authenticated = authenticated
        self.refreshable = refreshable
        self.expiresAtEpochSeconds = expiresAtEpochSeconds
        self.tokenFile = tokenFile
    }
}

public struct RelayLoginResult: Codable, Equatable, Sendable {
    public let executor: String
    public let issuer: String
    public let tokenFile: String
    public let refreshable: Bool
    public let expiresAtEpochSeconds: Int64?

    enum CodingKeys: String, CodingKey {
        case executor
        case issuer
        case tokenFile = "token_file"
        case refreshable
        case expiresAtEpochSeconds = "expires_at_epoch_seconds"
    }

    public init(
        executor: String,
        issuer: String,
        tokenFile: String,
        refreshable: Bool,
        expiresAtEpochSeconds: Int64?
    ) {
        self.executor = executor
        self.issuer = issuer
        self.tokenFile = tokenFile
        self.refreshable = refreshable
        self.expiresAtEpochSeconds = expiresAtEpochSeconds
    }
}

public struct RelayLogoutResult: Codable, Equatable, Sendable {
    public let executor: String
    public let removed: Bool

    public init(executor: String, removed: Bool) {
        self.executor = executor
        self.removed = removed
    }
}

public struct RelayResolvedCredential: Sendable {
    public let accessToken: String
    public let refreshable: Bool

    public init(accessToken: String, refreshable: Bool) {
        self.accessToken = accessToken
        self.refreshable = refreshable
    }
}

public struct RelayDeviceAuthorization: Decodable, Sendable {
    public let deviceCode: String
    public let userCode: String
    public let verificationURI: String
    public let verificationURIComplete: String?
    public let interval: Int64?
    public let expiresIn: Int64?

    enum CodingKeys: String, CodingKey {
        case deviceCode = "device_code"
        case userCode = "user_code"
        case verificationURI = "verification_uri"
        case verificationURIComplete = "verification_uri_complete"
        case interval
        case expiresIn = "expires_in"
    }

    public init(
        deviceCode: String,
        userCode: String,
        verificationURI: String,
        verificationURIComplete: String?,
        interval: Int64?,
        expiresIn: Int64?
    ) {
        self.deviceCode = deviceCode
        self.userCode = userCode
        self.verificationURI = verificationURI
        self.verificationURIComplete = verificationURIComplete
        self.interval = interval
        self.expiresIn = expiresIn
    }
}

private struct RelayOAuthTokenResponse: Decodable, Sendable {
    let accessToken: String?
    let refreshToken: String?
    let tokenType: String?
    let expiresIn: Int64?
    let error: String?
    let errorDescription: String?

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case refreshToken = "refresh_token"
        case tokenType = "token_type"
        case expiresIn = "expires_in"
        case error
        case errorDescription = "error_description"
    }
}

private struct RelayJWTPayload: Decodable {
    let exp: Int64?
}

private struct RelayJWTEmailPayload: Decodable {
    let email: String?
}

private struct RelayDeviceAuthorizationRequest: Encodable {
    let clientID: String
    let scope: String

    enum CodingKeys: String, CodingKey {
        case clientID = "client_id"
        case scope
    }
}

private struct RelayTokenRequest: Encodable {
    let grantType: String
    let deviceCode: String?
    let refreshToken: String?
    let clientID: String

    enum CodingKeys: String, CodingKey {
        case grantType = "grant_type"
        case deviceCode = "device_code"
        case refreshToken = "refresh_token"
        case clientID = "client_id"
    }
}

private final class RelayCredentialFileLock: @unchecked Sendable {
    private let descriptor: Int32

    init(tokenFile: URL) throws {
        let lockPath = tokenFile.path + ".lock"
        descriptor = lockPath.withCString { open($0, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR) }
        guard descriptor >= 0 else {
            throw RelayClientError("Could not create relay credential lock file.")
        }
        guard flock(descriptor, LOCK_EX) == 0 else {
            close(descriptor)
            throw RelayClientError("Could not lock relay credentials for refresh.")
        }
        _ = fchmod(descriptor, S_IRUSR | S_IWUSR)
    }

    deinit {
        _ = flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}

public enum RelayAuthentication {
    public typealias HTTPRequester = @Sendable (URLRequest) async throws -> (Data, HTTPURLResponse)
    public typealias Sleeper = @Sendable (UInt64) async throws -> Void

    /// Token file inside the caller-supplied application-support base. The CLI
    /// passes its store root; app clients pass their sandbox container.
    public static func defaultTokenFile(profileName: String, applicationSupportBase: URL) -> URL {
        applicationSupportBase
            .appendingPathComponent("executor-auth", isDirectory: true)
            .appendingPathComponent("\(profileName).json")
    }

    public static func discover(
        profile: WorkflowExecutorProfile,
        requester: HTTPRequester = send
    ) async throws -> RelayAuthConfiguration {
        guard let baseURL = profile.url,
              let url = URL(string: "\(baseURL)/.well-known/mere-run-relay") else {
            throw RelayClientError("Relay executor profile has an invalid URL.")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await requester(request)
        guard (200..<300).contains(response.statusCode) else {
            throw RelayClientError("Relay authentication discovery failed with HTTP \(response.statusCode).")
        }
        let document = try WorkflowBundleCodec.decoder().decode(RelayDiscoveryDocument.self, from: data)
        guard document.schemaVersion == 1, document.kind == "mere.run/relay" else {
            throw RelayClientError("Relay returned an unsupported discovery contract.")
        }
        try validate(configuration: document.auth)
        return document.auth
    }

    public static func login(
        profile: WorkflowExecutorProfile,
        tokenFile: URL,
        progress: @Sendable (String) -> Void,
        requester: HTTPRequester = send,
        sleeper: Sleeper = sleep
    ) async throws -> RelayLoginResult {
        let configuration = try await discover(profile: profile, requester: requester)
        let authorization = try await beginDeviceAuthorization(
            configuration: configuration,
            requester: requester
        )
        let approvalURL = authorization.verificationURIComplete ?? authorization.verificationURI
        progress("Open \(approvalURL) and enter code \(authorization.userCode).")
        let tokenSet = try await pollDeviceAuthorization(
            authorization,
            configuration: configuration,
            requester: requester,
            sleeper: sleeper
        )
        try save(tokenSet, to: tokenFile)
        return RelayLoginResult(
            executor: "relay:\(profile.name)",
            issuer: configuration.issuer,
            tokenFile: tokenFile.path,
            refreshable: tokenSet.refreshToken?.isEmpty == false,
            expiresAtEpochSeconds: tokenSet.expiresAtEpochSeconds()
        )
    }

    public static func resolveCredential(
        profile: WorkflowExecutorProfile,
        forceRefresh: Bool = false,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        now: @Sendable () -> Int64 = currentEpochSeconds,
        requester: HTTPRequester = send
    ) async throws -> RelayResolvedCredential {
        if let tokenFile = profile.tokenFile {
            let url = URL(fileURLWithPath: tokenFile)
            if FileManager.default.fileExists(atPath: url.path) {
                let data = try Data(contentsOf: url)
                if let tokenSet = try? WorkflowBundleCodec.decoder().decode(RelayOAuthTokenSet.self, from: data) {
                    if !forceRefresh, tokenSet.isFresh(now: now()) {
                        return RelayResolvedCredential(
                            accessToken: tokenSet.accessToken,
                            refreshable: tokenSet.refreshToken?.isEmpty == false
                        )
                    }
                    return try await refreshLocked(
                        tokenFile: url,
                        profile: profile,
                        forceRefresh: forceRefresh,
                        now: now,
                        requester: requester
                    )
                }
                let rawToken = String(decoding: data, as: UTF8.self)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !rawToken.isEmpty {
                    return RelayResolvedCredential(accessToken: rawToken, refreshable: false)
                }
            }
        }
        if let token = environment["MERERUN_RELAY_TOKEN"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !token.isEmpty {
            return RelayResolvedCredential(accessToken: token, refreshable: false)
        }
        throw RelayClientError(
            "Relay authentication requires `executor login`, --token-file, or MERERUN_RELAY_TOKEN."
        )
    }

    public static func status(
        profile: WorkflowExecutorProfile,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        now: Int64 = currentEpochSeconds()
    ) -> RelayAuthStatusResult {
        if let tokenFile = profile.tokenFile,
           let data = try? Data(contentsOf: URL(fileURLWithPath: tokenFile)) {
            if let tokenSet = try? WorkflowBundleCodec.decoder().decode(RelayOAuthTokenSet.self, from: data) {
                return RelayAuthStatusResult(
                    executor: "relay:\(profile.name)",
                    credentialKind: "oauth-token-set",
                    authenticated: tokenSet.isFresh(now: now, skewSeconds: 0),
                    refreshable: tokenSet.refreshToken?.isEmpty == false,
                    expiresAtEpochSeconds: tokenSet.expiresAtEpochSeconds(),
                    tokenFile: tokenFile
                )
            }
            let raw = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if !raw.isEmpty {
                return RelayAuthStatusResult(
                    executor: "relay:\(profile.name)",
                    credentialKind: "bearer-token-file",
                    authenticated: true,
                    refreshable: false,
                    expiresAtEpochSeconds: nil,
                    tokenFile: tokenFile
                )
            }
        }
        let environmentToken = environment["MERERUN_RELAY_TOKEN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return RelayAuthStatusResult(
            executor: "relay:\(profile.name)",
            credentialKind: environmentToken?.isEmpty == false ? "environment" : "none",
            authenticated: environmentToken?.isEmpty == false,
            refreshable: false,
            expiresAtEpochSeconds: nil,
            tokenFile: profile.tokenFile
        )
    }

    public static func save(
        _ tokenSet: RelayOAuthTokenSet,
        to url: URL,
        fileManager: FileManager = .default
    ) throws {
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try WorkflowBundleCodec.encoder().encode(tokenSet)
        try data.write(to: url, options: .atomic)
        try fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    public static func beginDeviceAuthorization(
        configuration: RelayAuthConfiguration,
        requester: HTTPRequester = send
    ) async throws -> RelayDeviceAuthorization {
        guard let url = URL(string: configuration.deviceAuthorizationEndpoint) else {
            throw RelayClientError("Relay device authorization endpoint is invalid.")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(configuration.issuer, forHTTPHeaderField: "Origin")
        request.httpBody = try WorkflowBundleCodec.encoder().encode(
            RelayDeviceAuthorizationRequest(clientID: configuration.clientID, scope: configuration.scope)
        )
        let (data, response) = try await requester(request)
        guard (200..<300).contains(response.statusCode) else {
            throw RelayClientError("Relay device sign-in failed with HTTP \(response.statusCode).")
        }
        return try WorkflowBundleCodec.decoder().decode(RelayDeviceAuthorization.self, from: data)
    }

    public static func pollDeviceAuthorization(
        _ authorization: RelayDeviceAuthorization,
        configuration: RelayAuthConfiguration,
        now: @Sendable () -> Int64 = currentEpochSeconds,
        requester: HTTPRequester = send,
        sleeper: Sleeper = sleep
    ) async throws -> RelayOAuthTokenSet {
        guard let url = URL(string: configuration.tokenEndpoint) else {
            throw RelayClientError("Relay token endpoint is invalid.")
        }
        let startedAt = now()
        let expiresAt = startedAt + max(authorization.expiresIn ?? 600, 1)
        var interval = max(authorization.interval ?? 5, 1)
        while now() < expiresAt {
            try await sleeper(UInt64(interval))
            let (data, response) = try await requestToken(
                url: url,
                issuer: configuration.issuer,
                body: RelayTokenRequest(
                    grantType: "urn:ietf:params:oauth:grant-type:device_code",
                    deviceCode: authorization.deviceCode,
                    refreshToken: nil,
                    clientID: configuration.clientID
                ),
                requester: requester
            )
            let payload = try decodeTokenResponse(data, status: response.statusCode)
            if let accessToken = payload.accessToken {
                return RelayOAuthTokenSet(
                    accessToken: accessToken,
                    refreshToken: payload.refreshToken,
                    tokenType: payload.tokenType,
                    expiresIn: payload.expiresIn,
                    obtainedAtEpochSeconds: now(),
                    issuer: configuration.issuer,
                    tokenEndpoint: configuration.tokenEndpoint,
                    clientID: configuration.clientID,
                    scope: configuration.scope
                )
            }
            switch payload.error {
            case "authorization_pending": continue
            case "slow_down": interval += 5
            case "access_denied": throw RelayClientError("Relay device sign-in was denied.")
            case "expired_token": throw RelayClientError("Relay device sign-in expired.")
            default:
                throw tokenError(payload, status: response.statusCode)
            }
        }
        throw RelayClientError("Relay device sign-in expired before approval.")
    }

    /// Storage-backed resolution for app clients: loads from the supplied
    /// storage, refreshing (and saving back) when stale.
    public static func resolveCredential(
        profile: WorkflowExecutorProfile,
        storage: any RelayCredentialStorage,
        forceRefresh: Bool = false,
        now: @Sendable () -> Int64 = currentEpochSeconds,
        requester: HTTPRequester = send
    ) async throws -> RelayResolvedCredential {
        guard let current = try storage.load() else {
            throw RelayClientError("No saved relay credential. Sign in again.")
        }
        if !forceRefresh, current.isFresh(now: now()) {
            return RelayResolvedCredential(
                accessToken: current.accessToken,
                refreshable: current.refreshToken?.isEmpty == false
            )
        }
        let refreshed = try await refreshedTokenSet(
            current: current,
            profile: profile,
            now: now,
            requester: requester
        )
        try storage.save(refreshed)
        return RelayResolvedCredential(accessToken: refreshed.accessToken, refreshable: true)
    }

    private static func refreshLocked(
        tokenFile: URL,
        profile: WorkflowExecutorProfile,
        forceRefresh: Bool,
        now: @Sendable () -> Int64,
        requester: HTTPRequester
    ) async throws -> RelayResolvedCredential {
        let lock = try RelayCredentialFileLock(tokenFile: tokenFile)
        defer { _ = lock }
        let current = try WorkflowBundleCodec.decoder().decode(
            RelayOAuthTokenSet.self,
            from: Data(contentsOf: tokenFile)
        )
        if !forceRefresh, current.isFresh(now: now()) {
            return RelayResolvedCredential(
                accessToken: current.accessToken,
                refreshable: current.refreshToken?.isEmpty == false
            )
        }
        let refreshed = try await refreshedTokenSet(
            current: current,
            profile: profile,
            now: now,
            requester: requester
        )
        try save(refreshed, to: tokenFile)
        return RelayResolvedCredential(accessToken: refreshed.accessToken, refreshable: true)
    }

    private static func refreshedTokenSet(
        current: RelayOAuthTokenSet,
        profile: WorkflowExecutorProfile,
        now: @Sendable () -> Int64,
        requester: HTTPRequester
    ) async throws -> RelayOAuthTokenSet {
        guard let refreshToken = current.refreshToken, !refreshToken.isEmpty else {
            throw RelayClientError("Relay session expired. Run `mere.run executor login relay:\(profile.name)`." )
        }
        let configuration: RelayAuthConfiguration
        if let issuer = current.issuer,
           let tokenEndpoint = current.tokenEndpoint,
           let clientID = current.clientID {
            configuration = RelayAuthConfiguration(
                issuer: issuer,
                deviceAuthorizationEndpoint: issuer.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                    + "/oauth/device_authorization",
                tokenEndpoint: tokenEndpoint,
                clientID: clientID,
                scope: current.scope ?? "openid profile email offline_access"
            )
        } else {
            configuration = try await discover(profile: profile, requester: requester)
        }
        guard let url = URL(string: configuration.tokenEndpoint) else {
            throw RelayClientError("Relay token endpoint is invalid.")
        }
        let (data, response) = try await requestToken(
            url: url,
            issuer: configuration.issuer,
            body: RelayTokenRequest(
                grantType: "refresh_token",
                deviceCode: nil,
                refreshToken: refreshToken,
                clientID: configuration.clientID
            ),
            requester: requester
        )
        let payload = try decodeTokenResponse(data, status: response.statusCode)
        guard let accessToken = payload.accessToken else {
            throw tokenError(payload, status: response.statusCode)
        }
        return RelayOAuthTokenSet(
            accessToken: accessToken,
            refreshToken: payload.refreshToken ?? refreshToken,
            tokenType: payload.tokenType ?? current.tokenType,
            expiresIn: payload.expiresIn ?? current.expiresIn,
            obtainedAtEpochSeconds: now(),
            issuer: configuration.issuer,
            tokenEndpoint: configuration.tokenEndpoint,
            clientID: configuration.clientID,
            scope: configuration.scope
        )
    }

    private static func requestToken(
        url: URL,
        issuer: String,
        body: RelayTokenRequest,
        requester: HTTPRequester
    ) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(issuer, forHTTPHeaderField: "Origin")
        request.httpBody = try WorkflowBundleCodec.encoder().encode(body)
        return try await requester(request)
    }

    private static func decodeTokenResponse(_ data: Data, status: Int) throws -> RelayOAuthTokenResponse {
        do {
            return try WorkflowBundleCodec.decoder().decode(RelayOAuthTokenResponse.self, from: data)
        } catch {
            throw RelayClientError("Relay token endpoint returned invalid JSON with HTTP \(status).")
        }
    }

    private static func tokenError(_ payload: RelayOAuthTokenResponse, status: Int) -> RelayClientError {
        let reason = payload.errorDescription ?? payload.error ?? "missing_access_token"
        return RelayClientError("Relay token request failed with HTTP \(status): \(reason)")
    }

    private static func validate(configuration: RelayAuthConfiguration) throws {
        guard let issuer = URL(string: configuration.issuer),
              let deviceEndpoint = URL(string: configuration.deviceAuthorizationEndpoint),
              let tokenEndpoint = URL(string: configuration.tokenEndpoint),
              !configuration.clientID.isEmpty,
              !configuration.scope.isEmpty,
              issuer.scheme == "https" || isLoopback(issuer),
              deviceEndpoint.scheme == "https" || isLoopback(deviceEndpoint),
              tokenEndpoint.scheme == "https" || isLoopback(tokenEndpoint) else {
            throw RelayClientError("Relay returned an invalid authentication configuration.")
        }
    }

    private static func isLoopback(_ url: URL) -> Bool {
        url.scheme == "http" && ["127.0.0.1", "localhost", "::1"].contains(url.host)
    }

    public static func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw RelayClientError("Relay authentication endpoint returned a non-HTTP response.")
        }
        return (data, http)
    }

    public static func sleep(seconds: UInt64) async throws {
        try await Task<Never, Never>.sleep(nanoseconds: seconds * 1_000_000_000)
    }

    public static func currentEpochSeconds() -> Int64 {
        Int64(Date().timeIntervalSince1970)
    }
}
