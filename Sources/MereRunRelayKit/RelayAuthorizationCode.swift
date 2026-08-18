import Crypto
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Authorization Code + PKCE for native clients. The relay's discovery
/// document advertises `authorization_endpoint` when its broker supports the
/// flow; clients fall back to the device grant when it is absent.
public enum RelayAuthorizationCodeFlow {
    public struct Started: Sendable {
        public let authorizationURL: URL
        public let codeVerifier: String
        public let state: String
    }

    /// Builds the authorize URL with an S256 challenge and CSRF state.
    public static func begin(
        configuration: RelayAuthConfiguration,
        clientID: String,
        redirectURI: String
    ) throws -> Started {
        guard let endpoint = configuration.authorizationEndpoint,
              var components = URLComponents(string: endpoint) else {
            throw RelayClientError("The relay's broker does not offer browser sign-in; use the device code instead.")
        }
        let verifier = randomURLSafeToken(bytes: 48)
        let challenge = Data(SHA256.hash(data: Data(verifier.utf8))).base64URLEncoded()
        let state = randomURLSafeToken(bytes: 24)
        components.queryItems = (components.queryItems ?? []) + [
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "client_id", value: clientID),
            URLQueryItem(name: "redirect_uri", value: redirectURI),
            URLQueryItem(name: "scope", value: configuration.scope),
            URLQueryItem(name: "state", value: state),
            URLQueryItem(name: "code_challenge", value: challenge),
            URLQueryItem(name: "code_challenge_method", value: "S256"),
        ]
        guard let url = components.url else {
            throw RelayClientError("Relay authorization endpoint is invalid.")
        }
        return Started(authorizationURL: url, codeVerifier: verifier, state: state)
    }

    /// Validates the callback against the started flow and returns the code.
    public static func code(
        fromCallback callback: URL,
        started: Started
    ) throws -> String {
        let items = URLComponents(url: callback, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard items.first(where: { $0.name == "state" })?.value == started.state else {
            throw RelayClientError("Sign-in was interrupted; try again.")
        }
        if let error = items.first(where: { $0.name == "error" })?.value {
            throw RelayClientError("Sign-in was refused: \(error)")
        }
        guard let code = items.first(where: { $0.name == "code" })?.value, !code.isEmpty else {
            throw RelayClientError("Sign-in did not return an authorization code.")
        }
        return code
    }

    /// Exchanges the code + verifier for a token set at the broker.
    public static func exchange(
        code: String,
        started: Started,
        configuration: RelayAuthConfiguration,
        clientID: String,
        redirectURI: String,
        requester: RelayAuthentication.HTTPRequester = RelayAuthentication.send,
        now: @Sendable () -> Int64 = RelayAuthentication.currentEpochSeconds
    ) async throws -> RelayOAuthTokenSet {
        guard let url = URL(string: configuration.tokenEndpoint) else {
            throw RelayClientError("Relay token endpoint is invalid.")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.setValue(configuration.issuer, forHTTPHeaderField: "Origin")
        request.httpBody = Data(
            formEncoded([
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", redirectURI),
                ("client_id", clientID),
                ("code_verifier", started.codeVerifier),
            ]).utf8
        )
        let (data, response) = try await requester(request)
        guard (200..<300).contains(response.statusCode) else {
            throw RelayClientError("Sign-in failed with HTTP \(response.statusCode).")
        }
        struct TokenResponse: Decodable {
            let accessToken: String
            let refreshToken: String?
            let tokenType: String?
            let expiresIn: Int64?

            enum CodingKeys: String, CodingKey {
                case accessToken = "access_token"
                case refreshToken = "refresh_token"
                case tokenType = "token_type"
                case expiresIn = "expires_in"
            }
        }
        let token = try WorkflowBundleCodec.decoder().decode(TokenResponse.self, from: data)
        return RelayOAuthTokenSet(
            accessToken: token.accessToken,
            refreshToken: token.refreshToken,
            tokenType: token.tokenType,
            expiresIn: token.expiresIn,
            obtainedAtEpochSeconds: now(),
            issuer: configuration.issuer,
            tokenEndpoint: configuration.tokenEndpoint,
            clientID: clientID,
            scope: configuration.scope
        )
    }

    private static func randomURLSafeToken(bytes count: Int) -> String {
        var generator = SystemRandomNumberGenerator()
        let bytes = (0..<count).map { _ in UInt8.random(in: .min ... .max, using: &generator) }
        return Data(bytes).base64URLEncoded()
    }

    private static func formEncoded(_ pairs: [(String, String)]) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        return pairs.map { name, value in
            let encoded = value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
            return "\(name)=\(encoded)"
        }.joined(separator: "&")
    }
}

extension Data {
    func base64URLEncoded() -> String {
        base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}
