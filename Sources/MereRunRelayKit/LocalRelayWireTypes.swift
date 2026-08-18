import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Wire types for the direct lane: `mere.run relay serve` hosts the relay
/// HTTP surface on the machine itself, and a client pairs straight to it over
/// the LAN or a tailnet with a locally issued bearer token — no broker, no
/// cloud hop. The job API is byte-identical to the hosted relay's; only
/// discovery and pairing differ.
public struct LocalRelayDiscoveryDocument: Codable, Equatable, Sendable {
    public static let kind = "mere.run/relay"
    public static let authMode = "local-pair"

    public let schemaVersion: Int
    public let kind: String
    public let authMode: String
    public let relayName: String
    public let contractVersions: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case authMode = "auth_mode"
        case relayName = "relay_name"
        case contractVersions = "graph_contract_versions"
    }

    public init(schemaVersion: Int, kind: String, authMode: String, relayName: String, contractVersions: [String]) {
        self.schemaVersion = schemaVersion
        self.kind = kind
        self.authMode = authMode
        self.relayName = relayName
        self.contractVersions = contractVersions
    }
}

public struct LocalRelayPairRequest: Codable, Equatable, Sendable {
    public let code: String
    public let deviceName: String

    enum CodingKeys: String, CodingKey {
        case code
        case deviceName = "device_name"
    }

    public init(code: String, deviceName: String) {
        self.code = code
        self.deviceName = deviceName
    }
}

public struct LocalRelayPairResponse: Codable, Equatable, Sendable {
    public let token: String
    public let relayName: String

    enum CodingKeys: String, CodingKey {
        case token
        case relayName = "relay_name"
    }

    public init(token: String, relayName: String) {
        self.token = token
        self.relayName = relayName
    }
}

extension RelayAuthentication {
    /// Fetches direct-lane discovery from a `relay serve` host and verifies
    /// the document identifies a local-pairing relay.
    public static func discoverLocalRelay(
        url baseURL: String,
        requester: HTTPRequester = send
    ) async throws -> LocalRelayDiscoveryDocument {
        guard let url = URL(string: "\(baseURL)/.well-known/mere-run-relay") else {
            throw RelayClientError("The relay address is not a valid URL.")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        let (data, response) = try await requester(request)
        guard response.statusCode == 200 else {
            throw RelayClientError("The relay did not answer discovery (HTTP \(response.statusCode)).")
        }
        guard let document = try? WorkflowBundleCodec.decoder().decode(LocalRelayDiscoveryDocument.self, from: data),
              document.kind == LocalRelayDiscoveryDocument.kind,
              document.authMode == LocalRelayDiscoveryDocument.authMode else {
            throw RelayClientError(
                "That address is not a direct mere.run relay. For a hosted relay, use the sign-in flow instead."
            )
        }
        return document
    }

    /// Exchanges a pairing code shown by `mere.run relay serve` for a
    /// long-lived bearer token.
    public static func pairLocalRelay(
        url baseURL: String,
        code: String,
        deviceName: String,
        requester: HTTPRequester = send
    ) async throws -> LocalRelayPairResponse {
        guard let url = URL(string: "\(baseURL)/api/pair") else {
            throw RelayClientError("The relay address is not a valid URL.")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try WorkflowBundleCodec.encoder().encode(
            LocalRelayPairRequest(code: code, deviceName: deviceName)
        )
        let (data, response) = try await requester(request)
        guard response.statusCode == 200 else {
            let detail = String(decoding: data, as: UTF8.self)
            if response.statusCode == 401 || response.statusCode == 403 {
                throw RelayClientError("The pairing code was not accepted. Check the code shown in the terminal.")
            }
            throw RelayClientError("Pairing failed (HTTP \(response.statusCode)): \(detail)")
        }
        return try WorkflowBundleCodec.decoder().decode(LocalRelayPairResponse.self, from: data)
    }
}
