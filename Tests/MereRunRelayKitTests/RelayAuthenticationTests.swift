import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import MereRunRelayKit

final class RelayAuthenticationTests: XCTestCase {
    func testExpiredOAuthTokenSetRefreshesAtomically() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenFile = directory.appendingPathComponent("relay.json")
        let profile = relayProfile(tokenFile: tokenFile.path)
        try RelayAuthentication.save(
            RelayOAuthTokenSet(
                accessToken: jwt(expiration: 100),
                refreshToken: "refresh-old",
                tokenType: "Bearer",
                expiresIn: 900,
                obtainedAtEpochSeconds: 0,
                issuer: "https://mere.world",
                tokenEndpoint: "https://mere.world/oauth/token",
                clientID: "mererun-node",
                scope: "openid profile email offline_access"
            ),
            to: tokenFile
        )
        let refreshedAccessToken = jwt(expiration: 2_000)
        let recorder = RequestRecorder(responses: [
            .init(status: 200, body: """
            {
              "access_token": "\(refreshedAccessToken)",
              "refresh_token": "refresh-new",
              "token_type": "Bearer",
              "expires_in": 900
            }
            """),
        ])

        let credential = try await RelayAuthentication.resolveCredential(
            profile: profile,
            environment: [:],
            now: { 1_000 },
            requester: { request in try await recorder.request(request) }
        )

        XCTAssertEqual(credential.accessToken, refreshedAccessToken)
        XCTAssertTrue(credential.refreshable)
        let saved = try JSONDecoder().decode(RelayOAuthTokenSet.self, from: Data(contentsOf: tokenFile))
        XCTAssertEqual(saved.refreshToken, "refresh-new")
        XCTAssertEqual(saved.obtainedAtEpochSeconds, 1_000)
        let recordedRequests = await recorder.requests()
        let body = try XCTUnwrap(recordedRequests.first?.httpBody)
        let request = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: String])
        XCTAssertEqual(request["grant_type"], "refresh_token")
        XCTAssertEqual(request["client_id"], "mererun-node")
        XCTAssertEqual(request["refresh_token"], "refresh-old")
        let permissions = try FileManager.default.attributesOfItem(atPath: tokenFile.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(permissions?.intValue, 0o600)
    }

    func testNodeCompatibleTokenSetDiscoversRelayAuthBeforeRefresh() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenFile = directory.appendingPathComponent("relay.json")
        let profile = relayProfile(tokenFile: tokenFile.path)
        try RelayAuthentication.save(
            RelayOAuthTokenSet(
                accessToken: jwt(expiration: 100),
                refreshToken: "node-refresh",
                tokenType: "Bearer",
                expiresIn: 900,
                obtainedAtEpochSeconds: 0,
                issuer: nil,
                tokenEndpoint: nil,
                clientID: nil,
                scope: nil
            ),
            to: tokenFile
        )
        let accessToken = jwt(expiration: 3_000)
        let recorder = RequestRecorder(responses: [
            .init(status: 200, body: discoveryJSON),
            .init(status: 200, body: """
            {"access_token":"\(accessToken)","expires_in":900,"token_type":"Bearer"}
            """),
        ])

        let credential = try await RelayAuthentication.resolveCredential(
            profile: profile,
            environment: [:],
            now: { 1_000 },
            requester: { request in try await recorder.request(request) }
        )

        XCTAssertEqual(credential.accessToken, accessToken)
        let requests = await recorder.requests()
        XCTAssertEqual(requests.map(\.url?.path), ["/.well-known/mere-run-relay", "/oauth/token"])
        let saved = try JSONDecoder().decode(RelayOAuthTokenSet.self, from: Data(contentsOf: tokenFile))
        XCTAssertEqual(saved.issuer, "https://mere.world")
        XCTAssertEqual(saved.clientID, "mererun-node")
        XCTAssertEqual(saved.refreshToken, "node-refresh")
    }

    func testRawTokenFileAndEnvironmentRemainCompatible() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenFile = directory.appendingPathComponent("relay.token")
        try Data(" raw-bearer \n".utf8).write(to: tokenFile)

        let fileCredential = try await RelayAuthentication.resolveCredential(
            profile: relayProfile(tokenFile: tokenFile.path),
            environment: [:]
        )
        XCTAssertEqual(fileCredential.accessToken, "raw-bearer")
        XCTAssertFalse(fileCredential.refreshable)

        let environmentCredential = try await RelayAuthentication.resolveCredential(
            profile: relayProfile(tokenFile: nil),
            environment: ["MERERUN_RELAY_TOKEN": " environment-bearer "]
        )
        XCTAssertEqual(environmentCredential.accessToken, "environment-bearer")
        XCTAssertFalse(environmentCredential.refreshable)
    }

    func testDevicePollingHandlesPendingThenPersistsRefreshableToken() async throws {
        let accessToken = jwt(expiration: 4_000)
        let recorder = RequestRecorder(responses: [
            .init(status: 400, body: "{\"error\":\"authorization_pending\"}"),
            .init(status: 200, body: """
            {"access_token":"\(accessToken)","refresh_token":"refresh","expires_in":900}
            """),
        ])
        let authorization = RelayDeviceAuthorization(
            deviceCode: "device",
            userCode: "ABCD-EFGH",
            verificationURI: "https://mere.world/device",
            verificationURIComplete: nil,
            interval: 1,
            expiresIn: 600
        )

        let tokenSet = try await RelayAuthentication.pollDeviceAuthorization(
            authorization,
            configuration: authConfiguration,
            now: { 1_000 },
            requester: { request in try await recorder.request(request) },
            sleeper: { _ in }
        )

        XCTAssertEqual(tokenSet.accessToken, accessToken)
        XCTAssertEqual(tokenSet.refreshToken, "refresh")
        let requestCount = await recorder.requests().count
        XCTAssertEqual(requestCount, 2)
    }

    func testAuthStatusNeverIncludesCredentialMaterial() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tokenFile = directory.appendingPathComponent("relay.json")
        try RelayAuthentication.save(
            RelayOAuthTokenSet(
                accessToken: jwt(expiration: 2_000),
                refreshToken: "secret-refresh",
                tokenType: "Bearer",
                expiresIn: 900,
                obtainedAtEpochSeconds: 1_000,
                issuer: "https://mere.world",
                tokenEndpoint: "https://mere.world/oauth/token",
                clientID: "mererun-node",
                scope: "openid"
            ),
            to: tokenFile
        )

        let status = RelayAuthentication.status(
            profile: relayProfile(tokenFile: tokenFile.path),
            environment: [:],
            now: 1_100
        )
        let encoded = String(decoding: try JSONEncoder().encode(status), as: UTF8.self)

        XCTAssertTrue(status.authenticated)
        XCTAssertTrue(status.refreshable)
        XCTAssertFalse(encoded.contains("secret-refresh"))
        XCTAssertFalse(encoded.contains("access_token"))
    }

    private var discoveryJSON: String {
        """
        {
          "schema_version": 1,
          "kind": "mere.run/relay",
          "auth": {
            "issuer": "https://mere.world",
            "device_authorization_endpoint": "https://mere.world/oauth/device_authorization",
            "token_endpoint": "https://mere.world/oauth/token",
            "client_id": "mererun-node",
            "scope": "openid profile email offline_access"
          }
        }
        """
    }

    private var authConfiguration: RelayAuthConfiguration {
        RelayAuthConfiguration(
            issuer: "https://mere.world",
            deviceAuthorizationEndpoint: "https://mere.world/oauth/device_authorization",
            tokenEndpoint: "https://mere.world/oauth/token",
            clientID: "mererun-node",
            scope: "openid profile email offline_access"
        )
    }

    private func relayProfile(tokenFile: String?) -> WorkflowExecutorProfile {
        WorkflowExecutorProfile(
            name: "fleet",
            kind: .relay,
            destination: nil,
            remoteRoot: nil,
            port: nil,
            identityFile: nil,
            mereRunPath: nil,
            url: "https://relay.example",
            tokenFile: tokenFile
        )
    }

    private func jwt(expiration: Int64) -> String {
        let header = base64URL(Data(#"{"alg":"none"}"#.utf8))
        let payload = base64URL(Data("{\"exp\":\(expiration)}".utf8))
        return "\(header).\(payload).signature"
    }

    private func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private actor RequestRecorder {
    struct Response: Sendable {
        let status: Int
        let body: String
    }

    private var queued: [Response]
    private var recorded: [URLRequest] = []

    init(responses: [Response]) {
        queued = responses
    }

    func request(_ request: URLRequest) throws -> (Data, HTTPURLResponse) {
        recorded.append(request)
        guard !queued.isEmpty else {
            throw URLError(.badServerResponse)
        }
        let response = queued.removeFirst()
        let http = HTTPURLResponse(
            url: request.url!,
            statusCode: response.status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        return (Data(response.body.utf8), http)
    }

    func requests() -> [URLRequest] {
        recorded
    }
}
