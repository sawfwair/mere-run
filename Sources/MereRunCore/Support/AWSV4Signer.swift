import Foundation
import CryptoKit

// MARK: - AWS V4 Signature (R2 direct download)

/// Signs requests for AWS S3-compatible services using AWS Signature Version 4.
/// Used for direct downloads from Cloudflare R2 storage.
public enum AWSV4Signer {
    public static func sign(
        method: String,
        url: URL,
        region: String,
        service: String,
        accessKey: String,
        secretKey: String,
        date: Date = Date()
    ) -> [String: String] {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        let amzDate = formatter.string(from: date)

        formatter.dateFormat = "yyyyMMdd"
        let dateStamp = formatter.string(from: date)

        let host = url.host ?? ""
        let canonicalUri = url.path.isEmpty ? "/" : url.path
        let canonicalQueryString = ""
        let payloadHash = sha256("")

        let canonicalHeaders = "host:\(host)\nx-amz-content-sha256:\(payloadHash)\nx-amz-date:\(amzDate)\n"
        let signedHeaders = "host;x-amz-content-sha256;x-amz-date"

        let canonicalRequest = "\(method)\n\(canonicalUri)\n\(canonicalQueryString)\n\(canonicalHeaders)\n\(signedHeaders)\n\(payloadHash)"

        let credentialScope = "\(dateStamp)/\(region)/\(service)/aws4_request"
        let stringToSign = "AWS4-HMAC-SHA256\n\(amzDate)\n\(credentialScope)\n\(sha256(canonicalRequest))"

        let signingKey = getSignatureKey(key: secretKey, dateStamp: dateStamp, region: region, service: service)
        let signature = hmacSHA256(data: stringToSign, key: signingKey).map { String(format: "%02x", $0) }.joined()

        let authorization = "AWS4-HMAC-SHA256 Credential=\(accessKey)/\(credentialScope), SignedHeaders=\(signedHeaders), Signature=\(signature)"

        return [
            "Authorization": authorization,
            "x-amz-date": amzDate,
            "x-amz-content-sha256": payloadHash
        ]
    }

    private static func sha256(_ string: String) -> String {
        let data = Data(string.utf8)
        let hash = SHA256.hash(data: data)
        return hash.map { String(format: "%02x", $0) }.joined()
    }

    private static func hmacSHA256(data: String, key: Data) -> Data {
        let key = SymmetricKey(data: key)
        let signature = HMAC<SHA256>.authenticationCode(for: Data(data.utf8), using: key)
        return Data(signature)
    }

    private static func getSignatureKey(key: String, dateStamp: String, region: String, service: String) -> Data {
        let kDate = hmacSHA256(data: dateStamp, key: Data("AWS4\(key)".utf8))
        let kRegion = hmacSHA256(data: region, key: kDate)
        let kService = hmacSHA256(data: service, key: kRegion)
        let kSigning = hmacSHA256(data: "aws4_request", key: kService)
        return kSigning
    }
}

// MARK: - R2 Request Builder

public enum R2DownloadRequestBuilder {
    public enum Mode: String, Sendable {
        case signedURL
        case signed
        case publicBaseURL
    }

    public struct BuildResult: Sendable {
        public let request: URLRequest
        public let mode: Mode

        public init(request: URLRequest, mode: Mode) {
            self.request = request
            self.mode = mode
        }
    }

    public enum BuildError: LocalizedError, Sendable {
        case invalidKey(String)
        case invalidBaseURL(String)
        case invalidSignedURLEndpoint(String)
        case invalidSignedURLResponse
        case signedURLLookupFailed(String)
        case missingConfiguration(String)

        public var errorDescription: String? {
            switch self {
            case .invalidKey(let key):
                return "Invalid R2 object key: \(key)"
            case .invalidBaseURL(let base):
                return "Invalid model source base URL: \(base)"
            case .invalidSignedURLEndpoint(let endpoint):
                return "Invalid R2 signed URL endpoint: \(endpoint)"
            case .invalidSignedURLResponse:
                return "Signed URL endpoint returned an invalid response"
            case .signedURLLookupFailed(let message):
                return "Signed URL lookup failed: \(message)"
            case .missingConfiguration(let message):
                return message
            }
        }
    }

    /// Request strategy precedence:
    /// 1) `MERERUN_R2_SIGNED_URL_ENDPOINT` / `R2_SIGNED_URL_ENDPOINT` (optional control-plane signer)
    /// 2) Direct SigV4 with env creds (`MERERUN_R2_*` / `R2_*` / `AWS_*`)
    /// 3) Explicit public archive base URL via `MERERUN_MODEL_SOURCE_BASE_URL`
    public static func makeGETRequest(
        key: String,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        defaultPublicBaseURL: String? = nil
    ) async throws -> BuildResult {
        let trimmedKey = key.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedKey = trimmedKey.hasPrefix("/") ? String(trimmedKey.dropFirst()) : trimmedKey
        guard !normalizedKey.isEmpty else {
            throw BuildError.invalidKey(key)
        }

        if let signedURLEndpoint = firstNonEmpty(environment, keys: ["MERERUN_R2_SIGNED_URL_ENDPOINT", "R2_SIGNED_URL_ENDPOINT"]) {
            let requireSignedURL = boolFlag(environment, keys: ["MERERUN_R2_SIGNED_URL_REQUIRED", "R2_SIGNED_URL_REQUIRED"])
            do {
                return try await makeSignedURLRequest(
                    key: normalizedKey,
                    endpoint: signedURLEndpoint,
                    environment: environment
                )
            } catch {
                if requireSignedURL {
                    if let buildError = error as? BuildError {
                        throw buildError
                    } else {
                        throw BuildError.signedURLLookupFailed(error.localizedDescription)
                    }
                }
            }
        }

        let accountId = firstNonEmpty(environment, keys: ["MERERUN_R2_ACCOUNT_ID", "R2_ACCOUNT_ID"])
        let accessKey = firstNonEmpty(environment, keys: ["MERERUN_R2_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"])
        let secretKey = firstNonEmpty(environment, keys: ["MERERUN_R2_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"])
        let bucket = firstNonEmpty(environment, keys: ["MERERUN_R2_BUCKET", "R2_BUCKET"]) ?? "public"

        if let accountId, let accessKey, let secretKey {
            let endpoint = "https://\(accountId).r2.cloudflarestorage.com"
            guard let url = URL(string: "\(endpoint)/\(bucket)/\(normalizedKey)") else {
                throw BuildError.invalidKey(normalizedKey)
            }

            let headers = AWSV4Signer.sign(
                method: "GET",
                url: url,
                region: "auto",
                service: "s3",
                accessKey: accessKey,
                secretKey: secretKey
            )

            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            for (header, value) in headers {
                request.setValue(value, forHTTPHeaderField: header)
            }
            return BuildResult(request: request, mode: .signed)
        }

        let baseURL: URL
        if let configuredBaseURL = MereRunModelSourceConfiguration.publicBaseURL(environment: environment) {
            baseURL = configuredBaseURL
        } else if let defaultPublicBaseURL {
            guard let resolved = URL(string: defaultPublicBaseURL) else {
                throw BuildError.invalidBaseURL(defaultPublicBaseURL)
            }
            baseURL = resolved
        } else {
            throw BuildError.missingConfiguration(
                MereRunModelSourceConfiguration.missingConfigurationMessage()
            )
        }
        guard let url = URL(string: normalizedKey, relativeTo: baseURL.appendingPathComponent(""))?.absoluteURL else {
            throw BuildError.invalidKey(normalizedKey)
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        return BuildResult(request: request, mode: .publicBaseURL)
    }

    private struct SignedURLLookupRequest: Encodable {
        let key: String
        let method: String
    }

    private struct SignedURLLookupResponse: Decodable {
        let url: String
        let method: String?
        let headers: [String: String]?
    }

    private static func makeSignedURLRequest(
        key: String,
        endpoint: String,
        environment: [String: String]
    ) async throws -> BuildResult {
        guard let endpointURL = URL(string: endpoint),
              let scheme = endpointURL.scheme?.lowercased(),
              (scheme == "https" || scheme == "http"),
              endpointURL.host != nil else {
            throw BuildError.invalidSignedURLEndpoint(endpoint)
        }

        var lookupRequest = URLRequest(url: endpointURL)
        lookupRequest.httpMethod = "POST"
        lookupRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        lookupRequest.httpBody = try JSONEncoder().encode(
            SignedURLLookupRequest(key: key, method: "GET")
        )

        if let token = firstNonEmpty(environment, keys: ["MERERUN_R2_SIGNED_URL_BEARER_TOKEN", "R2_SIGNED_URL_BEARER_TOKEN"]) {
            lookupRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await URLSession.shared.data(for: lookupRequest)
        guard let http = response as? HTTPURLResponse else {
            throw BuildError.signedURLLookupFailed("Invalid HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw BuildError.signedURLLookupFailed("HTTP \(http.statusCode)")
        }

        let resolved = try parseSignedURLResponse(data: data)

        var request = URLRequest(url: resolved.url)
        request.httpMethod = resolved.method
        for (header, value) in resolved.headers {
            request.setValue(value, forHTTPHeaderField: header)
        }
        return BuildResult(request: request, mode: .signedURL)
    }

    private static func parseSignedURLResponse(data: Data) throws -> (url: URL, method: String, headers: [String: String]) {
        if let text = String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !text.isEmpty,
           let directURL = URL(string: text) {
            return (directURL, "GET", [:])
        }

        if let decoded = try? JSONDecoder().decode(SignedURLLookupResponse.self, from: data),
           let url = URL(string: decoded.url) {
            return (
                url,
                (decoded.method?.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()).flatMap { $0.isEmpty ? nil : $0 } ?? "GET",
                decoded.headers ?? [:]
            )
        }

        throw BuildError.invalidSignedURLResponse
    }

    private static func firstNonEmpty(_ environment: [String: String], keys: [String]) -> String? {
        for key in keys {
            guard let value = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
                continue
            }
            return value
        }
        return nil
    }

    private static func boolFlag(_ environment: [String: String], keys: [String]) -> Bool {
        guard let raw = firstNonEmpty(environment, keys: keys)?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }
}
