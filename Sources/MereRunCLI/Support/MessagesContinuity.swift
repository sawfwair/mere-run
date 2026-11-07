import Foundation
import Crypto
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

/// Locally authenticated continuity, deliberately distinct from hosted signatures.
struct MessagesContinuity: Sendable {
    private let key: SymmetricKey
    init(key: Data) throws {
        guard key.count == 32 else { throw MessagesFailure.invalid("signature", "installation key must contain 32 bytes") }
        self.key = SymmetricKey(data: key)
    }

    static func installation(url: URL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".mere/api/reasoning.key")) throws -> Self {
        let manager = FileManager.default
        try manager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
                                    attributes: [.posixPermissions: 0o700])
        let lockURL = url.deletingLastPathComponent().appendingPathComponent("reasoning.key.lock")
        let lockDescriptor = open(lockURL.path, O_RDWR | O_CREAT | O_NOFOLLOW, 0o600)
        guard lockDescriptor >= 0 else { throw MessagesFailure.invalid("signature", "could not open the installation key lock") }
        defer { close(lockDescriptor) }
        guard flock(lockDescriptor, LOCK_EX) == 0 else { throw MessagesFailure.invalid("signature", "could not acquire the installation key lock") }
        defer { flock(lockDescriptor, LOCK_UN) }
        if !manager.fileExists(atPath: url.path) {
            let bytes = SymmetricKey(size: .bits256).withUnsafeBytes { Data($0) }
            let descriptor = open(url.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
            if descriptor >= 0 {
                defer { close(descriptor) }
                let written = bytes.withUnsafeBytes { buffer in write(descriptor, buffer.baseAddress, buffer.count) }
                guard written == bytes.count else { throw MessagesFailure.invalid("signature", "could not persist the installation key") }
            } else if errno != EEXIST {
                throw MessagesFailure.invalid("signature", "could not create the installation key")
            }
        }
        return try Self(key: Data(contentsOf: url))
    }

    func sign(thinking: String, model: String) -> String {
        let payload = Data("mere-reasoning-v1\u{0}\(model)\u{0}\(thinking)".utf8)
        let tag = HMAC<SHA256>.authenticationCode(for: payload, using: key)
        return "mere.v1." + Data(tag).base64EncodedString()
    }
    func verify(signature: String, thinking: String, model: String) throws {
        guard signature.hasPrefix("mere.v1."), let tag = Data(base64Encoded: String(signature.dropFirst(8))),
              HMAC<SHA256>.isValidAuthenticationCode(tag,
                  authenticating: Data("mere-reasoning-v1\u{0}\(model)\u{0}\(thinking)".utf8), using: key) else {
            throw MessagesFailure.invalid("thinking.signature", "foreign, modified or invalid local reasoning continuity token")
        }
    }
}
