import Foundation
#if canImport(Security)
import Security
#endif

/// Where a relay credential lives. The CLI keeps the 0600 token file so
/// profiles stay shareable with the node app; the iOS app stores the token
/// set in the Keychain.
public protocol RelayCredentialStorage: Sendable {
    func load() throws -> RelayOAuthTokenSet?
    func save(_ tokenSet: RelayOAuthTokenSet) throws
    func clear() throws
}

public struct FileCredentialStorage: RelayCredentialStorage {
    public let url: URL

    public init(url: URL) {
        self.url = url
    }

    public func load() throws -> RelayOAuthTokenSet? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try? WorkflowBundleCodec.decoder().decode(RelayOAuthTokenSet.self, from: Data(contentsOf: url))
    }

    public func save(_ tokenSet: RelayOAuthTokenSet) throws {
        try RelayAuthentication.save(tokenSet, to: url)
    }

    public func clear() throws {
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
}

#if canImport(Security) && !os(Linux)
/// Keychain-backed storage: device-only, available after first unlock, never
/// migrated to other devices or backups.
public struct KeychainCredentialStorage: RelayCredentialStorage {
    public let service: String
    public let account: String

    public init(service: String, account: String) {
        self.service = service
        self.account = account
    }

    private var query: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }

    public func load() throws -> RelayOAuthTokenSet? {
        var lookup = query
        lookup[kSecReturnData as String] = true
        lookup[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(lookup as CFDictionary, &result)
        guard status != errSecItemNotFound else { return nil }
        guard status == errSecSuccess, let data = result as? Data else {
            throw RelayClientError("Could not read the saved relay credential (\(status)).")
        }
        return try? WorkflowBundleCodec.decoder().decode(RelayOAuthTokenSet.self, from: data)
    }

    public func save(_ tokenSet: RelayOAuthTokenSet) throws {
        let data = try WorkflowBundleCodec.encoder().encode(tokenSet)
        var attributes = query
        attributes[kSecValueData as String] = data
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(attributes as CFDictionary, nil)
        if status == errSecDuplicateItem {
            let update = [kSecValueData as String: data] as CFDictionary
            let updateStatus = SecItemUpdate(query as CFDictionary, update)
            guard updateStatus == errSecSuccess else {
                throw RelayClientError("Could not update the saved relay credential (\(updateStatus)).")
            }
            return
        }
        guard status == errSecSuccess else {
            throw RelayClientError("Could not save the relay credential (\(status)).")
        }
    }

    public func clear() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw RelayClientError("Could not remove the saved relay credential (\(status)).")
        }
    }
}
#endif
