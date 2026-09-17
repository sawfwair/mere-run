import Foundation
import Security

/// Why a named secret could not be read, written, or removed.
package enum StudioSecretStoreError: Error, Equatable {
    package enum Operation: String, Equatable {
        case read, write, remove
    }

    /// The store refused the app: the Keychain is locked, or the user declined the access prompt.
    case accessDenied(operation: Operation)
    /// Any other failure; `status` is the Security framework result code.
    case failed(operation: Operation, status: OSStatus)
}

extension StudioSecretStoreError: LocalizedError {
    package var errorDescription: String? {
        switch self {
        case .accessDenied(let operation):
            return "Keychain \(operation.rawValue) was denied."
        case .failed(let operation, let status):
            let message = SecCopyErrorMessageString(status, nil).map { $0 as String } ?? "OSStatus \(status)"
            return "Keychain \(operation.rawValue) failed: \(message)"
        }
    }
}

/// The one owner of app credentials in StudioKit. Secrets are addressed by name; ordinary
/// preferences (host, port, paths) stay in `UserDefaults`.
package protocol StudioSecretStore {
    /// The stored value, or nil when nothing is stored under `name`.
    func secret(named name: String) throws -> String?
    /// Stores or replaces the value under `name`.
    func setSecret(_ value: String, named name: String) throws
    /// Removes the value under `name`; removing an absent secret is not an error.
    func removeSecret(named name: String) throws
}

/// Generic-password items in the user's login Keychain: one service for the app, one account
/// per secret name, readable after first unlock and never migrated to another device.
package struct KeychainSecretStore: StudioSecretStore {
    package static let defaultService = "run.mere.app"

    package let service: String

    package init(service: String = KeychainSecretStore.defaultService) {
        self.service = service
    }

    private func query(for name: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: name,
        ]
    }

    package func secret(named name: String) throws -> String? {
        var lookup = query(for: name)
        lookup[kSecReturnData as String] = true
        lookup[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(lookup as CFDictionary, &result)
        switch status {
        case errSecItemNotFound:
            return nil
        case errSecSuccess:
            guard let data = result as? Data, let value = String(data: data, encoding: .utf8) else {
                throw StudioSecretStoreError.failed(operation: .read, status: errSecDecode)
            }
            return value
        default:
            throw Self.error(.read, status)
        }
    }

    package func setSecret(_ value: String, named name: String) throws {
        let data = Data(value.utf8)
        var attributes = query(for: name)
        attributes[kSecValueData as String] = data
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(attributes as CFDictionary, nil)
        switch status {
        case errSecSuccess:
            return
        case errSecDuplicateItem:
            let update = [kSecValueData as String: data] as CFDictionary
            let updateStatus = SecItemUpdate(query(for: name) as CFDictionary, update)
            guard updateStatus == errSecSuccess else { throw Self.error(.write, updateStatus) }
        default:
            throw Self.error(.write, status)
        }
    }

    package func removeSecret(named name: String) throws {
        let status = SecItemDelete(query(for: name) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw Self.error(.remove, status)
        }
    }

    private static func error(_ operation: StudioSecretStoreError.Operation, _ status: OSStatus) -> StudioSecretStoreError {
        switch status {
        case errSecAuthFailed, errSecInteractionNotAllowed, errSecUserCanceled, errSecMissingEntitlement:
            return .accessDenied(operation: operation)
        default:
            return .failed(operation: operation, status: status)
        }
    }
}
