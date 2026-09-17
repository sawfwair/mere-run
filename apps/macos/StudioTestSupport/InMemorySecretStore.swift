import Foundation
import StudioKit

/// Test double for `StudioSecretStore`: a dictionary instead of the login Keychain, with
/// switchable failures so a test can play a locked or denied Keychain.
package final class InMemorySecretStore: StudioSecretStore {
    package private(set) var secrets: [String: String]
    /// When set, every read throws it.
    package var readError: StudioSecretStoreError?
    /// When set, every write and remove throws it and leaves `secrets` unchanged.
    package var writeError: StudioSecretStoreError?

    package init(secrets: [String: String] = [:]) {
        self.secrets = secrets
    }

    package func secret(named name: String) throws -> String? {
        if let readError { throw readError }
        return secrets[name]
    }

    package func setSecret(_ value: String, named name: String) throws {
        if let writeError { throw writeError }
        secrets[name] = value
    }

    package func removeSecret(named name: String) throws {
        if let writeError { throw writeError }
        secrets[name] = nil
    }
}
