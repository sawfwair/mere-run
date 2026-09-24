import Foundation
import StudioKit

/// Settings a test hands the code under test without touching the user's own. Values go into
/// this process's registration domain, which cfprefsd never writes to disk, so nothing lands in
/// ~/Library/Preferences (not even the empty plist that removing a suite's persistent domain
/// leaves behind). The registration domain is one per process and every `UserDefaults` reads it,
/// so each `register` is undone by a matching `restore`, which puts back exactly what it found.
package enum StudioTestDefaults {
    nonisolated(unsafe) private static var saved: [[String: Any]] = []

    package static func register(_ values: [String: Any]) {
        saved.append(UserDefaults.standard.volatileDomain(forName: UserDefaults.registrationDomain))
        UserDefaults.standard.register(defaults: values)
    }

    package static func restore() {
        UserDefaults.standard.setVolatileDomain(saved.removeLast(), forName: UserDefaults.registrationDomain)
    }

    /// Points `StudioOutputLocation` at folders under `root`: runs file under `outputs` (the
    /// configured root; `root/outputs` when nil), and App Outputs and the pages' draft files
    /// under `root/support`, so nothing a test prepares reaches ~/Pictures, ~/Music,
    /// ~/Documents, or ~/Library/Application Support. Undo with `restore()`.
    package static func redirectOutputs(under root: URL, outputs: URL? = nil) {
        register([
            StudioOutputLocation.rootDefaultsKey: (outputs ?? root.appendingPathComponent("outputs", isDirectory: true)).path,
            StudioOutputLocation.supportRootDefaultsKey: root.appendingPathComponent("support", isDirectory: true).path,
        ])
    }

    /// Only App Outputs and the draft files move; runs keep the per-media default folders, which
    /// a test may name but must never prepare. Undo with `restore()`.
    package static func redirectSupport(under root: URL) {
        register([
            StudioOutputLocation.rootDefaultsKey: "",
            StudioOutputLocation.supportRootDefaultsKey: root.appendingPathComponent("support", isDirectory: true).path,
        ])
    }
}
