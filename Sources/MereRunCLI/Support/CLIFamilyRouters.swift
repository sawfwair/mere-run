import Foundation
import MereRunContract

/// Commands whose own router picks the runtime family by rules the contract only approximates.
/// The capability gate runs the router itself, so the gate and `catalog resolve` name the family
/// the command will run. Routers live here, not in Core's `ModelFamilyIdentifier`, when the
/// router they call sits above Core.
enum CLIFamilyRouters {
    /// The family the command's router picks for the whole command line, or `nil` when it has no
    /// router or the router refuses the command line (the command then refuses it too).
    typealias Router = @Sendable (_ invocation: MereRunCommandInvocation) -> String?

    /// Keyed by capability id. Each domain registers its own table here.
    static let routers: [String: Router] = Dictionary(uniqueKeysWithValues: [
        [String: Router](),
    ].flatMap { $0.map { ($0.key, $0.value) } })

    static func family(capabilityID: String, invocation: MereRunCommandInvocation) -> String? {
        routers[capabilityID]?(invocation)
    }
}
