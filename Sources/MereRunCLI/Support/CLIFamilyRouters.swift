import Foundation
import MereRunContract

/// Commands whose own router picks the runtime family by rules the contract only approximates;
/// their routing says so with `routed_by_command`. The capability gate runs the router itself, so
/// the gate and `catalog resolve` name the family the command will run. Routers live here, not in
/// Core's `ModelFamilyIdentifier`, when the router they call sits above Core.
enum CLIFamilyRouters {
    /// The family the command's router picks for the whole command line, or `nil` when it has no
    /// router or the router refuses the command line (the command then refuses it too).
    typealias Router = @Sendable (_ invocation: MereRunCommandInvocation) -> String?

    /// Keyed by capability id. Each domain declares its table beside its commands and lists it
    /// here; capability ids are disjoint.
    static let routers: [String: Router] = domainRouters.reduce(into: [:]) { table, domain in
        table.merge(domain) { _, _ in preconditionFailure("Two family routers claim one capability.") }
    }

    private static let domainRouters: [[String: Router]] = [
        speechRouters
    ]

    static func family(capabilityID: String, invocation: MereRunCommandInvocation) -> String? {
        routers[capabilityID]?(invocation)
    }
}
