import Foundation
import MereRunContract

/// Answers the contract's family resolver for models the contract does not list. An alias or
/// upstream repository id normalizes to its managed id; a local model goes to its capability's
/// probe, which wraps the detector the command itself uses and maps its answer to a contract
/// family id. The probes own "what is this folder"; the contract owns "what may this family take".
public enum ModelFamilyIdentifier {
    /// Inspects `model` for one capability and returns a family id of that capability, or `nil`.
    public typealias Probe = @Sendable (_ model: String, _ invocation: MereRunCommandInvocation) -> String?

    /// Per-capability probes, keyed by capability id. Each domain registers its own table here.
    static let probes: [String: Probe] = Dictionary(uniqueKeysWithValues: [
        speechProbes,
    ].flatMap { $0.map { ($0.key, $0.value) } })

    public static func identify(
        capabilityID: String,
        model: String,
        invocation: MereRunCommandInvocation
    ) -> MereRunModelIdentification? {
        if let spec = ManagedModelCatalog.spec(for: model), spec.id != model {
            return .managedModel(spec.id)
        }
        return probes[capabilityID]?(model, invocation).map(MereRunModelIdentification.family)
    }
}
