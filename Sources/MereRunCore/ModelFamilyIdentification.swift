import Foundation
import MereRunContract

/// Answers the contract's family resolver for models the contract does not list. An alias or
/// upstream repository id normalizes to its managed id; a local model goes to its capability's
/// probe, which wraps the detector the command itself uses and maps its answer to a contract
/// family id. The probes own "what is this folder"; the contract owns "what may this family take".
public enum ModelFamilyIdentifier {
    /// Inspects `model` for one capability: a family of that capability, the managed model the
    /// folder holds (an excluded one resolves to its reason), or `nil`.
    public typealias Probe = @Sendable (
        _ model: String, _ invocation: MereRunCommandInvocation
    ) -> MereRunModelIdentification?

    /// Picks the default model this machine runs among candidates that span families, the way
    /// the command itself picks it.
    public typealias DefaultChooser = @Sendable (_ candidates: [String]) -> String?

    /// Per-capability probes, keyed by capability id. Each domain registers its own.
    static let probes: [String: Probe] = soundEffectProbes.merging(geoProbes) { first, _ in first }

    public static func identify(
        capabilityID: String,
        model: String,
        invocation: MereRunCommandInvocation
    ) -> MereRunModelIdentification? {
        if let spec = ManagedModelCatalog.spec(for: model), spec.id != model {
            return .managedModel(spec.id)
        }
        return probes[capabilityID]?(model, invocation)
    }

    /// Per-capability machine defaults, keyed by capability id.
    static let defaultChoosers: [String: DefaultChooser] = geoDefaultChoosers

    /// The candidate a capability's machine-chosen default runs on this machine, or `nil` when
    /// the capability registers no chooser.
    public static func machineDefault(capabilityID: String, candidates: [String]) -> String? {
        defaultChoosers[capabilityID]?(candidates)
    }
}
