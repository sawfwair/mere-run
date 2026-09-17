import Foundation

extension MachineInferenceClass {
    /// A model that declares at least this much unified memory needs most of a
    /// machine even when its download is small, for example one that loads a
    /// BF16 base before quantizing it at load time.
    public static let largeDeclaredMinimumMemoryGB = 64

    /// Classifies a resolved model estimate, retaining a modality's minimum cost.
    /// Unknown estimates use at least standard admission. A declared minimum
    /// memory at or above `largeDeclaredMinimumMemoryGB` is large regardless of
    /// the download estimate. Explicit exclusivity remains a caller policy, for
    /// example for an exclusive server backend.
    public static func forModel(
        estimatedBytes: Int64?,
        declaredMinimumMemoryGB: Int? = nil,
        minimum: MachineInferenceClass = .standard,
        requiresExclusive: Bool = false
    ) -> MachineInferenceClass {
        let gibibyte = Int64(1_073_741_824)
        if requiresExclusive || minimum == .large || estimatedBytes.map({ $0 >= 48 * gibibyte }) == true
            || declaredMinimumMemoryGB.map({ $0 >= largeDeclaredMinimumMemoryGB }) == true {
            return .large
        }
        if minimum == .small, let estimatedBytes, estimatedBytes <= 16 * gibibyte {
            return .small
        }
        return .standard
    }
}

extension MachineInferenceRequest {
    public init(
        label: String,
        estimatedModelBytes: Int64?,
        minimumClass: MachineInferenceClass = .standard,
        requiresExclusive: Bool = false
    ) {
        self.init(label: label, resourceClass: .forModel(
            estimatedBytes: estimatedModelBytes, minimum: minimumClass, requiresExclusive: requiresExclusive
        ))
    }
}
