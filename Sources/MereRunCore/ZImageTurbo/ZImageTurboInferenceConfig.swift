import Foundation

public struct ZImageTurboInferenceConfig: Sendable, Hashable {
    public var width: Int
    public var height: Int
    public var numInferenceSteps: Int
    public var imageStrength: Double?

    public init(
        width: Int,
        height: Int,
        numInferenceSteps: Int,
        imageStrength: Double? = nil
    ) {
        self.width = ZImageTurboInferenceConfig.roundDownToMultipleOf16(width)
        self.height = ZImageTurboInferenceConfig.roundDownToMultipleOf16(height)
        self.numInferenceSteps = max(numInferenceSteps, 1)
        self.imageStrength = imageStrength
    }

    public var initTimeStep: Int {
        guard let imageStrength, imageStrength > 0 else { return 0 }
        let strength = min(max(imageStrength, 0.0), 1.0)
        // Match mflux img2img semantics:
        // higher strength => later start => less added noise => stronger input-image preservation.
        return max(1, Int(Double(numInferenceSteps) * strength))
    }

    public var timeSteps: Range<Int> {
        initTimeStep..<numInferenceSteps
    }

    private static func roundDownToMultipleOf16(_ value: Int) -> Int {
        16 * (max(value, 16) / 16)
    }
}
