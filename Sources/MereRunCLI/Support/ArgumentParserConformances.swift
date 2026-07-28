import ArgumentParser
import MereRunCore

extension ACEStepTask: ExpressibleByArgument {}
extension ACEStepChunkMaskMode: ExpressibleByArgument {}
extension ACEStepRepaintMode: ExpressibleByArgument {}
extension ACEStepInferenceMethod: ExpressibleByArgument {}
extension ACEStepSamplerMode: ExpressibleByArgument {}
extension ACEStepGuidanceMode: ExpressibleByArgument {}
extension ACEStepQualityPreset: ExpressibleByArgument {}
extension ACEStepAudioFormat: ExpressibleByArgument {}
extension ACEStepNormalizationMode: ExpressibleByArgument {}
extension ACEStepAdapterKind: ExpressibleByArgument {}

extension ModelResolver.ModelID: ExpressibleByArgument {
    public init?(argument: String) {
        self.init(rawValue: argument)
    }
}

extension LagunaDFlashRoutingMode: ExpressibleByArgument {}
