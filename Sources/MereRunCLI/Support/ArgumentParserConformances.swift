import ArgumentParser
import MereRunContract
import MereRunCore

extension LTXVideoVariant: ExpressibleByArgument {}
extension LTXVideoQuality: ExpressibleByArgument {}
extension LTXVideoOutputMode: ExpressibleByArgument {}
extension LTXVideoDecoderKind: ExpressibleByArgument {}
extension LTXTransformerExecution: ExpressibleByArgument {}
extension LTXGuidanceProjectionCacheMode: ExpressibleByArgument {}
extension LTXSamplerMode: ExpressibleByArgument {}
extension LTXGenerationPreset: ExpressibleByArgument {}
extension LTXGenerationPipeline: ExpressibleByArgument {}
extension LTXHDRColorSpace: ExpressibleByArgument {}
extension LTXHDRTransfer: ExpressibleByArgument {}
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
