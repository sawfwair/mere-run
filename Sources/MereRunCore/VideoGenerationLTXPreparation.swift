import Foundation
import MereRunContract

/// Reads adapter headers to resolve HDR and reference-video preparation.
/// This does not load tensor payloads, install adapters, or create outputs.
public struct VideoGenerationLTXPreparation: Sendable {
    public let loras: [LTXLoRAConfiguration]
    public let detailingLoRAs: [LTXLoRAConfiguration]
    public let hdrColorSpace: LTXHDRColorSpace?
    public let hdrTransfer: LTXHDRTransfer
    public let hdrICLoRA: LTXHDRICLoRAOptions?
    public let referenceDownscaleFactor: Int
    public let referenceTemporalScaleFactor: Int

    public init(
        options: VideoGenerationOptions,
        profile: VideoGenerationModelProfile,
        loras: [LTXLoRAConfiguration],
        detailingLoRAs: [LTXLoRAConfiguration] = []
    ) throws {
        self.loras = loras
        self.detailingLoRAs = detailingLoRAs
        let hdrConfigurations = try loras.compactMap(ltxHDRLoRAConfiguration)
        guard Set(hdrConfigurations).count <= 1 else {
            throw Self.issue("Stacked HDR LoRAs must use the same HDR transform and reference downscale factor.")
        }
        let hdr = hdrConfigurations.first
        let referenceScale = options.videoConditionings.isEmpty
            ? LTXLoRAReferenceScaleConfiguration()
            : try ltxLoRAReferenceScaleConfiguration(loras)
        hdrColorSpace = options.hdrColorSpace ?? hdr.map { _ in .srgbLinear }
        hdrTransfer = options.hdrTransfer ?? hdr?.hdrTransform ?? .acesCCT
        referenceDownscaleFactor = options.referenceDownscaleFactor
            ?? hdr?.referenceDownscaleFactor ?? referenceScale.downscaleFactor
        referenceTemporalScaleFactor = options.referenceTemporalScaleFactor ?? referenceScale.temporalScaleFactor
        if hdrColorSpace != nil, !profile.isLTX25 {
            throw Self.issue("HDR and HDR IC-LoRA workflows require an official LTX 2.5 model root.")
        }
        if options.highQualityHDR, hdr == nil {
            throw Self.issue("--high-quality-hdr requires an HDR IC-LoRA with hdr_transform metadata.")
        }
        if hdr != nil {
            guard !options.videoConditionings.isEmpty else {
                throw Self.issue("An HDR IC-LoRA requires at least one --video-conditioning reference.")
            }
            guard options.effectiveOutputMode == .videoOnly else {
                throw Self.issue("HDR IC-LoRA is a video-only pipeline; use --output-mode video-only.")
            }
            hdrICLoRA = LTXHDRICLoRAOptions(highQuality: options.highQualityHDR)
        } else {
            hdrICLoRA = nil
        }
        if options.skipHDRMP4, hdrICLoRA == nil {
            throw Self.issue("--skip-mp4 is available only for the dedicated HDR IC-LoRA pipeline.")
        }
        if options.textEmbeddings != nil {
            guard hdrICLoRA != nil else {
                throw Self.issue("--text-embeddings is available for the dedicated HDR IC-LoRA pipeline.")
            }
            if options.enhancePrompt || !options.autoDuration.isEmpty {
                throw Self.issue("--text-embeddings cannot be combined with --enhance-prompt or --auto-duration.")
            }
        }
    }

    private static func issue(_ message: String) -> VideoGenerationIssue {
        VideoGenerationIssue(id: "ltx_preparation_invalid", title: "LTX preparation is invalid", message: message)
    }
}
