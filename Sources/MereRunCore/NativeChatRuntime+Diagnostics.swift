#if os(macOS) || os(Linux)
import Foundation

public struct NativeChatDiagnostics: Sendable {
    public var gemma4MTP: Gemma4MTPStats?
    public var lagunaDFlash: LagunaDFlashStats?
    public var museDFlash: MuseGlimmerDFlashStats?
    public var nemotronDSpark: NemotronHDSparkStats?
    public var lfm2DSpark: LFM2DSparkStats?
}

extension NativeChatRuntime {
    public func diagnostics() async -> NativeChatDiagnostics {
        var result = NativeChatDiagnostics()
        switch self {
        case .textChatGemma4(let generator, _): result.gemma4MTP = await generator.mtpStats()
        case .textChatLaguna(let generator, _): result.lagunaDFlash = await generator.dflashStats()
        case .textChatMuseGlimmer(let generator, _): result.museDFlash = await generator.dflashStats()
        case .textChatNemotronH(let generator, _): result.nemotronDSpark = await generator.dsparkStats()
        case .textChatLFM2(let generator, _): result.lfm2DSpark = await generator.dsparkStats()
        case .textCode, .textChatKlein, .textChatDiffusionGemma, .textChatQ35,
             .textChatDeepseekV4Flash, .textChatNemotronOmni, .textChatPsi, .textChatInkling:
            break
        }
        return result
    }
}
#endif
