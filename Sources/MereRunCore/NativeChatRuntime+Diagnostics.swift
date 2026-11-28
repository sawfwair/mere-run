#if os(macOS) || os(Linux)
import Foundation

public struct NativeChatDiagnostics: Sendable {
    public var gemma4MTP: Gemma4MTPStats?
    public var lagunaDFlash: LagunaDFlashStats?
    public var museDFlash: MuseGlimmerDFlashStats?
    public var nemotronDSpark: NemotronHDSparkStats?
    public var lfm2DSpark: LFM2DSparkStats?
    public var inkling2DSpark: Inkling2DSparkStats?
}

extension NativeChatRuntime {
    public func diagnostics() async -> NativeChatDiagnostics {
        await adapter.diagnostics()
    }
}
#endif
