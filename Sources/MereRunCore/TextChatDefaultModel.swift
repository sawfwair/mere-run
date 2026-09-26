import Foundation

/// The model `text chat` runs when `--model` is left off. Apple Silicon picks Gemma 4 12B
/// 4-bit. Linux first tries Qwen3.6-35B-A3B, as GGUF/llama.cpp on CUDA (~68 tok/s on GB10, vs
/// ~13 for MLX there) or MLX otherwise, then Gemma 4 12B 4-bit. Nano is the final fallback
/// everywhere. The capability contract's text chat default rules list the same candidates, and
/// the capability gate asks this chooser which one a machine runs.
public enum TextChatDefaultModel {
    /// The default on this machine.
    public static var current: String {
        #if os(Linux)
        let cuda = ProcessInfo.processInfo.environment["MERERUN_LINUX_ACCEL"]?.lowercased() == "cuda"
        #else
        let cuda = false
        #endif
        return id(on: .current, linuxCUDA: cuda)
    }

    public static func id(on machine: MereRunMachineProfile, linuxCUDA: Bool = false) -> String {
        func fits(_ id: String) -> Bool {
            guard let descriptor = ManagedModelCapabilityCatalog.descriptor(for: id) else { return false }
            return machine.unifiedMemoryGB >= descriptor.minimumUnifiedMemoryGB
        }

        let a3b = machine.isLinux
            ? (linuxCUDA ? "text-chat-q36-nano-gguf" : Q35Resources.q36NanoModelId)
            : Gemma4Resources.twelveB4BitModelId
        if fits(a3b) { return a3b }
        if fits(Gemma4Resources.twelveB4BitModelId) { return Gemma4Resources.twelveB4BitModelId }
        return Gemma4Resources.nanoModelId
    }
}
