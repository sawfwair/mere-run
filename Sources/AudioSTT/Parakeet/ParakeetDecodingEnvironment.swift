import Foundation

/// Environment-tunable decode behavior for the Parakeet transducer models.
enum ParakeetDecodingEnvironment {
    /// Number of encoder frames evaluated per batched joint call during
    /// greedy transducer decoding. The decoder state only changes on token
    /// emission, so blanks (the majority of frames) scan host-side from a
    /// single readback instead of one (RNN-T) or two (TDT) scalar readbacks
    /// per frame. `MERERUN_STT_DECODE_WINDOW` overrides; `1` restores the
    /// legacy per-frame readback cadence.
    static let windowedDecodeFrames: Int = {
        if let raw = ProcessInfo.processInfo.environment["MERERUN_STT_DECODE_WINDOW"],
           let value = Int(raw), value >= 1 {
            return value
        }
        return 16
    }()
}
