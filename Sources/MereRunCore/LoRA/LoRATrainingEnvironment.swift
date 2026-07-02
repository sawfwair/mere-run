import Foundation
import Darwin

/// Environment-tunable training behavior shared by the image-LoRA trainers.
public enum LoRATrainingEnvironment {
    /// Explicit gradient-checkpointing override from the environment:
    /// MERERUN_LORA_TRAIN_GRAD_CHECKPOINT=1 forces on, =0 forces off,
    /// unset defers to the resolution-aware default.
    public static let gradientCheckpointingOverride: Bool? = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_LORA_TRAIN_GRAD_CHECKPOINT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !raw.isEmpty else {
            return nil
        }
        if raw == "0" || raw == "false" || raw == "off" { return false }
        return true
    }()

    /// Resolution-aware gradient-checkpointing default. Measured on M4 Max /
    /// 128 GB: full-injection steps at 1024x1024 peak 159 GB (Krea 2) and
    /// 183 GB (Klein base, crashes), while the recipe workflows at <=512px and
    /// 768x416 fit comfortably without recompute overhead. The pixel threshold
    /// keeps proven low-resolution recipes byte-identical in behavior and
    /// protects high-resolution runs by default.
    public static func gradientCheckpointingEnabled(trainingPixels: Int) -> Bool {
        if let forced = gradientCheckpointingOverride {
            return forced
        }
        return trainingPixels >= 768 * 768
    }

    /// MLX buffer-cache cap during training, in gigabytes. The cache grows to
    /// the transient high-water mark and never shrinks, pinning the process
    /// footprint at the worst spike of the run. 0 leaves it unlimited.
    /// MERERUN_LORA_TRAIN_CACHE_LIMIT_GB overrides.
    public static let trainingCacheLimitGB: Int = {
        if let raw = ProcessInfo.processInfo.environment["MERERUN_LORA_TRAIN_CACHE_LIMIT_GB"],
           let value = Int(raw), value >= 0 {
            return value
        }
        return 16
    }()

    /// Adapter checkpoint cadence in optimizer steps for trainers without
    /// their own mid-run checkpointing. MERERUN_LORA_TRAIN_SAVE_EVERY
    /// overrides; 0 disables.
    public static let periodicSaveInterval: Int = {
        if let raw = ProcessInfo.processInfo.environment["MERERUN_LORA_TRAIN_SAVE_EVERY"],
           let value = Int(raw), value >= 0 {
            return value
        }
        return 100
    }()

    /// MERERUN_LORA_TRAIN_SYNC_EVAL=1 evaluates each training step
    /// synchronously, capping in-flight activations at one step's worth.
    public static let synchronousStepEval: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_LORA_TRAIN_SYNC_EVAL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw == "1" || raw == "true" || raw == "on"
    }()

    /// Physical memory footprint (the metric jetsam actually enforces —
    /// resident plus compressed), in gigabytes.
    public static func currentPhysicalFootprintGB() -> Double? {
        var usage = rusage_info_current()
        let status = withUnsafeMutablePointer(to: &usage) { pointer in
            pointer.withMemoryRebound(to: (rusage_info_t?).self, capacity: 1) { reboundPointer in
                proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, reboundPointer)
            }
        }
        guard status == 0 else { return nil }
        return Double(usage.ri_phys_footprint) / 1_000_000_000
    }
}
