import Foundation
import MereRunContract

/// The runtime `music generate` and `music serve` pick for a `--model` value. The commands route
/// through these detectors, and the capability gate's probes wrap the same ones, so the gate and
/// the run cannot disagree about a folder.
public enum MusicModelRuntime: Sendable, Equatable {
    case yue2
    case miniMaxMusic3
    case magentaRT2
    case aceStep

    /// `music generate`, in its routing order: YuE2 by managed id or a `config.json` whose
    /// `model_type` is `yue2`; MiniMax Music 3 by managed id or its three configs; Magenta RealTime
    /// 2 by managed id or an `.mlxfn` export beside MusicCoCa; ACE-Step otherwise.
    public static func generation(model: String) -> Self {
        if isYuE2(model) { return .yue2 }
        if isMiniMaxMusic3(model) { return .miniMaxMusic3 }
        if isMagentaRT2(model) { return .magentaRT2 }
        return .aceStep
    }

    /// `music serve`: MiniMax Music 3 by managed id or layout, ACE-Step otherwise.
    public static func serving(model: String) -> Self {
        isMiniMaxMusic3(model) ? .miniMaxMusic3 : .aceStep
    }

    /// The managed id or its upstream repository, in any case, rather than a local root.
    public static func namesManagedYuE2(_ model: String) -> Bool {
        names(model, id: YuE2Resources.modelID, repository: YuE2Resources.repository)
    }

    public static func namesManagedMiniMaxMusic3(_ model: String) -> Bool {
        names(model, id: MiniMaxMusic3Resources.modelID, repository: MiniMaxMusic3Resources.repository)
    }

    static func isYuE2(_ model: String) -> Bool {
        namesManagedYuE2(model) || YuE2Resources.looksLikeRoot(ACEStepRuntimePreparation.resolveUserPath(model))
    }

    static func isMiniMaxMusic3(_ model: String) -> Bool {
        namesManagedMiniMaxMusic3(model)
            || MiniMaxMusic3Resources.looksLikeRoot(ACEStepRuntimePreparation.resolveUserPath(model))
    }

    static func isMagentaRT2(_ model: String) -> Bool {
        MagentaRT2Resources.isMagentaRT2Model(model)
            || MagentaRT2Resources.looksLikeMagentaRT2Root(URL(fileURLWithPath: model).standardizedFileURL)
    }

    private static func names(_ model: String, id: String, repository: String) -> Bool {
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == id || normalized == repository.lowercased()
    }
}

extension ACEStepCheckpointVariant {
    /// The `music generate` family whose options this checkpoint takes.
    var musicGenerateFamily: String {
        if isTurbo { return "ace-step-turbo" }
        return supportsBaseTasks ? "ace-step-base" : "ace-step-sft"
    }

    /// The checkpoint `music generate` or `music serve` would load for these ACE-Step options, when
    /// its decoder is already on disk.
    static func local(_ invocation: MereRunCommandInvocation, model: String) -> Self? {
        let decoderSubdirectory = invocation.value("--decoder-subdirectory")
            ?? ACEStepRuntimePreparation.defaultDecoderSubdirectory
        guard let root = ACEStepRuntimePreparation.localCheckpointsRoot(
            model: model,
            checkpointsRoot: invocation.value("--checkpoints-root"),
            turboSubdirectory: decoderSubdirectory,
            vaeSubdirectory: invocation.value("--vae-subdirectory") ?? ACEStepRuntimePreparation.defaultVAESubdirectory,
            lmSubdirectory: invocation.contains("--use-lm") ? invocation.value("--lm-subdirectory") : nil,
            textSubdirectory: invocation.value("--text-subdirectory")
        ), let decoder = ACEStepRuntimePreparation.usableDecoderSubdirectory(at: root, explicit: decoderSubdirectory) else {
            return nil
        }
        // A decoder with an unreadable config is one the command fails on too; the gate leaves it to the run.
        return try? load(modelRootURL: root.appendingPathComponent(decoder, isDirectory: true))
    }
}

extension ModelFamilyIdentifier {
    /// `music generate` names a local model with `--model`, and an ACE-Step checkpoint also with
    /// `--checkpoints-root` or `--decoder-subdirectory`; whichever the gate passed, the probe reads
    /// the whole command line the way the command does. A managed id is also probed, because
    /// `MERERUN_MUSIC_ACESTEP_ROOT` can hold a different checkpoint than the id's own install; an
    /// alias reads as its managed id, whose install the command falls back to.
    static let musicProbes: [String: Probe] = [
        "music.generate": { _, invocation in
            let requested = invocation.value("--model") ?? ModelResolver.ModelID.aceStep.rawValue
            let model = ManagedModelCatalog.spec(for: requested)?.id ?? requested
            switch MusicModelRuntime.generation(model: model) {
            case .yue2: return "yue2"
            case .miniMaxMusic3: return "minimax-music3"
            case .magentaRT2: return "magenta-rt2"
            case .aceStep: return ACEStepCheckpointVariant.local(invocation, model: model)?.musicGenerateFamily
            }
        },
        "music.serve": { model, _ in
            MusicModelRuntime.serving(model: model) == .miniMaxMusic3 ? "minimax-music3" : "ace-step"
        }
    ]
}
