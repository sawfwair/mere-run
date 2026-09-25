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
    /// alias reads as its managed id, whose install the command falls back to. With nothing on
    /// disk, a managed id under the default decoder runs its own layout, so it answers as itself;
    /// a named decoder may pick another variant of the download, which only the run can tell.
    ///
    /// The ACE-Step commands never load a language model id themselves. They run one only when a
    /// checkpoints root loads first: `--checkpoints-root`, which they load or refuse on its own,
    /// or a usable root already on disk such as `MERERUN_MUSIC_ACESTEP_ROOT`. The probes answer
    /// for those ids only then, and the contract's exclusion stands otherwise.
    static let musicProbes: [String: Probe] = [
        "music.generate": { _, invocation in
            let requested = invocation.value("--model") ?? ModelResolver.ModelID.aceStep.rawValue
            let model = ManagedModelCatalog.spec(for: requested)?.id ?? requested
            let family: String? = switch MusicModelRuntime.generation(model: model) {
            case .yue2: "yue2"
            case .miniMaxMusic3: "minimax-music3"
            case .magentaRT2: "magenta-rt2"
            case .aceStep: ACEStepCheckpointVariant.local(invocation, model: model)?.musicGenerateFamily
            }
            if let family { return .family(family) }
            // An explicit root the command can't use fails the run on its own; only the id's own
            // download stands behind a managed id.
            let decoder = invocation.value("--decoder-subdirectory")?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard ManagedModelCatalog.spec(for: model) != nil, invocation.value("--checkpoints-root") == nil,
                  decoder == nil || decoder == ACEStepRuntimePreparation.defaultDecoderSubdirectory else { return nil }
            return .managedModel(model)
        },
        "music.serve": { model, invocation in
            if isACEStepLanguageModel(model) {
                return aceStepRootLoadsFirst(model: model, invocation) ? .family("ace-step") : nil
            }
            return .family(MusicModelRuntime.serving(model: model) == .miniMaxMusic3 ? "minimax-music3" : "ace-step")
        },
        "music.analyze": aceStepOnlyProbe,
        "music.train-adapter": aceStepOnlyProbe
    ]

    /// `music analyze` and `music train-adapter` run ACE-Step alone: a language model id runs
    /// only behind a root that loads first; anything else is the command's to judge.
    private static let aceStepOnlyProbe: Probe = { model, invocation in
        guard isACEStepLanguageModel(model), aceStepRootLoadsFirst(model: model, invocation) else { return nil }
        return .family("ace-step")
    }

    private static func isACEStepLanguageModel(_ model: String) -> Bool {
        [ModelResolver.ModelID.aceStepLM17B.rawValue, ModelResolver.ModelID.aceStepLM4B.rawValue]
            .contains(ManagedModelCatalog.spec(for: model)?.id ?? model)
    }

    /// True when the command would load a checkpoints root before it ever resolved `model`: an
    /// explicit `--checkpoints-root` (loaded, or refused as incomplete), or a usable root on disk.
    private static func aceStepRootLoadsFirst(model: String, _ invocation: MereRunCommandInvocation) -> Bool {
        if let explicit = invocation.value("--checkpoints-root"),
           !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return true
        }
        return ACEStepRuntimePreparation.localCheckpointsRoot(
            model: model,
            checkpointsRoot: nil,
            turboSubdirectory: invocation.value("--decoder-subdirectory") ?? ACEStepRuntimePreparation.defaultDecoderSubdirectory,
            vaeSubdirectory: invocation.value("--vae-subdirectory") ?? ACEStepRuntimePreparation.defaultVAESubdirectory,
            lmSubdirectory: nil,
            textSubdirectory: invocation.value("--text-subdirectory")
        ) != nil
    }
}
