import Foundation
import MereRunContract

// Image ▸ Train, Chat ▸ Train, and Music ▸ Train are Project surfaces over a task draft: the
// runner names, validates, prepares, records, and remembers their runs like any other task's.
// What is theirs alone lives here, away from SwiftUI so the tests can hold it: the switches the
// pages set on a fresh draft, the check-only variant of a run, the clip list written beside the
// music adapter, and how each page arranges its template's options.

/// One section of a trainer page's settings column: its title and the flags it holds, in order.
package struct StudioTrainingSection: Identifiable, Equatable {
    package let title: String
    package let flags: [String]

    package var id: String { title }

    package init(title: String, flags: [String]) {
        self.title = title
        self.flags = flags
    }
}

package enum StudioTrainingRun {
    // MARK: Fresh drafts

    /// The setting the image trainer's page starts every draft with beyond the template's
    /// defaults: a fixed seed, like the other two trainers' templates. The page used to add a
    /// checkpoint and a preview every 250 steps as well; both are Klein-only options, so they
    /// blocked a Krea 2 preflight before anything was chosen, and now wait under Checkpoints and
    /// previews. Applied to a fresh draft only; a parked or imported one keeps what it has.
    package static func applyingPageDefaults(_ draft: StudioTaskDraft) -> StudioTaskDraft {
        guard draft.templateID == .imageTrainLoRA, draft.form.values["--seed"] == nil else { return draft }
        var draft = draft
        draft.form["--seed"] = .integer(42)
        return draft
    }

    /// The baseline the page's Reset buttons and "changed" counts read against: a fresh draft
    /// with the page's defaults.
    package static func baseline(for templateID: CommandTemplateID) -> StudioTaskDraft {
        applyingPageDefaults(StudioTaskDraft(templateID: templateID))
    }

    /// Whether nothing has been set on a draft yet, so the page's defaults may still be applied:
    /// it is a fresh draft of its template, whatever destination it carries (routing names one
    /// per run; a stamped one was the template's, not the user's).
    package static func isUntouched(_ draft: StudioTaskDraft) -> Bool {
        draft.withoutDestinations() == StudioTaskDraft(templateID: draft.templateID).withoutDestinations()
    }

    // MARK: Recipes and Klein

    /// The image trainer's options a recipe decides. The CLI prefers an explicit flag over the
    /// recipe's value, so a seeded default left on the command line would silently override the
    /// recipe (and `klein-fast-style` would run with `--model image-krea2-raw`); the page left
    /// these off when a recipe was chosen unless the user asked to override it.
    package static let recipeGovernedFlags: [String] = [
        "--width", "--height", "--training-steps", "--model", "--learning-rate", "--rank", "--alpha",
        "--caption-dropout", "--checkpoint-interval", "--max-resolution", "--low-ram", "--no-compile",
        "--lora-target-preset", "--lr-warmup-steps", "--lr-min-factor",
    ]

    /// Whether `flag` is one the chosen recipe decides for this draft.
    package static func isRecipeGoverned(_ flag: String, in draft: StudioTaskDraft) -> Bool {
        draft.templateID == .imageTrainLoRA && !draft.text("--recipe").isBlank && recipeGovernedFlags.contains(flag)
    }

    /// A recipe chosen where there was none: the seeded values of the options it decides are
    /// cleared (`applyingRecipe`). Changing one recipe for another, reopening the page, and
    /// launching leave the draft alone, so a value set once the recipe decides — even one equal
    /// to the seeded default, like a 1024 width under a 768 recipe — is the override the CLI
    /// takes it for.
    package static func choosingRecipe(_ draft: StudioTaskDraft, previous: String) -> StudioTaskDraft {
        guard previous.isBlank else { return draft }
        return applyingRecipe(draft)
    }

    /// The draft with the seeded values of the recipe-governed options cleared once a recipe is
    /// chosen, so the recipe decides them; a value the user typed (anything other than the
    /// seeded one) stays as an explicit override, which is what the CLI does with it.
    package static func applyingRecipe(_ draft: StudioTaskDraft) -> StudioTaskDraft {
        guard draft.templateID == .imageTrainLoRA, !draft.text("--recipe").isBlank else { return draft }
        let seeded = baseline(for: .imageTrainLoRA).form.values
        var draft = draft
        for flag in recipeGovernedFlags where draft.form.values[flag] != nil && draft.form.values[flag] == seeded[flag] {
            draft.form.values[flag] = nil
        }
        return draft
    }

    /// The base model a chosen recipe trains when `--model` is left to it: the contract's default
    /// for the recipe, so the Klein recipe trains the FLUX.2 Klein 9B base and the Krea recipes
    /// Krea 2 raw. Nil without a recipe, and for the other trainers.
    package static func recipeBaseModel(for draft: StudioTaskDraft) -> String? {
        let recipe = draft.text("--recipe")
        guard draft.templateID == .imageTrainLoRA, !recipe.isBlank,
              case .family(_, let model?, .defaultModel) = imageTrainingFamily(["--recipe", recipe]) else { return nil }
        return model
    }

    /// Whether the image draft trains on the contract's FLUX.2 Klein family: the model it names,
    /// or the Klein recipe's base when it names none. A local folder is not guessed from its name.
    package static func trainsKlein(_ draft: StudioTaskDraft) -> Bool {
        guard draft.templateID == .imageTrainLoRA,
              case .family(let family, _, _) = imageTrainingFamily(draft.arguments) else { return false }
        return family == "klein"
    }

    /// The family the image trainer runs for `arguments`; the CLI answers for a local folder.
    private static func imageTrainingFamily(_ arguments: [String]) -> MereRunFamilyResolution {
        StudioScopeSource.live.scope(capability: MereRunCapabilityCatalog.imageTrainLoRA, commandLine: arguments).resolution
    }

    /// Klein checkpoints and previews are opt-in on the command line; without them a run has
    /// nothing to resume from and nothing to watch. A launch against a Klein base sets both to
    /// every 250 steps when they were left unset. Krea 2 takes neither.
    package static func applyingKleinDefaults(_ draft: StudioTaskDraft) -> StudioTaskDraft {
        guard draft.templateID == .imageTrainLoRA, trainsKlein(draft) else { return draft }
        var draft = draft
        for flag in ["--checkpoint-interval", "--sample-interval"] where draft.form.values[flag] == nil {
            draft.form[flag] = .integer(250)
        }
        return draft
    }

    /// What the page launches for a draft: Klein's cadence where the draft trains Klein. The
    /// recipe's options were cleared when it was chosen (`choosingRecipe`); what is set now is
    /// set on purpose.
    package static func launchDraft(_ draft: StudioTaskDraft) -> StudioTaskDraft {
        applyingKleinDefaults(draft)
    }

    /// Whether a managed model is a base the trainer can train: a model of the image trainer's
    /// contract families (Krea 2 Raw and the Klein bases), ACE-Step for music, any text-chat
    /// model for text.
    package static func isTrainableBase(_ modelID: String, for templateID: CommandTemplateID) -> Bool {
        switch templateID {
        case .imageTrainLoRA:
            return MereRunCapabilityCatalog.imageTrainLoRA.routing?.families.contains { $0.models.contains(modelID) } == true
        case .musicTrainAdapter: return modelID.hasPrefix("music-acestep")
        default: return true
        }
    }

    // MARK: Preflight

    /// The run that checks a request without training, with the machine-readable report the
    /// page reads: `--preflight --json` for the image trainer, `--dry-run --json` for the text
    /// trainer. Nil for the music trainer, whose checks are the clip list's own; ACE-Step checks
    /// its model files when training starts.
    package static func preflightDraft(_ draft: StudioTaskDraft) -> StudioTaskDraft? {
        var checked = draft
        switch draft.templateID {
        case .imageTrainLoRA: checked.form["--preflight"] = .flag(true)
        case .textTrainLoRA: checked.form["--dry-run"] = .flag(true)
        default: return nil
        }
        checked.form["--json"] = .flag(true)
        return checked
    }

    /// The switches `preflightDraft` adds, for "Use these settings" on a check's Library row:
    /// the draft it restores is the training run the check was for, so Start trains.
    package static func withoutCheckSwitches(_ draft: StudioTaskDraft) -> StudioTaskDraft {
        var training = draft
        switch draft.templateID {
        case .imageTrainLoRA: training.form.values["--preflight"] = nil
        case .textTrainLoRA: training.form.values["--dry-run"] = nil
        default: return draft
        }
        training.form.values["--json"] = nil
        return training
    }

    /// The page's check on the schedule before a run: the steps, rank, and learning rate the
    /// draft sets must be positive. One left blank is the recipe's or the CLI's own default,
    /// which is how a recipe leaves them.
    package static func scheduleProblem(in draft: StudioTaskDraft) -> String? {
        let steps = draft.templateID == .musicTrainAdapter ? "--steps" : "--training-steps"
        let notPositive = [steps, "--rank", "--learning-rate"].contains { flag in
            let text = draft.text(flag)
            return !text.isBlank && (Double(text) ?? 0) <= 0
        }
        return notPositive ? "Steps, rank, and learning rate must be positive." : nil
    }

    /// The draft once a run is submitted: a checkpoint resume is used once, so the next run
    /// starts fresh unless another checkpoint is chosen.
    package static func afterSubmitting(_ draft: StudioTaskDraft) -> StudioTaskDraft {
        var next = draft
        switch draft.templateID {
        case .imageTrainLoRA:
            next.form.values["--resume-from"] = nil
        case .textTrainLoRA:
            next.form.values["--resume-from"] = nil
            next.form.values["--resume-step"] = nil
        default:
            break
        }
        return next
    }

    // MARK: Music

    /// Music ▸ Train's launch: the adapter named by routing, the clips written beside it as
    /// `<adapter>.dataset.jsonl`, and `--dataset` pointed at that file, so the run folder holds
    /// what made the adapter. The name is derived before the manifest replaces the page's saved
    /// copy in `--dataset`, so the adapter the Command view previews is the adapter this writes;
    /// the runner then prepares the request as it is rather than naming it again.
    package static func musicLaunch(
        _ draft: StudioTaskDraft,
        manifest: StudioMusicTrainingManifest,
        fileManager: FileManager = .default
    ) throws -> StudioTaskDraft {
        let named = StudioOutputLocation.destination(for: draft, fileManager: fileManager)
        let manifestURL = StudioMusicTrainingManifest.manifestURL(besideOutput: named.text("--output"), fileManager: fileManager)
        try fileManager.createDirectory(at: manifestURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try manifest.jsonl().write(to: manifestURL, options: .atomic)
        var launch = named
        launch.form["--dataset"] = .text(manifestURL.path)
        return launch
    }

    // MARK: Sections

    /// The flags a trainer page shows under its model picker: where the base model's files are.
    package static func modelFlags(for templateID: CommandTemplateID) -> [String] {
        switch templateID {
        case .textTrainLoRA:
            return ["--model-path"]
        case .musicTrainAdapter:
            return ["--checkpoints-root", "--decoder-subdirectory", "--vae-subdirectory", "--text-subdirectory"]
        default:
            return []
        }
    }

    /// How a trainer page arranges its template's options: the sections its page always had,
    /// in order. The well's slots, the model picker's flags, the destinations routing fills, and
    /// the launcher's switches are not listed; every other option the contract declares and no
    /// section names falls under Advanced.
    package static func sections(for templateID: CommandTemplateID) -> [StudioTrainingSection] {
        switch templateID {
        case .imageTrainLoRA:
            return [
                StudioTrainingSection(title: "Training", flags: [
                    "--recipe", "--width", "--height", "--training-steps", "--batch-size", "--rank", "--alpha",
                    "--learning-rate", "--seed",
                ]),
                StudioTrainingSection(title: "Memory and schedule", flags: [
                    "--progressive", "--low-ram", "--gradient-checkpointing", "--no-compile", "--lite",
                    "--base-quantization-bits", "--scheduler-steps", "--lr-warmup-steps", "--no-cosine-scheduler",
                    "--lr-min-factor", "--adam-weight-decay", "--caption-dropout", "--max-resolution", "--max-text-length",
                ]),
                StudioTrainingSection(title: "Checkpoints and previews", flags: [
                    "--checkpoint-interval", "--sample-interval", "--sample-prompt", "--sample-model", "--sample-steps",
                    "--sample-cfg", "--sample-lora-scale", "--sample-seed", "--exclude-preview-images",
                ]),
                StudioTrainingSection(title: "Klein targets and timesteps", flags: [
                    "--lora-target-mode", "--lora-rank-preset", "--lora-target-preset", "--lora-target-ranks",
                    "--timestep-sampling", "--timestep-loss-weighting", "--loss-weighting", "--timestep-low", "--timestep-high",
                ]),
            ]
        case .textTrainLoRA:
            return [
                StudioTrainingSection(title: "Training", flags: [
                    "--adapter-name", "--training-steps", "--batch-size", "--rank", "--alpha", "--learning-rate", "--seed",
                    "--max-sequence-length", "--target-modules", "--reasoning-effort",
                ]),
            ]
        case .musicTrainAdapter:
            return [
                StudioTrainingSection(title: "Training", flags: [
                    "--kind", "--rank", "--alpha", "--factor", "--steps", "--learning-rate", "--weight-decay", "--seed",
                    "--max-duration", "--log-every",
                ]),
            ]
        default:
            return []
        }
    }

    // MARK: Handoffs

    /// Points a trainer's task draft at a dataset from another surface (Image ▸ Datasets ▸
    /// Discover's "Train on it"): the parked draft, or a fresh one with the page's defaults,
    /// gets `path` in its dataset well and is parked again, so the Train page opens on it. The
    /// caller then opens the task.
    @MainActor
    package static func attachDataset(_ path: String, to task: StudioTask, sessions: StudioTaskSessions) {
        guard let templateID = task.trainingTemplateID,
              let slot = StudioTaskSchema.primarySlot(for: templateID) else { return }
        var draft = sessions.taskDraft(for: task).map(applyingPageDefaults) ?? baseline(for: templateID)
        draft.setAttachmentText(path, for: slot.storage)
        sessions.setTaskDraft(draft, for: task)
    }
}

extension StudioTask {
    /// The template a Train task runs; nil for every other task.
    package var trainingTemplateID: CommandTemplateID? {
        switch self {
        case .imageTrain: return .imageTrainLoRA
        case .chatTrain: return .textTrainLoRA
        case .musicTrain: return .musicTrainAdapter
        default: return nil
        }
    }
}
