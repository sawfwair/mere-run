import Foundation

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

    /// The base model a chosen recipe trains when `--model` is left to it, as the CLI resolves
    /// the recipe (`resolveLoRATrainingRecipe`): the Klein recipe trains the FLUX.2 Klein 9B
    /// base, the Krea recipes Krea 2 raw. Nil without a recipe, and for the other trainers.
    package static func recipeBaseModel(for draft: StudioTaskDraft) -> String? {
        guard draft.templateID == .imageTrainLoRA else { return nil }
        let recipe = draft.text("--recipe").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !recipe.isEmpty else { return nil }
        return recipe.contains("klein") ? "image-klein-base-9b" : "image-krea2-raw"
    }

    /// Whether the image draft trains a FLUX.2 Klein base: the Klein recipe, or a model id that
    /// says so.
    package static func trainsKlein(_ draft: StudioTaskDraft) -> Bool {
        draft.text("--recipe") == "klein-fast-style" || StudioTaskSchema.modelID(for: draft).lowercased().contains("klein")
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

    /// What the page launches for a draft: the recipe's options cleared, then Klein's cadence.
    package static func launchDraft(_ draft: StudioTaskDraft) -> StudioTaskDraft {
        applyingKleinDefaults(applyingRecipe(draft))
    }

    /// Whether a managed model is a base the trainer can train: Krea 2 and the Klein base models
    /// for images, ACE-Step for music, any text-chat model for text.
    package static func isTrainableBase(_ modelID: String, for templateID: CommandTemplateID) -> Bool {
        switch templateID {
        case .imageTrainLoRA: return modelID.hasPrefix("image-krea2") || modelID.hasPrefix("image-klein-base")
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
