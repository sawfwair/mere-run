import Foundation

extension MereRunCapabilityCatalog {
    public static let videoGenerate = MereRunCommandCapability(
        id: "video.generate",
        command: ["video", "generate"],
        title: "Generate video",
        summary: "Generate LTX, Wan, or synchronized MiniMax-H3 video from text and ordered media conditioning.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(
                flag: "--variant", label: "Compatibility variant", kind: .choice, choices: ["unified-av", "distilled"],
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--ltx-transformer-execution", label: "LTX transformer execution", kind: .choice,
                choices: ["eager", "compiled"], defaultValue: "eager", group: Group.run, tier: .expert
            ),
            .init(
                flag: "--ltx-guidance-projection-cache", label: "LTX guidance projection cache", kind: .choice,
                choices: ["automatic", "disabled", "enabled"], defaultValue: "disabled", group: Group.run, tier: .expert
            ),
            .init(
                flag: "--ltx-teacache", label: "Enable LTX TeaCache", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(
                flag: "--ltx-teacache-threshold", label: "LTX TeaCache threshold", kind: .number, group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--ltx-teacache-calibration-output", label: "LTX TeaCache calibration output", kind: .file,
                group: Group.output, tier: .expert
            ),
            .init(
                flag: "--h3-render-width", label: "H3 render width", kind: .integer, group: Group.output, tier: .expert
            ),
            .init(
                flag: "--h3-render-height", label: "H3 render height", kind: .integer, group: Group.output, tier: .expert
            ),
            .init(
                flag: "--h3-adapter", label: "H3 adapter", kind: .string, group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--h3-adapter-strength", label: "H3 adapter strength", kind: .number, defaultValue: "1.0",
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--h3-frame", label: "H3 timed frame", kind: .string, repeatable: true, group: Group.inputs,
                tier: .expert
            ),
            .init(
                flag: "--h3-window-frames", label: "H3 window frames", kind: .integer, group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--h3-window-overlap", label: "H3 window overlap", kind: .integer, defaultValue: "18",
                group: Group.sampling, tier: .expert
            ),
            .init(flag: "--output", label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(
                flag: "--quality",
                label: "Quality",
                kind: .choice,
                choices: LTXVideoQuality.allCases.map(\.rawValue),
                group: Group.sampling, tier: .essential
            ),
            .init(
                flag: "--output-mode",
                label: "Output mode",
                kind: .choice,
                choices: LTXVideoOutputMode.allCases.map(\.rawValue),
                group: Group.output, tier: .essential
            ),
            .init(flag: "--model-root", label: "Model root", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--auto-duration", label: "Auto duration range", kind: .string, repeatable: true, group: Group.sampling, tier: .expert),
            .init(
                flag: "--video-decoder", label: "Video decoder", kind: .choice, choices: ["diffusion", "convolutional"],
                group: Group.run, tier: .expert
            ),
            .init(
                flag: "--hdr", label: "HDR color space", kind: .choice, choices: ["srgb-linear", "acescg", "acescct"],
                group: Group.output, tier: .expert
            ),
            .init(
                flag: "--hdr-transfer", label: "HDR transfer", kind: .choice, choices: ["acescct", "logc3"],
                group: Group.output, tier: .expert, dependsOn: "--hdr"
            ),
            .init(flag: "--high-quality-hdr", label: "High quality HDR", kind: .boolean, group: Group.output, tier: .expert, dependsOn: "--hdr"),
            .init(flag: "--text-embeddings", label: "Precomputed text contexts", kind: .file, group: Group.inputs, tier: .expert),
            .init(
                flag: "--spatial-tile", label: "VAE spatial tile", kind: .integer,
                group: Group.run, tier: .expert, range: .init(min: 256, max: 4_096, step: 32)
            ),
            .init(
                flag: "--spatial-overlap", label: "VAE spatial overlap", kind: .integer,
                defaultValue: "256", group: Group.run, tier: .expert, range: .init(min: 0, max: 1_024, step: 32)
            ),
            .init(flag: "--skip-mp4", label: "EXR only", kind: .boolean, group: Group.output, tier: .expert, dependsOn: "--hdr"),
            .init(
                flag: "--width", label: "Width", kind: .integer,
                group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--height", label: "Height", kind: .integer,
                group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--num-frames", label: "Frames", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 1_000, step: 1)
            ),
            .init(
                flag: "--duration", label: "Duration", kind: .number,
                group: Group.sampling, tier: .essential, range: .init(min: 1, max: 60, step: 0.5)
            ),
            .init(
                flag: "--fps", label: "Frames per second", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 60, step: 1)
            ),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(
                flag: "--steps", label: "Denoising steps", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 100, step: 1)
            ),
            .init(
                flag: "--h3-weight-mode",
                label: "MiniMax-H3 weight mode",
                kind: .choice,
                choices: ["auto", "quantized", "resident-bf16"],
                defaultValue: "auto", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--h3-acceleration",
                label: "MiniMax-H3 acceleration",
                kind: .choice,
                choices: [
                    "quality", "balanced", "maximum",
                    "layers-45", "layers-40", "velocity-reuse-2", "token-reduction"
                ],
                defaultValue: "quality", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--guidance-scale", label: "Wan guidance", kind: .number,
                defaultValue: "5.0", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--shift", label: "Wan schedule shift", kind: .number,
                defaultValue: "5.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--enhance-prompt", label: "Enhance prompt", kind: .boolean, group: Group.prompt, tier: .standard),
            .init(
                flag: "--prompt-enhancer-model", label: "Prompt enhancer", kind: .string,
                group: Group.prompt, tier: .expert, dependsOn: "--enhance-prompt"
            ),
            .init(
                flag: "--prompt-enhancer-model-root", label: "Prompt enhancer root", kind: .directory,
                group: Group.prompt, tier: .expert, dependsOn: "--enhance-prompt"
            ),
            .init(flag: "--audio", label: "Source audio", kind: .file, group: Group.inputs, tier: .standard),
            .init(
                flag: "--audio-start-time", label: "Audio start", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .standard, range: .init(min: 0, step: 0.1), dependsOn: "--audio"
            ),
            .init(
                flag: "--audio-max-duration", label: "Audio max duration", kind: .number,
                group: Group.inputs, tier: .expert, range: .init(min: 0, step: 0.1), dependsOn: "--audio"
            ),
            .init(
                flag: "--a2v-guidance-scale", label: "Audio-to-video guidance", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--video-cfg-guidance-scale", label: "Video CFG guidance", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--audio-cfg-guidance-scale", label: "Audio CFG guidance", kind: .number,
                defaultValue: "7.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--v2a-guidance-scale", label: "Video-to-audio guidance", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--a2v-steps", label: "Audio-to-video steps", kind: .integer,
                defaultValue: "30", group: Group.sampling, tier: .expert, range: .init(min: 1, max: 100, step: 1)
            ),
            .init(
                flag: "--ltx-preset", label: "LTX preset", kind: .choice, choices: ["standard", "hq"],
                defaultValue: "standard", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--ltx-pipeline", label: "LTX pipeline", kind: .choice, choices: ["two-stage", "keyframe-interpolation", "dev-one-stage"],
                defaultValue: "two-stage", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--ltx-sampler", label: "LTX sampler", kind: .choice,
                choices: ["euler", "res2s", "euler-ancestral", "cfg-plus-plus", "gradient-estimating-euler"],
                group: Group.sampling, tier: .expert
            ),
            .init(flag: "--ltx-sigmas", label: "Stage one sigmas", kind: .string, repeatable: true, group: Group.sampling, tier: .expert),
            .init(flag: "--ltx-stage-2-sigmas", label: "Stage two sigmas", kind: .string, repeatable: true, group: Group.sampling, tier: .expert),
            .init(
                flag: "--distilled-lora-strength-stage-1", label: "Stage one distilled strength", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--distilled-lora-strength-stage-2", label: "Stage two distilled strength", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--ltx-sampler-eta", label: "Sampler eta", kind: .number,
                defaultValue: "0.5", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--video-stg-scale", label: "Video STG", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(
                flag: "--video-guidance-rescale", label: "Video rescale", kind: .number,
                defaultValue: "0.7", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--video-stg-block", label: "Video STG block", kind: .integer, repeatable: true,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--video-guidance-skip-step", label: "Video guidance skip", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 1)
            ),
            .init(
                flag: "--audio-stg-scale", label: "Audio STG", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(
                flag: "--audio-guidance-rescale", label: "Audio rescale", kind: .number,
                defaultValue: "0.7", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--audio-stg-block", label: "Audio STG block", kind: .integer, repeatable: true,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--audio-guidance-skip-step", label: "Audio guidance skip", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 1)
            ),
            .init(flag: "--no-res2s-bong-math", label: "Disable Res2s anchor refinement", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--res2s-bong-max-iterations", label: "Res2s iterations", kind: .integer,
                defaultValue: "100", group: Group.sampling, tier: .expert, range: .init(min: 1, max: 1_000, step: 1)
            ),
            .init(
                flag: "--gradient-estimation-gamma", label: "Gradient estimate gamma", kind: .number,
                defaultValue: "2.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(flag: "--image", label: "Start image", kind: .file, group: Group.inputs, tier: .essential),
            .init(
                flag: "--image-strength", label: "Start image strength", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--image"
            ),
            .init(flag: "--end-image", label: "End image", kind: .file, group: Group.inputs, tier: .standard, dependsOn: "--image"),
            .init(
                flag: "--end-image-strength", label: "End image strength", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--end-image"
            ),
            .init(
                flag: "--image-conditioning", label: "Timed image guide", kind: .string, repeatable: true,
                group: Group.inputs, tier: .expert
            ),
            .init(
                flag: "--num-generated-keyframes", label: "Generated keyframe count", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 16, step: 1)
            ),
            .init(
                flag: "--generated-keyframe", label: "Generated keyframe", kind: .integer, repeatable: true,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(flag: "--lora", label: "LTX LoRA", kind: .string, repeatable: true, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--video-conditioning", label: "IC-LoRA reference video", kind: .string, repeatable: true,
                group: Group.inputs, tier: .expert
            ),
            .init(
                flag: "--conditioning-attention-strength", label: "Reference attention", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--video-conditioning"
            ),
            .init(
                flag: "--conditioning-attention-mask", label: "Reference attention mask", kind: .file,
                group: Group.inputs, tier: .expert, dependsOn: "--video-conditioning"
            ),
            .init(flag: "--skip-stage-2", label: "Stage one preview", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--reference-downscale-factor", label: "Reference spatial scale", kind: .integer,
                group: Group.inputs, tier: .expert, range: .init(min: 1, max: 8, step: 1), dependsOn: "--video-conditioning"
            ),
            .init(
                flag: "--reference-temporal-scale-factor", label: "Reference temporal scale", kind: .integer,
                group: Group.inputs, tier: .expert, range: .init(min: 1, max: 8, step: 1), dependsOn: "--video-conditioning"
            ),
            .init(flag: "--dfr", label: "Diffusion fidelity rendering", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--temporal-upsample-rounds", label: "Temporal refinement rounds", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 2, step: 1), dependsOn: "--dfr"
            ),
            .init(
                flag: "--detailing-lora", label: "Detailing IC-LoRA", kind: .string, repeatable: true,
                group: Group.modelAndAdapters, tier: .expert, dependsOn: "--dfr"
            ),
            .init(
                flag: "--detailing-reference-downscale-factor", label: "Detail reference scale", kind: .integer,
                group: Group.modelAndAdapters, tier: .expert, range: .init(min: 1, max: 8, step: 1), dependsOn: "--dfr"
            ),
            .init(flag: "--reference", label: "Ordered H3 reference", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--timings", label: "Timings", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--timings-output", label: "Timings output", kind: .file, group: Group.run, tier: .expert),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let videoRetake = MereRunCommandCapability(
        id: "video.retake",
        command: ["video", "retake"],
        title: "Retake video region",
        summary: "Regenerate a bounded video and/or audio region with native LTX-2.5.",
        arguments: [
            .init(name: "prompt", label: "Replacement prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--source", label: "Source video or EXR folder", kind: .file, required: true),
            .init(flag: "--frame-rate", label: "EXR frame rate", kind: .integer),
            .init(flag: "--start-time", label: "Start time", kind: .number, required: true),
            .init(flag: "--end-time", label: "End time", kind: .number, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--enhance-prompt", label: "Enhance prompt", kind: .boolean),
            .init(flag: "--prompt-enhancer-model", label: "Prompt enhancer", kind: .string),
            .init(flag: "--prompt-enhancer-model-root", label: "Prompt enhancer root", kind: .directory),
            .init(flag: "--steps", label: "Denoising steps", kind: .integer),
            .init(flag: "--sigmas", label: "Sigma schedule", kind: .string, repeatable: true),
            .init(flag: "--lora", label: "LTX LoRA", kind: .string, repeatable: true),
            .init(flag: "--video-cfg-guidance-scale", label: "Video CFG", kind: .number),
            .init(flag: "--video-stg-scale", label: "Video STG", kind: .number),
            .init(flag: "--video-guidance-rescale", label: "Video rescale", kind: .number),
            .init(flag: "--video-modality-scale", label: "Video modality guidance", kind: .number),
            .init(flag: "--video-stg-block", label: "Video STG block", kind: .integer, repeatable: true),
            .init(flag: "--video-guidance-skip-step", label: "Video guidance skip", kind: .integer),
            .init(flag: "--audio-cfg-guidance-scale", label: "Audio CFG", kind: .number),
            .init(flag: "--audio-stg-scale", label: "Audio STG", kind: .number),
            .init(flag: "--audio-guidance-rescale", label: "Audio rescale", kind: .number),
            .init(flag: "--audio-modality-scale", label: "Audio modality guidance", kind: .number),
            .init(flag: "--audio-stg-block", label: "Audio STG block", kind: .integer, repeatable: true),
            .init(flag: "--audio-guidance-skip-step", label: "Audio guidance skip", kind: .integer),
            .init(flag: "--video-decoder", label: "Video decoder", kind: .choice, choices: ["diffusion", "convolutional"]),
            .init(flag: "--hdr", label: "HDR color space", kind: .choice, choices: ["srgb-linear", "acescg", "acescct"]),
            .init(flag: "--hdr-transfer", label: "HDR transfer", kind: .choice, choices: ["acescct", "logc3"]),
            .init(flag: "--preserve-video", label: "Preserve video", kind: .boolean),
            .init(flag: "--preserve-audio", label: "Preserve audio", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let videoDubIt = MereRunCommandCapability(
        id: "video.dub-it",
        command: ["video", "dub-it"],
        title: "Dub-It",
        summary: "Rephrase a reference performance while preserving speaker identity and lip motion.",
        arguments: [
            .init(name: "prompt", label: "New performance prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--reference-video", label: "Reference AV", kind: .file, required: true),
            .init(flag: "--ic-lora", label: "Dub-It IC-LoRA", kind: .file, required: true),
            .init(flag: "--ic-lora-strength", label: "IC-LoRA strength", kind: .number),
            .init(flag: "--reference-strength", label: "Reference strength", kind: .number),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--image-conditioning", label: "Timed image guide", kind: .string, repeatable: true),
            .init(flag: "--stage-1-sigmas", label: "Stage one sigmas", kind: .string, repeatable: true),
            .init(flag: "--stage-2-sigmas", label: "Stage two sigmas", kind: .string, repeatable: true),
            .init(flag: "--enhance-prompt", label: "Enhance prompt", kind: .boolean),
            .init(flag: "--prompt-enhancer-model", label: "Prompt enhancer", kind: .string),
            .init(flag: "--prompt-enhancer-model-root", label: "Prompt enhancer root", kind: .directory),
            .init(flag: "--video-decoder", label: "Video decoder", kind: .choice, choices: ["diffusion", "convolutional"]),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let videoAnimate = MereRunCommandCapability(
        id: "video.animate",
        command: ["video", "animate"],
        title: "Animate subject",
        summary: "Animate or replace masked subjects with native SCAIL-2.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--reference", label: "Reference image", kind: .file, required: true),
            .init(flag: "--reference-mask", label: "Reference mask", kind: .file, required: true),
            .init(flag: "--driving-video", label: "Driving video", kind: .file, required: true),
            .init(flag: "--driving-mask", label: "Driving mask", kind: .file, required: true),
            .init(flag: "--additional-reference", label: "Additional reference", kind: .file, repeatable: true),
            .init(flag: "--additional-reference-mask", label: "Additional reference mask", kind: .file, repeatable: true),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--mode", label: "Mode", kind: .choice, choices: ["animation", "replacement"]),
            .init(flag: "--profile", label: "Profile", kind: .choice, choices: ["fast", "quality"]),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--steps", label: "Steps", kind: .integer),
            .init(flag: "--guidance-scale", label: "Guidance", kind: .number),
            .init(flag: "--shift", label: "Shift", kind: .number),
            .init(flag: "--sampler", label: "Sampler", kind: .choice, choices: ["unipc", "euler"]),
            .init(flag: "--distilled-adapter", label: "Distilled adapter", kind: .file),
            .init(flag: "--distilled-adapter-strength", label: "Adapter strength", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--fps", label: "Frames per second", kind: .integer),
            .init(flag: "--segment-length", label: "Segment length", kind: .integer),
            .init(flag: "--segment-overlap", label: "Segment overlap", kind: .integer),
            .init(flag: "--tail-policy", label: "Tail policy", kind: .choice, choices: ["drop", "pad-trim"]),
            .init(flag: "--audio-source", label: "Audio source", kind: .choice, choices: ["none", "driving"]),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let videoCosmos3 = MereRunCommandCapability(
        id: "video.cosmos3",
        command: ["video", "cosmos3"],
        title: "Cosmos3",
        summary: "Run Cosmos3-Edge generation, dynamics, policy, and reasoning modes.",
        arguments: [
            .init(name: "prompt", label: "Prompt or action task", kind: .string, required: true)
        ],
        options: [
            .init(
                flag: "--mode",
                label: "Mode",
                kind: .choice,
                choices: [
                    "text-to-image", "image-to-image", "text-to-video", "image-to-video",
                    "video-to-video", "policy", "forward-dynamics", "inverse-dynamics", "reasoner"
                ]
            ),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--actions-output", label: "Actions output", kind: .file),
            .init(flag: "--image", label: "Conditioning image", kind: .file),
            .init(flag: "--video", label: "Conditioning video", kind: .file),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--num-frames", label: "Frames", kind: .integer),
            .init(flag: "--steps", label: "Steps", kind: .integer),
            .init(flag: "--guidance-scale", label: "Guidance", kind: .number),
            .init(flag: "--shift", label: "Shift", kind: .number),
            .init(flag: "--schedule", label: "Schedule", kind: .choice, choices: ["nvidia", "published-karras"]),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--fps", label: "Frames per second", kind: .integer),
            .init(flag: "--condition-latent-frame", label: "Conditioned frames", kind: .string, repeatable: true),
            .init(flag: "--keep-video-tail", label: "Keep video tail", kind: .boolean),
            .init(flag: "--action-domain", label: "Action domain", kind: .string),
            .init(flag: "--action-file", label: "Action file", kind: .file),
            .init(flag: "--action-chunk-size", label: "Action chunk", kind: .integer),
            .init(flag: "--action-resolution", label: "Action resolution", kind: .integer),
            .init(flag: "--action-viewpoint", label: "Action viewpoint", kind: .choice, choices: ["ego_view", "third_person_view", "wrist_view", "concat_view"]),
            .init(flag: "--max-new-tokens", label: "Reasoner tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--max-video-frames", label: "Reasoner video frames", kind: .integer),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, flag: "--output")
    )

    public static let videoPrepareMasks = MereRunCommandCapability(
        id: "video.prepare-masks",
        command: ["video", "prepare-masks"],
        title: "Prepare SCAIL-2 masks",
        summary: "Create immutable palette-safe SCAIL-2 masks with native SAM 3.1.",
        options: [
            .init(flag: "--plan", label: "Mask plan", kind: .file, required: true),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory, required: true),
            .init(flag: "--preview-frame", label: "Preview frame", kind: .integer),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output-dir")
    )

    public static let videoExportLatents = MereRunCommandCapability(
        id: "video.export-latents",
        command: ["video", "export-latents"],
        title: "Export video latents",
        summary: "Run native LTX denoising and export final latents.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--num-frames", label: "Frames", kind: .integer),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let videoSession = MereRunCommandCapability(
        id: "video.session",
        command: ["video", "session"],
        title: "Resident LTX session",
        summary: "Keep an LTX 2.3 runtime resident for JSONL generation requests.",
        options: [
            .init(
                flag: "--video-decoder", label: "Video decoder", kind: .choice, choices: ["convolutional", "diffusion"],
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--ltx-transformer-execution", label: "LTX transformer execution", kind: .choice,
                choices: ["eager", "compiled"], defaultValue: "eager", group: Group.run, tier: .expert
            ),
            .init(
                flag: "--ltx-guidance-projection-cache", label: "LTX guidance projection cache", kind: .choice,
                choices: ["automatic", "disabled", "enabled"], defaultValue: "disabled", group: Group.run, tier: .expert
            ),
            .init(
                flag: "--ltx-teacache", label: "Enable LTX TeaCache", kind: .boolean, group: Group.run, tier: .expert
            ),
            .init(
                flag: "--ltx-teacache-threshold", label: "LTX TeaCache threshold", kind: .number, group: Group.run,
                tier: .expert
            ),
            .init(
                flag: "--prompt-cache-capacity", label: "Prompt cache capacity", kind: .integer, defaultValue: "8",
                group: Group.run, tier: .expert
            ),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service)
    )
}
