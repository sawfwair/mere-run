import Foundation
import HTTPTypes
import Hummingbird
import MereRunContract
import MereRunCore

/// `api serve`'s share of the model scope check. A request's fields, under the CLI flags they
/// stand for, are resolved and scoped the way the CLI gate scopes a command line: the same
/// contract, family resolver, and identifier. What the family refuses fails the request with the
/// API's 400 before anything loads; what it ignores comes back as `x-mere-warning` response
/// headers, one per warning, in the CLI's words.
///
/// Each scoped route runs the command's own validation and generator, so what the contract
/// refuses, the route refused before too, at the latest once the model loaded. The route's own
/// checks still run after this one.
struct APIModelScope: Equatable {
    static let warningHeader = HTTPField.Name("x-mere-warning")!

    /// What a scoped route does with a model the CLI command can't run.
    enum ExcludedModels {
        /// Refuses it: the route runs the command's own model resolution.
        case refuse
        /// Leaves it to the route, which serves some of them under another engine (chat serves
        /// DeepSeek V4 Flash through its proxy).
        case outOfScope
    }

    let warnings: [String]

    /// `invocation` holds the request's fields under the capability's flags; `fields` names the
    /// request field each flag came from, which a refusal names. A flag without a field is one
    /// the request passed in CLI syntax, in video's `options`.
    static func check(
        _ capabilityID: String,
        _ invocation: MereRunCommandInvocation,
        fields: [String: String],
        excludedModels: ExcludedModels = .refuse
    ) throws -> APIModelScope {
        let capability = capability(capabilityID)
        let resolution = CLICapabilityGate.resolution(capability, invocation)
        switch resolution {
        case .unrouted, .unidentified:
            return APIModelScope(warnings: [])
        case .excluded where excludedModels == .outOfScope:
            return APIModelScope(warnings: [])
        case .excluded, .unmatched:
            let refusal = capability.report(for: resolution, invocation).violations.joined(separator: " ")
            let modelField = capability.routing?.modelFlags.lazy.compactMap { fields[$0] }.first ?? "model"
            throw APIRequestValidationError.invalidField(modelField, sentence(refusal))
        case .family(let family, _, _):
            let found = capability.violations(
                invocation, family: family, identify: CLICapabilityGate.identify(capability, invocation)
            )
            if let refusal = found.first(where: { $0.severity == .error }) {
                throw APIRequestValidationError.invalidField(fields[refusal.flag] ?? "options", sentence(refusal.message))
            }
            return APIModelScope(warnings: found.map(\.message))
        }
    }

    private static func capability(_ id: String) -> MereRunCommandCapability {
        guard let capability = MereRunCapabilityCatalog.command(id: id) else {
            preconditionFailure("\(id) is not a cataloged capability")
        }
        return capability
    }

    /// `APIRequestValidationError` adds its own period.
    private static func sentence(_ message: String) -> String {
        message.hasSuffix(".") ? String(message.dropLast()) : message
    }
}

extension Response {
    /// The response with each of the scope's warnings as an `x-mere-warning` header.
    func addingWarnings(of scope: APIModelScope) -> Response {
        var response = self
        for warning in scope.warnings {
            response.headers.append(HTTPField(name: APIModelScope.warningHeader, value: warning))
        }
        return response
    }
}

// MARK: - Routes

extension APIModelScope {
    /// `/v1/images/generations` and `/v1/images/edits` as `image generate`. The mask part is left
    /// out: the route conditions on the whole image for every model.
    static func image(_ plan: APIServerContract.ImageGenerationPlan) throws -> APIModelScope {
        var values: [String: [String]] = ["--model": [plan.modelID]]
        values["--negative-prompt"] = plan.negativePrompt.map { [$0] }
        values["--cfg"] = plan.guidanceScale.map { [String($0)] }
        values["--steps"] = plan.steps.map { [String($0)] }
        values["--input"] = plan.inputImage.map { [$0.path] }
        values["--ref-image"] = plan.additionalInputImages.isEmpty ? nil : plan.additionalInputImages.map(\.path)
        values["--strength"] = plan.strength.map { [String($0)] }
        return try check("image.generate", MereRunCommandInvocation(values: values), fields: [
            "--model": "model", "--negative-prompt": "negative_prompt", "--cfg": "guidance_scale",
            "--steps": "steps", "--input": "image", "--ref-image": "image", "--strength": "strength",
        ])
    }

    /// `/v1/videos/generations` as `video generate`: the typed fields as the flags the route sets
    /// on the parsed command, and `options` as the command line it parses.
    static func video(_ plan: APIServerContract.VideoGenerationPlan) throws -> APIModelScope {
        var arguments = [
            "--model", plan.modelID, "--width", String(plan.width), "--height", String(plan.height),
            "--fps", String(plan.fps),
        ]
        arguments += plan.seconds.map { ["--duration", String($0)] } ?? []
        arguments += plan.numFrames.map { ["--num-frames", String($0)] } ?? []
        arguments += plan.seed.map { ["--seed", String($0)] } ?? []
        arguments += plan.quality.map { ["--quality", $0.rawValue] } ?? []
        arguments += plan.outputMode.map { ["--output-mode", $0.rawValue] } ?? []
        return try check(
            "video.generate",
            MereRunCommandInvocation(capability: capability("video.generate"), arguments: arguments + plan.options),
            fields: [
                "--model": "model", "--width": "size", "--height": "size", "--fps": "fps", "--duration": "seconds",
                "--num-frames": "num_frames", "--seed": "seed", "--quality": "quality", "--output-mode": "output_mode",
            ]
        )
    }

    /// `/v1/audio/speech` as `speech synthesize`, which runs the voice description in style mode.
    static func speech(_ plan: APIServerContract.SpeechPlan) throws -> APIModelScope {
        try check("speech.synthesize", MereRunCommandInvocation(values: [
            "--model": [plan.modelID], "--voice": [plan.voiceDescription],
        ]), fields: ["--model": "model", "--voice": "voice"])
    }

    /// `/v1/audio/transcriptions` as `speech transcribe`. The route's default token budget is
    /// its own, so `--max-tokens` stands only for a `max_tokens` the request sent.
    static func transcription(_ plan: APIServerContract.TranscriptionPlan, form: MultipartFormData) throws -> APIModelScope {
        var values: [String: [String]] = ["--model": [plan.modelID], "--task": [plan.task.rawValue]]
        values["--language"] = plan.language.map { [$0] }
        if APIServerContract.normalizedOptional(form.field("max_tokens")) != nil {
            values["--max-tokens"] = [String(plan.maxTokens)]
        }
        return try check("speech.transcribe", MereRunCommandInvocation(values: values), fields: [
            "--model": "model", "--task": "task", "--language": "language", "--max-tokens": "max_tokens",
        ])
    }

    /// `/v1/audio/diarizations` as `speech diarize`; `--latency` only when the request sent one.
    static func diarization(_ plan: APIServerContract.DiarizationPlan, form: MultipartFormData) throws -> APIModelScope {
        var values: [String: [String]] = ["--model": [plan.modelID]]
        if APIServerContract.normalizedOptional(form.field("latency")) != nil {
            values["--latency"] = [plan.latency.rawValue]
        }
        return try check("speech.diarize", MereRunCommandInvocation(values: values), fields: [
            "--model": "model", "--latency": "latency",
        ])
    }

    /// `/v1/chat/completions` as `text chat`, for the fields the client sent; `resolved` is the
    /// route's own reading of them. A top-k of 0 is the route's "off".
    static func chat(
        _ request: OpenAIChatRequest, resolved: ChatRequest, modelID: String
    ) throws -> APIModelScope {
        var values: [String: [String]] = ["--model": [modelID]]
        values["--top-k"] = request.top_k.flatMap { $0 == 0 ? nil : [String($0)] }
        values["--top-p"] = request.top_p.map { [String($0)] }
        values["--min-p"] = request.min_p.map { [String($0)] }
        values["--seed"] = request.seed.map { [String($0)] }
        values["--temperature"] = request.temperature.map { [String($0)] }
        values["--response-format"] = request.response_format.map { [$0.type == "text" ? "text" : "json_object"] }
        values["--tools"] = resolved.tools.map { [$0.map(\.name).joined(separator: ",")] }
        values["--reasoning-effort"] = resolved.reasoningEffort.map { [String($0)] }
        values["--show-unmasking"] = resolved.showUnmasking ? [] : nil
        values["--image"] = resolved.messages.compactMap(\.imageUrl).last.map { [$0] }
        values["--audio"] = resolved.messages.compactMap(\.audioUrl).last.map { [$0] }
        values["--video"] = resolved.messages.compactMap(\.videoUrl).last.map { [$0] }
        return try check("text.chat", MereRunCommandInvocation(values: values), fields: [
            "--model": "model", "--top-k": "top_k", "--top-p": "top_p", "--min-p": "min_p", "--seed": "seed",
            "--temperature": "temperature", "--response-format": "response_format", "--tools": "tools",
            "--reasoning-effort": "reasoning_effort", "--show-unmasking": "mere_show_unmasking",
            "--image": "messages.content", "--audio": "messages.content", "--video": "messages.content",
        ], excludedModels: .outOfScope)
    }
}
