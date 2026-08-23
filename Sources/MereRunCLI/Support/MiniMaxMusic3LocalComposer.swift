import ArgumentParser
import Foundation
import MLX
import MereRunCore

struct MiniMaxMusic3CompositionReceipt: Codable {
    static let currentSchemaVersion = 1

    var schemaVersion: Int
    var createdAt: Date
    var modelID: String
    var request: MiniMaxMusic3CompositionRequest
    var blueprint: MiniMaxMusic3SongBlueprint
    var song: MiniMaxMusic3ComposedSong
    var lyricPreflight: MiniMaxMusic3LyricPreflightReport

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case createdAt = "created_at"
        case modelID = "model_id"
        case request
        case blueprint
        case song
        case lyricPreflight = "lyric_preflight"
    }

    func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(self).write(to: url, options: .atomic)
    }
}

enum MiniMaxMusic3LocalComposer {
    typealias ChatPass = (ChatRequest) async throws -> ChatResponse

    static func compose(
        request: MiniMaxMusic3CompositionRequest,
        modelID: String,
        modelRoot: String?,
        requireInstalled: Bool,
        quiet: Bool
    ) async throws -> MiniMaxMusic3CompositionReceipt {
        let normalizedModelID = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        try TextChat.validate(responseFormat: .jsonObject, modelID: normalizedModelID)
        guard Gemma4Resources.handles(modelSpec: normalizedModelID)
                || Q35Resources.profile(for: normalizedModelID) != nil else {
            throw ValidationError(
                "--composer-model must select a native Gemma4 or Qwen-family MLX chat model."
            )
        }

        let explicitRoot = try modelRoot.map { rawValue -> String in
            let url = ACEStepCLIHelper.resolveUserPath(rawValue)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Composer model root not found: \(url.path)")
            }
            return url.path
        }
        let installedRoot = ManagedModelResolver.resolveInstalledModel(id: normalizedModelID)?.path
        if requireInstalled, explicitRoot == nil, installedRoot == nil {
            throw ValidationError(
                "Composer model '\(normalizedModelID)' is not installed. Run "
                    + "'mere.run model pull \(normalizedModelID)' explicitly."
            )
        }
        let runtimeRoot = explicitRoot ?? (requireInstalled ? installedRoot : nil)
        let progressHandler: (@Sendable (ChatProgress) -> Void)?
        if quiet {
            progressHandler = nil
        } else {
            progressHandler = { progress in
                guard progress.stage.rawValue != "generating" else { return }
                CLIStderr.write(
                    "[composer:\(progress.stage.rawValue)] \(progress.message ?? "")\n"
                )
            }
        }

        if !quiet {
            CLIStderr.write("Composing MiniMax Music 3 timeline with \(normalizedModelID)\n")
        }
        let runPass: (ChatPass) async throws -> MiniMaxMusic3CompositionReceipt = { chat in
                if !quiet {
                    CLIStderr.write("Planning MiniMax song timeline\n")
                }
                let blueprintPrompt = try MiniMaxMusic3ComposerContract.blueprintPrompt(request)
                var blueprintResponse = try await chat(Self.chatRequest(
                    prompt: blueprintPrompt,
                    maxTokens: 2_048
                ))
                var blueprint: MiniMaxMusic3SongBlueprint?
                for attempt in 0...1 {
                    do {
                        let proposed = try decode(
                            MiniMaxMusic3SongBlueprint.self,
                            from: blueprintResponse.response,
                            pass: "blueprint"
                        )
                        blueprint = try MiniMaxMusic3ComposerContract.normalize(
                            blueprint: proposed,
                            request: request
                        )
                        break
                    } catch {
                        guard attempt == 0 else { throw error }
                        if !quiet {
                            CLIStderr.write("Repairing MiniMax song timeline: \(errorMessage(error))\n")
                        }
                        blueprintResponse = try await chat(Self.chatRequest(
                            prompt: repairPrompt(
                                phase: "blueprint",
                                originalPrompt: blueprintPrompt,
                                previousResponse: blueprintResponse.response,
                                error: error
                            ),
                            maxTokens: 2_048
                        ))
                    }
                }
                guard let blueprint else {
                    throw ValidationError("MiniMax composer did not produce a blueprint.")
                }
                if !quiet {
                    CLIStderr.write(
                        "Composed \(blueprint.sections.count)-section, \(blueprint.bpm) BPM timeline\n"
                    )
                }

                if !quiet {
                    CLIStderr.write("Writing MiniMax lyrics and structured caption\n")
                }
                let songPrompt = try MiniMaxMusic3ComposerContract.songPrompt(
                    request,
                    blueprint: blueprint
                )
                var songResponse = try await chat(Self.chatRequest(
                    prompt: songPrompt,
                    maxTokens: 4_096
                ))
                var song: MiniMaxMusic3ComposedSong?
                for attempt in 0...1 {
                    do {
                        let proposed = try decode(
                            MiniMaxMusic3ComposedSong.self,
                            from: songResponse.response,
                            pass: "song"
                        )
                        song = try MiniMaxMusic3ComposerContract.normalize(
                            song: proposed,
                            request: request,
                            blueprint: blueprint
                        )
                        break
                    } catch {
                        guard attempt == 0 else { throw error }
                        if !quiet {
                            CLIStderr.write("Repairing MiniMax song inputs: \(errorMessage(error))\n")
                        }
                        songResponse = try await chat(Self.chatRequest(
                            prompt: repairPrompt(
                                phase: "song",
                                originalPrompt: songPrompt,
                                previousResponse: songResponse.response,
                                error: error
                            ),
                            maxTokens: 4_096
                        ))
                    }
                }
                guard let song else {
                    throw ValidationError("MiniMax composer did not produce finished song inputs.")
                }
                let preflight = MiniMaxMusic3LyricPreflight.inspect(
                    lyrics: song.lyrics,
                    durationSeconds: request.durationSeconds,
                    instrumental: request.instrumental,
                    blueprint: blueprint
                )
                return MiniMaxMusic3CompositionReceipt(
                    schemaVersion: MiniMaxMusic3CompositionReceipt.currentSchemaVersion,
                    createdAt: Date(),
                    modelID: normalizedModelID,
                    request: request,
                    blueprint: blueprint,
                    song: song,
                    lyricPreflight: preflight
                )
            }

        let receipt: MiniMaxMusic3CompositionReceipt
        if Gemma4Resources.handles(modelSpec: normalizedModelID) {
            let generator = Gemma4Generator(modelId: normalizedModelID)
            do {
                receipt = try await runPass { chatRequest in
                    try await generator.chat(
                        chatRequest,
                        modelPath: runtimeRoot,
                        progressHandler: progressHandler
                    )
                }
                await generator.unload()
            } catch {
                await generator.unload()
                throw error
            }
        } else {
            let generator = Q35Generator(modelId: normalizedModelID)
            do {
                receipt = try await runPass { chatRequest in
                    try await generator.chat(
                        chatRequest,
                        modelPath: runtimeRoot,
                        progressHandler: progressHandler
                    )
                }
                await generator.unload()
            } catch {
                await generator.unload()
                throw error
            }
        }
        MLX.Memory.clearCache()
        return receipt
    }

    private static func chatRequest(prompt: String, maxTokens: Int) -> ChatRequest {
        ChatRequest(
            messages: [
                ChatMessage(role: .system, content: MiniMaxMusic3ComposerContract.systemPrompt),
                ChatMessage(role: .user, content: prompt),
            ],
            maxTokens: maxTokens,
            temperature: 0.4,
            topP: 0.9,
            topK: 50,
            showThinking: false,
            requiresJSON: true
        )
    }

    private static func decode<Value: Decodable>(
        _ type: Value.Type,
        from response: String,
        pass: String
    ) throws -> Value {
        do {
            return try JSONDecoder().decode(Value.self, from: Data(response.utf8))
        } catch {
            throw ValidationError(
                "MiniMax composer \(pass) JSON is invalid: \(decodingDescription(error))"
            )
        }
    }

    private static func repairPrompt(
        phase: String,
        originalPrompt: String,
        previousResponse: String,
        error: Error
    ) -> String {
        """
        Correct the previous \(phase) JSON. Return exactly one corrected JSON object and no Markdown.

        Validation error:
        \(errorMessage(error))

        Original contract:
        \(originalPrompt)

        Previous response:
        \(previousResponse)
        """
    }

    private static func decodingDescription(_ error: Error) -> String {
        guard let error = error as? DecodingError else {
            return error.localizedDescription
        }
        switch error {
        case .keyNotFound(let key, let context):
            return "missing key '\(key.stringValue)' at \(codingPath(context.codingPath))"
        case .typeMismatch(let type, let context):
            return "expected \(type) at \(codingPath(context.codingPath)): \(context.debugDescription)"
        case .valueNotFound(let type, let context):
            return "missing \(type) at \(codingPath(context.codingPath))"
        case .dataCorrupted(let context):
            return "invalid data at \(codingPath(context.codingPath)): \(context.debugDescription)"
        @unknown default:
            return "unknown decoding error"
        }
    }

    private static func codingPath(_ path: [CodingKey]) -> String {
        let value = path.map(\.stringValue).joined(separator: ".")
        return value.isEmpty ? "root" : value
    }

    private static func errorMessage(_ error: Error) -> String {
        if let validationError = error as? ValidationError {
            return validationError.message
        }
        if let localizedError = error as? LocalizedError,
           let description = localizedError.errorDescription,
           !description.isEmpty
        {
            return description
        }
        return error.localizedDescription
    }
}
