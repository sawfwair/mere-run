import Foundation

public enum LTXPromptEnhancerError: LocalizedError {
    case emptyResponse

    public var errorDescription: String? {
        switch self {
        case .emptyResponse:
            return "The LTX prompt enhancer returned an empty caption."
        }
    }
}

public struct LTXPromptEnhancer {
    public static let maximumNewTokens = 600
    public static let noRepeatNgramSize = 5

    public static func enhance(
        prompt: String,
        modelID: String? = nil,
        modelRoot: URL? = nil,
        referenceImage: URL? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> String {
        let selectedModelID = modelID ?? (referenceImage == nil
            ? Gemma4Resources.twelveB4BitModelId
            : Gemma4Resources.visionTwelveBModelId)
        let generator = Gemma4Generator(modelId: selectedModelID)
        do {
            let response = try await generator.chat(
                ChatRequest(
                    messages: messages(prompt: prompt, referenceImage: referenceImage),
                    maxTokens: maximumNewTokens,
                    temperature: 0,
                    topP: 1,
                    showThinking: false,
                    noRepeatNgramSize: noRepeatNgramSize
                ),
                modelPath: modelRoot?.path,
                progressHandler: progressHandler
            )
            await generator.unload()
            let enhanced = cleanResponse(response.response)
            guard !enhanced.isEmpty else { throw LTXPromptEnhancerError.emptyResponse }
            return enhanced
        } catch {
            await generator.unload()
            throw error
        }
    }

    public static func messages(
        prompt: String,
        referenceImage: URL? = nil
    ) -> [ChatMessage] {
        if let referenceImage {
            return [
                ChatMessage(role: .system, content: imageToVideoSystemPrompt),
                ChatMessage(
                    role: .user,
                    content: "User Raw Input Prompt: \(prompt).",
                    imageUrl: referenceImage.absoluteString
                ),
            ]
        }
        return [
            ChatMessage(role: .system, content: textToVideoSystemPrompt),
            ChatMessage(role: .user, content: "user prompt: \(prompt)"),
        ]
    }

    public static func cleanResponse(_ text: String) -> String {
        let replacements: [Character: Character] = [
            "\u{2018}": "'",
            "\u{2019}": "'",
            "\u{201C}": "\"",
            "\u{201D}": "\"",
            "\u{2014}": "-",
            "\u{2013}": "-",
            "\u{00A0}": " ",
            "\u{2032}": "'",
            "\u{2212}": "-",
        ]
        let translated = String(text.map { replacements[$0] ?? $0 })
        guard let firstLetter = translated.firstIndex(where: \.isLetter) else {
            return translated
        }
        return String(translated[firstLetter...])
    }

    // These are the Gemma-4 captioning contracts shipped by LTX-2 v1.2.0.
    static let textToVideoSystemPrompt = """
    You are given a user's short text-to-video request. Write a single, highly detailed audio-visual caption describing the video that best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

    Match this captioning style precisely:

    1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

    2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

    3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

    4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
       - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up)
       - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene.
       - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle).
       Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

    5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

    6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep every stated action in motion.

    7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

    If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

    AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene LOOKS at its most visually striking, never adding new objects or actions. Keep everything else (framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
    """

    static let imageToVideoSystemPrompt = """
    You are given a REFERENCE IMAGE (the exact first frame of the video) and a user's short image-to-video request. Write a single, highly detailed audio-visual caption describing the video that BEGINS from this exact reference image and best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

    FIRST-FRAME / IMAGE GROUNDING (do this first): the opening of your caption must match the reference image exactly — same subject(s), identity, appearance, clothing, setting, lighting, and composition as shown. The video starts on this frame; describe it faithfully, then narrate chronologically as the user's requested action unfolds from it. Never contradict, replace, or invent things not consistent with the image. Single continuous take — no hard cuts.

    Match this captioning style precisely:

    1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

    2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

    3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements — grounded in and consistent with the reference image. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

    4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
       - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up) — consistent with how the reference image is framed at the start.
       - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene starting from this frame.
       - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle) — matching the reference image's viewpoint at the opening.
       Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

    5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

    6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep the user's requested motion/action central and in motion throughout.

    7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

    If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

    AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style or contradicting the reference image): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene, starting from this frame, LOOKS at its most visually striking, never adding new objects or actions and never contradicting the first frame. Keep everything else (first-frame grounding, framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
    """
}
