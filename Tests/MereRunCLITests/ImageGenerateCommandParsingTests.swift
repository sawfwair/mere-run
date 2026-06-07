import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ImageGenerateCommandParsingTests: XCTestCase {
    func testDefaultManagedImageModelIsNano() {
        XCTAssertEqual(ImageGenerate.defaultManagedModelID, .zetaNano)
    }

    func testParsesHiDreamReferenceOptions() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "place this subject in a studio",
            "--model", "image-hidream-o1-dev",
            "--ref-image", "/tmp/ref1.png",
            "--ref-image", "/tmp/ref2.png",
            "--keep-original-aspect",
            "--width", "1024",
            "--height", "768",
        ])

        XCTAssertEqual(cmd.prompt, "place this subject in a studio")
        XCTAssertEqual(cmd.model, "image-hidream-o1-dev")
        XCTAssertEqual(cmd.referenceImages, ["/tmp/ref1.png", "/tmp/ref2.png"])
        XCTAssertTrue(cmd.keepOriginalAspect)
        XCTAssertEqual(cmd.width, 1024)
        XCTAssertEqual(cmd.height, 768)
        XCTAssertNil(cmd.steps)
        XCTAssertNil(cmd.cfgScale)
    }

    func testParsesExplicitHiDreamStepAndCFGOverrides() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a brass camera",
            "--model", "image-hidream-o1",
            "--steps", "4",
            "--cfg", "1.0",
        ])

        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.cfgScale, 1.0)
    }

    func testParsesStructuredPromptOptions() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "a knight and a white horse",
            "--model", "image-ideogram4-sdnq-uint4",
            "--structured-prompt",
            "--structured-prompt-model", "text-chat-q36-nano",
            "--structured-prompt-model-root", "/tmp/q36",
            "--structured-prompt-max-tokens", "3072",
            "--structured-prompt-output", "/tmp/prompt.json",
        ])

        XCTAssertTrue(cmd.structuredPrompt)
        XCTAssertEqual(cmd.structuredPromptModel, "text-chat-q36-nano")
        XCTAssertEqual(cmd.structuredPromptModelRoot, "/tmp/q36")
        XCTAssertEqual(cmd.structuredPromptMaxTokens, 3_072)
        XCTAssertEqual(cmd.structuredPromptOutput, "/tmp/prompt.json")
    }

    func testParsesJSONPromptAlias() throws {
        let cmd = try ImageGenerate.parse([
            "--prompt", "editorial product photo",
            "--json-prompt",
        ])

        XCTAssertTrue(cmd.structuredPrompt)
        XCTAssertEqual(cmd.structuredPromptModel, "text-chat-gemma4-12b-4bit")
    }

    func testStructuredPromptAdapterValidatesPaperSchema() throws {
        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(Self.validStructuredCaptionJSON)

        XCTAssertEqual(caption.shortDescription, "A knight riding a white horse in a sunlit meadow.")
        XCTAssertEqual(caption.objects.first?.shapeAndColor, "armored human figure in silver and blue")
        XCTAssertEqual(caption.textRender, [])
    }

    func testStructuredPromptAdapterCleansJSONCodeFence() throws {
        let wrapped = """
        <think>draft</think>
        ```json
        \(Self.validStructuredCaptionJSON)
        ```
        """

        let cleaned = StructuredImagePromptAdapter.cleanedJSONCandidate(from: wrapped)
        XCTAssertNoThrow(try StructuredImagePromptAdapter.validateCaptionJSON(cleaned))
    }

    func testStructuredPromptAdapterNormalizesQ36NearSchema() throws {
        let normalized = try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.q36NearStructuredCaptionJSON,
            fallbackPrompt: "a small matte red cube on a white table"
        )

        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(normalized)
        XCTAssertEqual(caption.shortDescription, "A clean product photograph featuring a small matte red cube.")
        XCTAssertEqual(caption.objects.first?.description, "matte red cube, small, matte finish, red color")
        XCTAssertEqual(caption.objects.first?.texture, "matte surface")
        XCTAssertEqual(caption.textRender, [])
    }

    func testStructuredPromptAdapterNormalizesCompactGemmaSchema() throws {
        let normalized = try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.gemmaCompactStructuredCaptionJSON,
            fallbackPrompt: "a small matte red cube on a white table"
        )

        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(normalized)
        XCTAssertEqual(caption.objects.first?.description, "small matte red cube")
        XCTAssertEqual(caption.lighting.conditions, "Soft window light, natural diffusion")
        XCTAssertEqual(caption.photographicCharacteristics.depthOfField, "Product photography, sharp focus")
        XCTAssertEqual(caption.textRender, [])
    }

    func testQuotedStringsExtraction() {
        let prompt = """
        a trailhead sign reads 'THE LOCAL WILD' with a smaller line below reading \
        'do not feed the models', the sign's letters carved into wood, titled "Field Notes"
        """
        let quoted = StructuredImagePromptAdapter.quotedStrings(in: prompt)
        XCTAssertEqual(quoted, ["Field Notes", "THE LOCAL WILD", "do not feed the models"])
    }

    func testQuotedStringsIgnoresApostrophes() {
        let quoted = StructuredImagePromptAdapter.quotedStrings(
            in: "the sign's weathered face catches the morning's first light"
        )
        XCTAssertEqual(quoted, [])
    }

    func testEnsuringTextRenderInjectsQuotedPromptText() throws {
        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(Self.validStructuredCaptionJSON)
        XCTAssertEqual(caption.textRender, [])

        let ensured = StructuredImagePromptAdapter.ensuringTextRender(
            caption,
            prompt: "a banner above the knight reads 'ONWARD'"
        )
        XCTAssertEqual(ensured.textRender.count, 1)
        XCTAssertEqual(ensured.textRender.first?.text, "ONWARD")
    }

    func testEnsuringTextRenderKeepsModelProvidedEntries() throws {
        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(Self.validStructuredCaptionJSON)
        let withEntry = StructuredImagePromptAdapter.ensuringTextRender(
            caption,
            prompt: "a banner reads 'ONWARD'"
        )
        // A second pass must not duplicate or overwrite the existing entry.
        let unchanged = StructuredImagePromptAdapter.ensuringTextRender(
            withEntry,
            prompt: "a banner reads 'SOMETHING ELSE'"
        )
        XCTAssertEqual(unchanged.textRender, withEntry.textRender)
    }

    func testNormalizedCaptionJSONInjectsTextRenderForQuotedPrompt() throws {
        let normalized = try StructuredImagePromptAdapter.normalizedCaptionJSON(
            from: Self.validStructuredCaptionJSON,
            fallbackPrompt: "a knight under a banner that reads 'ONWARD', sunny meadow"
        )
        let caption = try StructuredImagePromptAdapter.validateCaptionJSON(normalized)
        XCTAssertEqual(caption.textRender.map(\.text), ["ONWARD"])
    }

    func testIsParseableJSONDistinguishesNearMissFromTokenSalad() {
        XCTAssertTrue(StructuredImagePromptAdapter.isParseableJSON(#"{"objects": "wrong shape"}"#))
        XCTAssertFalse(StructuredImagePromptAdapter.isParseableJSON(
            "{ed feetization mas Dod asked Lolamente lesser0 or<audio|>"
        ))
        XCTAssertFalse(StructuredImagePromptAdapter.isParseableJSON(""))
    }

    private static let validStructuredCaptionJSON = """
    {
      "short description": "A knight riding a white horse in a sunlit meadow.",
      "objects": [
        {
          "description": "A calm medieval knight seated on a horse, wearing polished armor and a blue cloak.",
          "location": "center foreground",
          "relationship": "The knight is riding the horse and looking toward the horizon.",
          "relative size": "large within frame",
          "shape and color": "armored human figure in silver and blue",
          "texture": "metallic armor and woven fabric",
          "appearance details": "helmet visor raised, cloak moving lightly",
          "number of objects": null,
          "pose": "upright seated riding pose",
          "expression": "calm and focused",
          "clothing": "plate armor and blue cloak",
          "action": "riding a horse",
          "gender": "unspecified",
          "skin tone and texture": null,
          "orientation": "facing right"
        }
      ],
      "background setting": "Open meadow with distant trees and low hills under a clear sky.",
      "lighting": {
        "conditions": "bright daylight",
        "direction": "side-lit from left",
        "shadows": "soft shadows falling to the right"
      },
      "aesthetics": {
        "composition": "centered heroic composition",
        "color scheme": "natural greens with silver and blue accents",
        "mood atmosphere": "noble and serene"
      },
      "photographic characteristics": {
        "depth of field": "moderate",
        "focus": "sharp focus on knight and horse",
        "camera angle": "eye-level",
        "lens focal length": "normal lens"
      },
      "style medium": "digital illustration",
      "text render": [],
      "context": "Fantasy character illustration suitable for concept art.",
      "artistic style": "cinematic realism"
    }
    """

    private static let q36NearStructuredCaptionJSON = """
    {
      "short description": "A clean product photograph featuring a small matte red cube.",
      "objects": [
        {
          "name": "matte red cube",
          "attributes": [
            "small",
            "matte finish",
            "red color"
          ]
        }
      ],
      "background setting": "Minimalist white tabletop and clean neutral background.",
      "lighting": {
        "conditions": "Soft, diffused natural light",
        "direction": "From a side window",
        "shadows": "Soft, subtle shadows"
      },
      "aesthetics": {
        "composition": "Centered product composition",
        "color scheme": "Red and white with neutral tones",
        "mood atmosphere": "Clean and professional"
      },
      "photographic characteristics": {
        "depth of field": "Shallow to medium",
        "focus": "Sharp focus on the cube",
        "camera angle": "Slightly elevated",
        "lens focal length": "50mm"
      },
      "style medium": "photograph",
      "artistic style": "minimal product",
      "context": "Product showcase",
      "text render": null
    }
    """

    private static let gemmaCompactStructuredCaptionJSON = """
    ```json
    {
      "short description": "A small matte red cube on a white table.",
      "objects": [
        "small matte red cube",
        "white table"
      ],
      "background setting": "Minimalist interior, clean studio environment",
      "lighting": "Soft window light, natural diffusion",
      "aesthetics": "Minimalist, clean, modern",
      "photographic characteristics": "Product photography, sharp focus",
      "style medium": "Photography",
      "artistic style": "Commercial product photography",
      "context": "Product showcase",
      "text render": "none"
    }
    ```
    """
}
