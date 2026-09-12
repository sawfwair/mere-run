import Foundation
import MereRunCore
import MediaIO

extension APIServerContract {
    static let maxImageInferenceSteps = 100
    static let defaultImageModelID = ModelResolver.ModelID.zetaNano.rawValue

    static func decodeImageGenerationRequest(from data: Data) throws -> OpenAIImageGenerationRequest {
        try decodeJSONRequest(OpenAIImageGenerationRequest.self, from: data)
    }

    struct ImageGenerationPlan: Equatable, Sendable {
        let modelID: String
        let prompt: String
        let width: Int
        let height: Int
        let responseFormat: String
        let seed: UInt64?
        let negativePrompt: String?
        let steps: Int?
        let guidanceScale: Double?
        let inputImage: URL?
        let additionalInputImages: [URL]
        let maskImage: URL?
        let strength: Double?

        init(
            modelID: String,
            prompt: String,
            width: Int,
            height: Int,
            responseFormat: String,
            seed: UInt64?,
            negativePrompt: String?,
            steps: Int?,
            guidanceScale: Double?,
            inputImage: URL?,
            additionalInputImages: [URL] = [],
            maskImage: URL? = nil,
            strength: Double?
        ) {
            self.modelID = modelID
            self.prompt = prompt
            self.width = width
            self.height = height
            self.responseFormat = responseFormat
            self.seed = seed
            self.negativePrompt = negativePrompt
            self.steps = steps
            self.guidanceScale = guidanceScale
            self.inputImage = inputImage
            self.additionalInputImages = additionalInputImages
            self.maskImage = maskImage
            self.strength = strength
        }
    }

    static func imageGenerationPlan(
        from request: OpenAIImageGenerationRequest
    ) throws -> ImageGenerationPlan {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        if let n = request.n, n != 1 {
            throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
        }
        let steps = try imageInferenceSteps(request.steps)
        if let guidanceScale = request.guidance_scale,
           (!guidanceScale.isFinite || guidanceScale < 0) {
            throw APIRequestValidationError.invalidField("guidance_scale", "must be a positive number")
        }
        let size = try imageSize(from: request.size)
        let responseFormat = try imageResponseFormat(request.response_format)
        return ImageGenerationPlan(
            modelID: normalizedImageModelID(request.model),
            prompt: prompt,
            width: size.width,
            height: size.height,
            responseFormat: responseFormat,
            seed: request.seed,
            negativePrompt: normalizedOptional(request.negative_prompt),
            steps: steps,
            guidanceScale: request.guidance_scale,
            inputImage: nil,
            strength: nil
        )
    }

    static func imageEditPlan(
        from form: MultipartFormData,
        inputImageURL: URL
    ) throws -> ImageGenerationPlan {
        try imageEditPlan(from: form, inputImageURLs: [inputImageURL], maskImageURL: nil)
    }

    static func imageEditPlan(
        from form: MultipartFormData,
        inputImageURLs: [URL],
        maskImageURL: URL?
    ) throws -> ImageGenerationPlan {
        guard let inputImageURL = inputImageURLs.first else {
            throw APIRequestValidationError.invalidField("image", "image file is required")
        }
        let prompt = normalizedOptional(form.field("prompt")) ?? ""
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        if let rawN = normalizedOptional(form.field("n")) {
            guard let n = Int(rawN), n == 1 else {
                throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
            }
        }
        let steps = try imageInferenceSteps(
            optionalPositiveIntField(form.field("steps"), field: "steps")
        )
        let guidanceScale = try optionalPositiveDoubleField(form.field("guidance_scale"), field: "guidance_scale")
        let strength = try optionalUnitDoubleField(form.field("strength"), field: "strength")
        let size = try imageSize(from: form.field("size"))
        return ImageGenerationPlan(
            modelID: normalizedImageModelID(form.field("model")),
            prompt: prompt,
            width: size.width,
            height: size.height,
            responseFormat: try imageResponseFormat(form.field("response_format")),
            seed: try optionalUInt64Field(form.field("seed")),
            negativePrompt: normalizedOptional(form.field("negative_prompt")),
            steps: steps,
            guidanceScale: guidanceScale,
            inputImage: inputImageURL,
            additionalInputImages: Array(inputImageURLs.dropFirst()),
            maskImage: maskImageURL,
            strength: strength
        )
    }

    static func imageResponse(
        outputURL: URL,
        plan: ImageGenerationPlan,
        createdAt: Date = Date()
    ) throws -> OpenAIImageGenerationResponse {
        let datum: OpenAIImageGenerationData
        switch plan.responseFormat {
        case "url":
            datum = OpenAIImageGenerationData(url: outputURL.absoluteString, revised_prompt: plan.prompt)
        case "b64_json":
            let data = try Data(contentsOf: outputURL)
            datum = OpenAIImageGenerationData(
                b64_json: data.base64EncodedString(),
                revised_prompt: plan.prompt
            )
        default:
            throw APIRequestValidationError.invalidField("response_format", "unsupported response format")
        }
        return OpenAIImageGenerationResponse(
            created: Int(createdAt.timeIntervalSince1970),
            data: [datum]
        )
    }

    private static func imageSize(from rawValue: String?) throws -> (width: Int, height: Int) {
        guard let rawValue = normalizedOptional(rawValue), rawValue.lowercased() != "auto" else {
            return (1024, 1024)
        }
        let parts = rawValue.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let width = Int(parts[0]),
              let height = Int(parts[1]),
              width > 0,
              height > 0 else {
            throw APIRequestValidationError.invalidField("size", "expected WIDTHxHEIGHT, for example 1024x1024")
        }

        let maxDimension = 4_096
        guard width <= maxDimension, height <= maxDimension else {
            throw APIRequestValidationError.invalidField(
                "size",
                "width and height must each be at most \(maxDimension) pixels"
            )
        }

        // Validate the pixel product with division so attacker-controlled
        // dimensions can never overflow an Int before the request reaches a
        // tensor allocation.
        let maxPixels = 4_194_304
        guard width <= maxPixels / height else {
            throw APIRequestValidationError.invalidField(
                "size",
                "total image area must be at most \(maxPixels) pixels"
            )
        }

        let imageAlignment = 16
        guard width >= imageAlignment,
              height >= imageAlignment,
              width % imageAlignment == 0,
              height % imageAlignment == 0 else {
            throw APIRequestValidationError.invalidField(
                "size",
                "width and height must each be at least \(imageAlignment) pixels and divisible by \(imageAlignment)"
            )
        }
        return (width, height)
    }

    private static func imageResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "b64_json"
        guard value == "b64_json" || value == "url" else {
            throw APIRequestValidationError.invalidField("response_format", "expected b64_json or url")
        }
        return value
    }

    private static func imageInferenceSteps(_ value: Int?) throws -> Int? {
        guard let value else { return nil }
        guard (1...maxImageInferenceSteps).contains(value) else {
            throw APIRequestValidationError.invalidField(
                "steps",
                "must be between 1 and \(maxImageInferenceSteps)"
            )
        }
        return value
    }

    private static func optionalUInt64Field(_ rawValue: String?) throws -> UInt64? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = UInt64(rawValue) else {
            throw APIRequestValidationError.invalidField("seed", "must be an unsigned integer")
        }
        return value
    }

    private static func optionalPositiveDoubleField(_ rawValue: String?, field: String) throws -> Double? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Double(rawValue), value.isFinite, value >= 0 else {
            throw APIRequestValidationError.invalidField(field, "must be a positive number")
        }
        return value
    }

    private static func optionalUnitDoubleField(_ rawValue: String?, field: String) throws -> Double? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Double(rawValue), value.isFinite, (0...1).contains(value) else {
            throw APIRequestValidationError.invalidField(field, "must be between 0 and 1")
        }
        return value
    }

    private static func normalizedImageModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultImageModelID)
        switch modelID.lowercased() {
        case "gpt-image-1", "dall-e-3", "dall-e-2":
            return defaultImageModelID
        default:
            return modelID
        }
    }
}
