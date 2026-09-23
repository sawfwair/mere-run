import Foundation
import MereRunCore

extension APIServerContract {
    struct DiarizationPlan: Equatable, Sendable {
        let modelID: String
        let responseFormat: SpeechDiarizationOutputFormat
        let threshold: Float
        let minDuration: Float
        let mergeGap: Float
        let latency: Nemotron3DiarizationLatency
    }

    struct LiveDiarizationPlan: Equatable, Sendable {
        let modelID: String
        let latency: Nemotron3DiarizationLatency
        let threshold: Float
    }

    static func liveDiarizationPlan(parameters: [String: String]) throws -> LiveDiarizationPlan {
        guard Set(parameters.keys).isSubset(of: ["model", "latency", "threshold"]) else {
            throw APIRequestValidationError.invalidField("query", "unsupported live diarization parameter")
        }
        let modelID = normalizedOptional(parameters["model"])
            ?? ModelResolver.ModelID.nemotron3Diarization.rawValue
        guard modelID == ModelResolver.ModelID.nemotron3Diarization.rawValue else {
            throw APIRequestValidationError.invalidField("model", "expected speech-diarization-nemotron3")
        }
        let latencyValue = normalizedOptional(parameters["latency"]) ?? "1.04"
        guard let latency = Nemotron3DiarizationLatency(rawValue: latencyValue), latency != .offline else {
            throw APIRequestValidationError.invalidField("latency", "expected 1.04, 0.64, or 0.32")
        }
        return LiveDiarizationPlan(
            modelID: modelID,
            latency: latency,
            threshold: try diarizationFloat(parameters["threshold"], name: "threshold", defaultValue: 0.5, maximum: 1)
        )
    }

    static func diarizationPlan(from form: MultipartFormData) throws -> DiarizationPlan {
        try form.validateFields(
            textFields: ["model", "response_format", "threshold", "min_duration", "merge_gap", "latency"],
            fileFields: ["file"],
            unsupportedTextMessage: "unsupported diarization field",
            unsupportedFileMessage: "only file uploads are accepted"
        )
        guard form.files(named: "file").count == 1,
              let file = form.file(named: "file"), !file.body.isEmpty else {
            throw APIRequestValidationError.invalidField("file", "one nonempty audio file is required")
        }
        let modelID = normalizedOptional(form.field("model"))
            ?? ModelResolver.ModelID.nemotron3Diarization.rawValue
        guard modelID == ModelResolver.ModelID.nemotron3Diarization.rawValue
                || modelID == ModelResolver.ModelID.sortformerDiarization.rawValue else {
            throw APIRequestValidationError.invalidField("model", "expected a managed diarization model ID")
        }
        let formatValue = normalizedOptional(form.field("response_format"))?.lowercased() ?? "json"
        guard let responseFormat = SpeechDiarizationOutputFormat(rawValue: formatValue) else {
            throw APIRequestValidationError.invalidField("response_format", "expected json or rttm")
        }
        let latencyValue = normalizedOptional(form.field("latency")) ?? "offline"
        guard let latency = Nemotron3DiarizationLatency(rawValue: latencyValue) else {
            throw APIRequestValidationError.invalidField("latency", "expected offline, 1.04, 0.64, or 0.32")
        }
        if modelID == ModelResolver.ModelID.sortformerDiarization.rawValue, latency != .offline {
            throw APIRequestValidationError.invalidField("latency", "supported only by Nemotron 3")
        }
        return DiarizationPlan(
            modelID: modelID,
            responseFormat: responseFormat,
            threshold: try diarizationFloat(form.field("threshold"), name: "threshold", defaultValue: 0.5, maximum: 1),
            minDuration: try diarizationFloat(form.field("min_duration"), name: "min_duration", defaultValue: 0.25),
            mergeGap: try diarizationFloat(form.field("merge_gap"), name: "merge_gap", defaultValue: 0.25),
            latency: latency
        )
    }

    private static func diarizationFloat(
        _ raw: String?,
        name: String,
        defaultValue: Float,
        maximum: Float? = nil
    ) throws -> Float {
        guard let raw = normalizedOptional(raw) else { return defaultValue }
        guard let value = Float(raw), value.isFinite, value >= 0,
              maximum.map({ value <= $0 }) ?? true else {
            let description: String
            if let maximum {
                description = "expected a finite value from 0 through \(maximum)"
            } else {
                description = "expected a nonnegative finite value"
            }
            throw APIRequestValidationError.invalidField(name, description)
        }
        return value
    }
}
