import Foundation
import MereRunCore
import XCTest

@testable import MereRunCLI

final class APIDiarizationContractTests: XCTestCase {
    func testPlanParsesNativeOptions() throws {
        let plan = try APIServerContract.diarizationPlan(from: form([
            "model": ModelResolver.ModelID.nemotron3Diarization.rawValue,
            "response_format": "rttm",
            "threshold": "0.42",
            "min_duration": "0.1",
            "merge_gap": "0.2",
            "latency": "0.64",
        ]))
        XCTAssertEqual(plan.modelID, ModelResolver.ModelID.nemotron3Diarization.rawValue)
        XCTAssertEqual(plan.responseFormat, .rttm)
        XCTAssertEqual(plan.threshold, 0.42)
        XCTAssertEqual(plan.minDuration, 0.1)
        XCTAssertEqual(plan.mergeGap, 0.2)
        XCTAssertEqual(plan.latency, .low)
    }

    func testPlanRejectsAmbiguousOrInvalidUploads() {
        let unsupported = form(["model": "../models/private", "latency": "0.32"])
        XCTAssertThrowsError(try APIServerContract.diarizationPlan(from: unsupported))
        XCTAssertThrowsError(try APIServerContract.diarizationPlan(from: form(["threshold": "nan"])))
        XCTAssertThrowsError(try APIServerContract.diarizationPlan(from: form(["latency": "fast"])))
        XCTAssertThrowsError(try APIServerContract.diarizationPlan(from: form([
            "model": ModelResolver.ModelID.sortformerDiarization.rawValue,
            "latency": "0.64",
        ])))
        let missing = MultipartFormData(parts: [])
        XCTAssertThrowsError(try APIServerContract.diarizationPlan(from: missing))
        let duplicated = MultipartFormData(parts: form([:]).parts + form([:]).parts)
        XCTAssertThrowsError(try APIServerContract.diarizationPlan(from: duplicated))
    }

    private func form(_ fields: [String: String]) -> MultipartFormData {
        let file = MultipartFormData.Part(
            name: "file", filename: "meeting.wav", contentType: "audio/wav", body: Data([1, 2, 3])
        )
        return MultipartFormData(parts: [file] + fields.map { name, value in
            .init(name: name, filename: nil, contentType: nil, body: Data(value.utf8))
        })
    }
}
