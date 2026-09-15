import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class APIMultipartPolicyTests: XCTestCase {
    private typealias Plan = (MultipartFormData) throws -> Void

    func testVisionRoutesRejectDuplicateTextFields() {
        let plans: [(String, Plan)] = [
            ("resolution", { _ = try APIServerContract.imageTo3DPlan(from: $0) }),
            ("resolution", { _ = try APIServerContract.instantMeshPlan(from: $0) }),
            ("process_resolution", { _ = try APIServerContract.multiViewGeometryPlan(from: $0) }),
            ("max_frames", { _ = try APIServerContract.depthVideoPlan(from: $0) })
        ]
        for (field, plan) in plans {
            let form = MultipartFormData(parts: [
                .init(name: field, filename: nil, contentType: nil, body: Data("32".utf8)),
                .init(name: field, filename: nil, contentType: nil, body: Data("64".utf8))
            ])
            XCTAssertThrowsError(try plan(form)) { error in
                XCTAssertEqual(
                    error as? APIRequestValidationError,
                    .invalidField(field, "must be supplied at most once")
                )
            }
        }
    }

    func testImageRoutesRequireUTF8TextBeforePlanningUploads() {
        let plans: [Plan] = [
            { _ = try APIServerContract.imageTo3DPlan(from: $0) },
            { _ = try APIServerContract.instantMeshPlan(from: $0) },
            { _ = try APIServerContract.multiViewGeometryPlan(from: $0) }
        ]
        let form = MultipartFormData(parts: [
            .init(name: "model", filename: nil, contentType: nil, body: Data([0xff]))
        ])
        for plan in plans {
            XCTAssertThrowsError(try plan(form)) { error in
                XCTAssertEqual(
                    error as? APIRequestValidationError,
                    .invalidField("model", "must contain valid UTF-8 text")
                )
            }
        }
    }

    func testVideoDepthPreservesOptionalUTF8FieldDecoding() throws {
        let form = MultipartFormData(parts: [
            .init(name: "input_size", filename: nil, contentType: nil, body: Data([0xff])),
            .init(name: "video", filename: "input.mp4", contentType: "video/mp4", body: Data([1]))
        ])
        let plan = try APIServerContract.depthVideoPlan(from: form)
        XCTAssertEqual(plan.modelID, APIServerContract.defaultDepthVideoModelID)
        XCTAssertEqual(plan.inputSize, VideoDepthAnythingLimits.defaultInputSize)
        XCTAssertEqual(plan.maximumFrameCount, VideoDepthAnythingLimits.defaultMaximumFrameCount)
    }

    func testUnknownFileFieldsKeepRouteSpecificDiagnostics() {
        let plans: [(String, Plan)] = [
            ("only one uploaded image file is accepted", {
                _ = try APIServerContract.imageTo3DPlan(from: $0)
            }),
            ("only uploaded image or image[] view files are accepted", {
                _ = try APIServerContract.instantMeshPlan(from: $0)
            }),
            ("unsupported file part; only uploaded image/image[] and cameras JSON files are accepted", {
                _ = try APIServerContract.multiViewGeometryPlan(from: $0)
            }),
            ("only a single uploaded 'video' file is accepted", {
                _ = try APIServerContract.depthVideoPlan(from: $0)
            })
        ]
        let form = MultipartFormData(parts: [
            .init(name: "output", filename: "output.bin", contentType: nil, body: Data([1]))
        ])
        for (message, plan) in plans {
            XCTAssertThrowsError(try plan(form)) { error in
                XCTAssertEqual(error as? APIRequestValidationError, .invalidField("output", message))
            }
        }
    }
}
