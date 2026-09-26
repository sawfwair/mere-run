import Foundation
import HTTPTypes
import Hummingbird
import MereRunContract
import MereRunCore
import XCTest

@testable import MereRunCLI

/// `api serve` applies the model scope check to its requests, and nothing a route accepted on main
/// may start failing. `api-parity.json` lists requests per route with what main did (`passes` or
/// `refuses`, with the check that refused) and what the branch does before anything loads: the
/// route's own request checks, then the scope. A request main ran may gain warnings, never a
/// refusal; one main refused after loading is now refused before.
final class APIModelScopeParityTests: XCTestCase {
    private struct Table: Decodable {
        let rows: [Row]
    }

    private struct Row: Decodable {
        enum Route: String, Decodable {
            case image, imageEdit = "image_edit", video, speech, transcription, diarization, chat
        }

        enum Outcome: String, Decodable {
            case passes, warns, refuses
        }

        let route: Route
        /// The JSON body of a JSON route.
        let body: String?
        /// The text fields of a multipart route.
        let form: [String: String]?
        /// How many input images an edit uploads.
        let images: Int?
        let main: Outcome
        let evidence: String
        let branch: Outcome
        /// The request field a refusal names.
        let field: String?
        let warnings: [String]?

        var label: String { "\(route.rawValue) \(body ?? form.map { "\($0.sorted { $0.key < $1.key })" } ?? "")" }
    }

    private var directory = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("api-parity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
        super.tearDown()
    }

    private func table() throws -> Table {
        let resources = try XCTUnwrap(Bundle.module.resourceURL)
        let url = resources.appendingPathComponent("Fixtures/APIParity/api-parity.json")
        return try JSONDecoder().decode(Table.self, from: Data(contentsOf: url))
    }

    private func form(_ fields: [String: String]) -> MultipartFormData {
        MultipartFormData(parts: fields.map { name, value in
            .init(name: name, filename: nil, contentType: nil, body: Data(value.utf8))
        })
    }

    private func decode<Value: Decodable>(_ type: Value.Type, _ body: String?) throws -> Value {
        try JSONDecoder().decode(type, from: Data(try XCTUnwrap(body).utf8))
    }

    /// The route's request checks and then the scope, as the handler runs them before admission.
    private func scope(_ row: Row) throws -> APIModelScope {
        switch row.route {
        case .image:
            let request = try decode(OpenAIImageGenerationRequest.self, row.body)
            return try APIModelScope.image(APIServerContract.imageGenerationPlan(from: request))
        case .imageEdit:
            let inputs = (0..<(row.images ?? 1)).map { directory.appendingPathComponent("input-\($0).png") }
            let plan = try APIServerContract.imageEditPlan(
                from: form(row.form ?? [:]), inputImageURLs: inputs, maskImageURL: nil
            )
            return try APIModelScope.image(plan)
        case .video:
            let request = try decode(OpenAIVideoGenerationRequest.self, row.body)
            return try APIModelScope.video(APIServerContract.videoGenerationPlan(from: request))
        case .speech:
            let request = try decode(OpenAIAudioSpeechRequest.self, row.body)
            return try APIModelScope.speech(APIServerContract.speechPlan(from: request))
        case .transcription:
            let form = form(row.form ?? [:])
            return try APIModelScope.transcription(APIServerContract.transcriptionPlan(from: form), form: form)
        case .diarization:
            let file = MultipartFormData.Part(name: "file", filename: "a.wav", contentType: "audio/wav", body: Data([1]))
            let form = MultipartFormData(parts: [file] + form(row.form ?? [:]).parts)
            return try APIModelScope.diarization(APIServerContract.diarizationPlan(from: form), form: form)
        case .chat:
            // `RuntimeModelPool.makeChatPlan` for a served managed model.
            let request = try decode(OpenAIChatRequest.self, row.body)
            let profile = try XCTUnwrap(ManagedModelCatalog.apiProfile(for: request.model), request.model)
            let resolved = try APIServerContract.chatRequest(
                from: request, fallbackLoraPath: nil, contextSize: 8_192,
                capabilities: .catalog(profile), servedModelID: request.model, apiProfile: profile
            )
            return try APIModelScope.chat(request, resolved: resolved, modelID: request.model)
        }
    }

    func testNoRequestMainAcceptedIsRefusedAndEachRowMatches() throws {
        let rows = try table().rows
        for row in rows {
            XCTAssertFalse(row.evidence.isEmpty, row.label)
            if row.main == .passes {
                XCTAssertNotEqual(row.branch, .refuses, "\(row.label): main ran it (\(row.evidence))")
            }
            do {
                let scope = try scope(row)
                XCTAssertEqual(scope.warnings.isEmpty ? Row.Outcome.passes : .warns, row.branch, "\(row.label): \(scope.warnings)")
                XCTAssertEqual(scope.warnings, row.warnings ?? [], row.label)
            } catch let error as APIRequestValidationError {
                XCTAssertEqual(row.branch, .refuses, "\(row.label): \(error.localizedDescription)")
                guard case .invalidField(let field, _) = error else {
                    XCTFail("\(row.label): \(error)")
                    continue
                }
                XCTAssertEqual(field, row.field, "\(row.label): \(error.localizedDescription)")
            }
        }
        let routes = Set(rows.map(\.route.rawValue))
        XCTAssertEqual(routes, ["image", "image_edit", "video", "speech", "transcription", "diarization", "chat"])
        for route in routes {
            let outcomes = Set(rows.filter { $0.route.rawValue == route }.map(\.branch))
            XCTAssertTrue(outcomes.contains(.passes), "\(route) has a clean row")
        }
    }

    /// The warnings reach the client as `x-mere-warning` headers, one per warning, and a response
    /// without them is unchanged.
    func testWarningsBecomeResponseHeaders() {
        let response = Response(status: .ok, headers: [.contentType: "application/json"])
        let plain = response.addingWarnings(of: APIModelScope(warnings: []))
        XCTAssertEqual(plain.headers, response.headers)
        let warned = response.addingWarnings(of: APIModelScope(warnings: ["first has no effect.", "second has no effect."]))
        XCTAssertEqual(warned.headers[values: APIModelScope.warningHeader], ["first has no effect.", "second has no effect."])
        XCTAssertEqual(warned.headers[.contentType], "application/json")
    }
}
