import ArgumentParser
import Crypto
import Foundation
import Hummingbird
import MereRunCore
import NIOCore

struct VisionServe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Serve resident, binary-frame vision grounding over HTTP.",
        discussion: """
        Starts a generic HTTP service backed by the native Falcon Perception runtime.
        The model remains resident across requests. Clients send image bytes and one or
        more text queries as multipart/form-data to POST /v1/vision/ground.

        The service is loopback-only by default. Non-loopback binds require --api-key.
        """
    )

    @Option(name: [.short, .long], help: "Port to listen on.")
    var port: Int = 8_091

    @Option(name: [.long], help: "Host to bind to.")
    var host: String = "127.0.0.1"

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Falcon Perception model root.")
    var model: String?

    @Option(name: [.long], help: "Bearer token required by inference endpoints.")
    var apiKey: String?

    @Option(name: [.customLong("max-frame-bytes")], help: "Maximum encoded image upload size.")
    var maxFrameBytes: Int = 16 * 1_024 * 1_024

    @Flag(name: [.customLong("preflight")], help: "Validate configuration without starting the server.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    mutating func validate() throws {
        guard (1...Int(UInt16.max)).contains(port) else {
            throw ValidationError("--port must be between 1 and 65535.")
        }
        guard maxFrameBytes > 0 else {
            throw ValidationError("--max-frame-bytes must be greater than zero.")
        }
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for vision serve.")
        }
        let normalizedKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        if !APIServe.isLoopbackHost(host), normalizedKey?.isEmpty != false {
            throw ValidationError("Binding vision serve to a non-loopback host requires --api-key.")
        }
    }

    func run() async throws {
        let resolvedModel = try VisionGround.resolveModelRoot(model)
        if preflight {
            let report = VisionServePreflightReport(
                status: "ready",
                capability: "vision.serve",
                modelID: resolvedModel.modelID,
                modelRoot: resolvedModel.rootURL.path,
                host: host,
                port: port,
                maximumFrameBytes: maxFrameBytes,
                authenticationRequired: apiKey?.isEmpty == false
            )
            if json {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                print(String(decoding: try encoder.encode(report), as: UTF8.self))
            } else {
                print("Ready: \(report.capability) with \(report.modelID)")
                print("Endpoint: http://\(report.host):\(report.port)/v1/vision/ground")
                print("Maximum frame bytes: \(report.maximumFrameBytes)")
            }
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: false)
        let runtime = try ResidentFalconGroundingRuntime(resolvedModel: resolvedModel)
        let server = VisionGroundingServer(
            runtime: runtime,
            apiKey: apiKey,
            maximumFrameBytes: maxFrameBytes
        )
        try await server.run(host: host, port: port)
    }
}

struct VisionServePreflightReport: Codable, Equatable, Sendable {
    let status: String
    let capability: String
    let modelID: String
    let modelRoot: String
    let host: String
    let port: Int
    let maximumFrameBytes: Int
    let authenticationRequired: Bool

    enum CodingKeys: String, CodingKey {
        case status
        case capability
        case modelID = "model_id"
        case modelRoot = "model_root"
        case host
        case port
        case maximumFrameBytes = "maximum_frame_bytes"
        case authenticationRequired = "authentication_required"
    }
}

struct VisionGroundingRequestPlan: Equatable, Sendable {
    let queries: [String]
    let streamID: String?
    let frameID: String?
    let capturedAt: String?
    let maxNewTokens: Int
    let segmentationThreshold: Float
}

struct VisionGroundingDetectionResponse: Codable, Equatable, Sendable {
    let index: Int
    let label: String
    let score: Float?
    let center: FalconPerceptionCenter
    let size: FalconPerceptionSize
    let box: FalconPerceptionBoundingBox
}

struct VisionGroundingTimingResponse: Codable, Equatable, Sendable {
    let inferenceSeconds: Double
    let totalSeconds: Double

    enum CodingKeys: String, CodingKey {
        case inferenceSeconds = "inference_seconds"
        case totalSeconds = "total_seconds"
    }
}

struct VisionGroundingResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let model: String
    let streamID: String?
    let frameID: String?
    let capturedAt: String?
    let imageSHA256: String
    let queries: [String]
    let detections: [VisionGroundingDetectionResponse]
    let timing: VisionGroundingTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case model
        case streamID = "stream_id"
        case frameID = "frame_id"
        case capturedAt = "captured_at"
        case imageSHA256 = "image_sha256"
        case queries
        case detections
        case timing
    }
}

enum VisionGroundingServerError: LocalizedError, Equatable {
    case invalidField(String, String)
    case unsupportedMediaType
    case invalidImageType

    var errorDescription: String? {
        switch self {
        case .invalidField(let field, let reason):
            return "Invalid '\(field)': \(reason)."
        case .unsupportedMediaType:
            return "Content-Type must be multipart/form-data."
        case .invalidImageType:
            return "Uploaded image must be PNG, JPEG, or WebP."
        }
    }
}

protocol VisionGroundingRuntime: Sendable {
    var modelID: String { get }

    func ground(
        imageURL: URL,
        queries: [String],
        outputDirectoryURL: URL,
        maxNewTokens: Int,
        segmentationThreshold: Float
    ) async throws -> FalconPerceptionGroundingRun
}

actor ResidentFalconGroundingRuntime: VisionGroundingRuntime {
    nonisolated let modelID: String
    private let grounder: FalconPerceptionGrounder

    init(resolvedModel: VisionGround.ResolvedModel) throws {
        self.modelID = resolvedModel.modelID
        self.grounder = try FalconPerceptionGrounder(
            modelRootURL: resolvedModel.rootURL,
            expectedModelID: resolvedModel.isManaged ? resolvedModel.modelID : nil
        )
    }

    deinit {
        grounder.unload()
    }

    func ground(
        imageURL: URL,
        queries: [String],
        outputDirectoryURL: URL,
        maxNewTokens: Int,
        segmentationThreshold: Float
    ) throws -> FalconPerceptionGroundingRun {
        try grounder.ground(
            imageURL: imageURL,
            queries: queries,
            annotatedImageURL: outputDirectoryURL.appendingPathComponent("annotated.png"),
            jsonOutputURL: outputDirectoryURL.appendingPathComponent("detections.json"),
            maxNewTokens: maxNewTokens,
            segmentationThreshold: segmentationThreshold
        )
    }
}

final class VisionGroundingServer: @unchecked Sendable {
    private let runtime: any VisionGroundingRuntime
    private let apiKey: String?
    private let maximumFrameBytes: Int
    private let fileManager: FileManager

    init(
        runtime: any VisionGroundingRuntime,
        apiKey: String?,
        maximumFrameBytes: Int,
        fileManager: FileManager = .default
    ) {
        self.runtime = runtime
        self.apiKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.maximumFrameBytes = maximumFrameBytes
        self.fileManager = fileManager
    }

    func run(host: String, port: Int) async throws {
        let app = Application(
            router: buildRouter(),
            configuration: .init(address: .hostname(host, port: port))
        )
        CLIStderr.write("Resident vision service: http://\(host):\(port)/v1/vision/ground\n")
        CLIStderr.write("Model: \(runtime.modelID)\n")
        try await app.runService()
    }

    func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        router.get("/health") { [self] request, _ in
            guard isAuthorized(request) else {
                return errorResponse(status: .unauthorized, message: "Invalid API key.")
            }
            return try jsonResponse([
                "status": "ok",
                "service": "mere.run/vision-grounding",
                "model": runtime.modelID,
            ])
        }
        router.post("/v1/vision/ground") { [self] request, _ in
            await handleGrounding(request)
        }
        return router
    }

    static func requestPlan(from form: MultipartFormData) throws -> VisionGroundingRequestPlan {
        let queries = form.parts
            .filter { ($0.name == "query" || $0.name == "query[]") && $0.filename == nil }
            .compactMap { String(data: $0.body, encoding: .utf8) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !queries.isEmpty else {
            throw VisionGroundingServerError.invalidField(
                "query",
                "at least one non-empty query or query[] field is required"
            )
        }
        guard queries.count <= 32 else {
            throw VisionGroundingServerError.invalidField("query", "at most 32 queries are allowed")
        }
        let maxNewTokens = try integerField(
            form.field("max_new_tokens"),
            name: "max_new_tokens",
            defaultValue: 512,
            range: 1...2_048
        )
        let segmentationThreshold = try floatField(
            form.field("segmentation_threshold"),
            name: "segmentation_threshold",
            defaultValue: 0.5,
            range: 0...1
        )
        return VisionGroundingRequestPlan(
            queries: queries,
            streamID: try identifierField(form.field("stream_id"), name: "stream_id"),
            frameID: try identifierField(form.field("frame_id"), name: "frame_id"),
            capturedAt: normalizedOptional(form.field("captured_at")),
            maxNewTokens: maxNewTokens,
            segmentationThreshold: segmentationThreshold
        )
    }

    static func response(
        from run: FalconPerceptionGroundingRun,
        plan: VisionGroundingRequestPlan,
        imageSHA256: String,
        startedAt: Date,
        inferenceStartedAt: Date,
        finishedAt: Date
    ) -> VisionGroundingResponse {
        VisionGroundingResponse(
            created: Int(finishedAt.timeIntervalSince1970),
            object: "vision.grounding",
            model: run.modelID,
            streamID: plan.streamID,
            frameID: plan.frameID,
            capturedAt: plan.capturedAt,
            imageSHA256: imageSHA256,
            queries: plan.queries,
            detections: run.detections.enumerated().map { index, detection in
                VisionGroundingDetectionResponse(
                    index: index,
                    label: detection.label,
                    score: detection.score,
                    center: detection.xy,
                    size: detection.hw,
                    box: detection.box
                )
            },
            timing: VisionGroundingTimingResponse(
                inferenceSeconds: finishedAt.timeIntervalSince(inferenceStartedAt),
                totalSeconds: finishedAt.timeIntervalSince(startedAt)
            )
        )
    }

    private func handleGrounding(_ request: Request) async -> Response {
        guard isAuthorized(request) else {
            return errorResponse(status: .unauthorized, message: "Invalid API key.")
        }
        let startedAt = Date()
        do {
            guard let boundary = APIServerContract.multipartBoundary(from: request.headers[.contentType]) else {
                throw VisionGroundingServerError.unsupportedMediaType
            }
            let body = try await request.body.collect(upTo: maximumFrameBytes + 1_024 * 1_024)
            let form = try MultipartFormData.parse(
                body: Data(body.readableBytesView),
                boundary: boundary
            )
            let plan = try Self.requestPlan(from: form)
            guard let image = form.file(named: "image") ?? form.file(named: "frame") else {
                throw VisionGroundingServerError.invalidField("image", "image or frame file is required")
            }
            guard !image.body.isEmpty, image.body.count <= maximumFrameBytes else {
                throw VisionGroundingServerError.invalidField(
                    "image",
                    "encoded image must contain 1...\(maximumFrameBytes) bytes"
                )
            }
            let fileExtension = try Self.imageExtension(for: image)
            let requestDirectory = fileManager.temporaryDirectory
                .appendingPathComponent("mere-run-vision-serve", isDirectory: true)
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            try fileManager.createDirectory(at: requestDirectory, withIntermediateDirectories: true)
            defer { try? fileManager.removeItem(at: requestDirectory) }
            let imageURL = requestDirectory.appendingPathComponent("frame").appendingPathExtension(fileExtension)
            try image.body.write(to: imageURL, options: [.atomic])
            let digest = SHA256.hash(data: image.body)
                .map { String(format: "%02x", $0) }
                .joined()
            let inferenceStartedAt = Date()
            let run = try await runtime.ground(
                imageURL: imageURL,
                queries: plan.queries,
                outputDirectoryURL: requestDirectory,
                maxNewTokens: plan.maxNewTokens,
                segmentationThreshold: plan.segmentationThreshold
            )
            let finishedAt = Date()
            return try jsonResponse(
                Self.response(
                    from: run,
                    plan: plan,
                    imageSHA256: digest,
                    startedAt: startedAt,
                    inferenceStartedAt: inferenceStartedAt,
                    finishedAt: finishedAt
                )
            )
        } catch {
            return errorResponse(
                status: error is VisionGroundingServerError || error is MultipartFormData.ParseError
                    ? .badRequest
                    : .internalServerError,
                message: error.localizedDescription
            )
        }
    }

    private static func imageExtension(for part: MultipartFormData.Part) throws -> String {
        switch part.contentType?.lowercased() {
        case "image/png": return "png"
        case "image/jpeg", "image/jpg": return "jpg"
        case "image/webp": return "webp"
        case "application/octet-stream", nil:
            let ext = part.filename.map { URL(fileURLWithPath: $0).pathExtension.lowercased() }
            guard let ext, ["png", "jpg", "jpeg", "webp"].contains(ext) else {
                throw VisionGroundingServerError.invalidImageType
            }
            return ext == "jpeg" ? "jpg" : ext
        default:
            throw VisionGroundingServerError.invalidImageType
        }
    }

    private static func integerField(
        _ raw: String?,
        name: String,
        defaultValue: Int,
        range: ClosedRange<Int>
    ) throws -> Int {
        guard let raw = normalizedOptional(raw) else { return defaultValue }
        guard let value = Int(raw), range.contains(value) else {
            throw VisionGroundingServerError.invalidField(name, "must be in \(range)")
        }
        return value
    }

    private static func floatField(
        _ raw: String?,
        name: String,
        defaultValue: Float,
        range: ClosedRange<Float>
    ) throws -> Float {
        guard let raw = normalizedOptional(raw) else { return defaultValue }
        guard let value = Float(raw), value.isFinite, range.contains(value) else {
            throw VisionGroundingServerError.invalidField(name, "must be between 0 and 1")
        }
        return value
    }

    private static func identifierField(_ raw: String?, name: String) throws -> String? {
        guard let value = normalizedOptional(raw) else { return nil }
        guard value.utf8.count <= 256,
              value.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }) else {
            throw VisionGroundingServerError.invalidField(name, "must be at most 256 bytes without control characters")
        }
        return value
    }

    private static func normalizedOptional(_ raw: String?) -> String? {
        guard let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        return value
    }

    private func isAuthorized(_ request: Request) -> Bool {
        guard let apiKey, !apiKey.isEmpty else { return true }
        return request.headers[.authorization] == "Bearer \(apiKey)"
    }

    private func jsonResponse<T: Encodable>(_ value: T) throws -> Response {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: try encoder.encode(value)))
        )
    }

    private func errorResponse(status: HTTPResponse.Status, message: String) -> Response {
        let data = (try? JSONEncoder().encode(["error": message]))
            ?? Data("{\"error\":\"vision service error\"}".utf8)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }
}
