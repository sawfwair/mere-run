import Foundation
import MereRunCore
import MereRunRelayKit
import UIKit

/// The on-device lane: small models running natively on this iPhone through
/// MereRunCore's memory-optimized iOS generators. Models install into the app
/// sandbox through the same managed store, catalog pins, and verified
/// snapshots as every other mere.run install. Experimental: first-run
/// generation exercises MLX Metal on the device.
@MainActor
final class LocalEngine: ObservableObject {
    /// One shared engine so Create and Chat agree on what is installed.
    static let shared = LocalEngine()

    struct Model: Identifiable, Equatable {
        enum Kind { case image, chat }

        let id: String
        let kind: Kind
        let title: String
        let detail: String

        var sizeLabel: String {
            guard let bytes = ManagedModelCatalog.spec(for: id)?.estimatedDownloadBytes else {
                return ""
            }
            return String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
        }
    }

    static let imageModels: [Model] = [
        Model(
            id: "image-klein-nano",
            kind: .image,
            title: "Klein nano",
            detail: "FLUX.2 Klein distilled to nano. The fastest local option."
        ),
        Model(
            id: "image-bonsai-binary",
            kind: .image,
            title: "Bonsai 1-bit",
            detail: "Bonsai 4B with binary weights. Small download, full pipeline."
        ),
        Model(
            id: "image-bonsai-ternary",
            kind: .image,
            title: "Bonsai ternary",
            detail: "Bonsai 4B with ternary weights. A quality step up from 1-bit."
        ),
    ]

    static let chatModel = Model(
        id: "text-chat-bonsai-27b-1bit",
        kind: .chat,
        title: "Bonsai 27B 1-bit",
        detail: "A 27B chat model in binary weights, running entirely on this iPhone."
    )

    enum ModelState: Equatable {
        case notInstalled
        case downloading(String)
        case ready
    }

    enum Activity: Equatable {
        case idle
        case generatingImage
        case chatting
    }

    @Published private(set) var states: [String: ModelState] = [:]
    @Published private(set) var activity: Activity = .idle
    @Published private(set) var lastImage: UIImage?
    @Published var selectedImageModelID = LocalEngine.imageModels[0].id
    @Published private(set) var lastError: String?

    private let imageGenerator = Flux2KleinGeneratoriOS()
    private var chatGenerator: Q35Generator?

    init() {
        refresh()
    }

    func refresh() {
        for model in Self.imageModels + [Self.chatModel] {
            if case .downloading = states[model.id] { continue }
            states[model.id] = ManagedModelResolver.resolveInstalledModel(id: model.id) != nil
                ? .ready
                : .notInstalled
        }
        if states[selectedImageModelID] != .ready,
           let installed = Self.imageModels.first(where: { states[$0.id] == .ready }) {
            selectedImageModelID = installed.id
        }
    }

    func state(of modelID: String) -> ModelState {
        states[modelID] ?? .notInstalled
    }

    func download(_ modelID: String) async {
        states[modelID] = .downloading("Starting…")
        lastError = nil
        do {
            _ = try await ManagedModelResolver.installManagedModel(
                id: modelID,
                progress: { progress in
                    let label: String
                    switch progress {
                    case .downloadingBytes(let completed, let total):
                        if let total, total > 0 {
                            label = "\(Int(Double(completed) / Double(total) * 100))% of \(total / 1_048_576) MB"
                        } else {
                            label = "\(completed / 1_048_576) MB"
                        }
                    case .downloadingPercent(let percent, _):
                        label = "\(percent)%"
                    case .extracting:
                        label = "Preparing…"
                    }
                    Task { @MainActor in
                        LocalEngine.shared.states[modelID] = .downloading(label)
                    }
                }
            )
            states[modelID] = .ready
        } catch {
            states[modelID] = .notInstalled
            lastError = error.localizedDescription
        }
    }

    func generateImage(prompt: String, width: Int = 512, height: Int = 512, steps: Int = 4) async {
        let modelID = selectedImageModelID
        guard state(of: modelID) == .ready, activity == .idle else { return }
        activity = .generatingImage
        lastError = nil
        defer { activity = .idle }
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("local-\(UUID().uuidString).png")
        let request = GenerationRequest(
            prompt: prompt,
            width: width,
            height: height,
            steps: steps,
            outputURL: output,
            model: modelID
        )
        do {
            let result = try await imageGenerator.generate(request, progressHandler: nil)
            lastImage = UIImage(contentsOfFile: result.outputURL.path)
        } catch {
            lastError = error.localizedDescription
        }
    }

    /// Runs one chat turn on-device, streaming visible token deltas to
    /// `onDelta` as the accumulated reply so far.
    func chat(
        messages: [ChatMessage],
        onDelta: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        let model = Self.chatModel
        guard let root = ManagedModelResolver.resolveInstalledModel(id: model.id) else {
            throw RelayAppError("Download \(model.title) before chatting on-device.")
        }
        guard activity == .idle else {
            throw RelayAppError("The on-device engine is busy with another run.")
        }
        activity = .chatting
        defer { activity = .idle }

        let generator = chatGenerator ?? Q35Generator(modelId: model.id)
        chatGenerator = generator

        let sampling = Q35Resources.recommendedSampling(forModelId: model.id)
        let request = ChatRequest(
            messages: messages,
            maxTokens: 1024,
            temperature: sampling?.temperature ?? 0.7,
            topP: sampling?.topP ?? 0.9,
            showThinking: false
        )

        let accumulated = StreamedText()
        let response = try await generator.chat(
            request,
            modelPath: root.path,
            progressHandler: { progress in
                guard progress.stage == .generating,
                      let piece = progress.message, !piece.isEmpty else { return }
                Task { @MainActor in
                    let text = accumulated.append(piece)
                    let visible = GeneratedTextFilters.strippingThinking(text, streaming: true)
                    if !visible.isEmpty { onDelta(visible) }
                }
            }
        )
        let cleaned = GeneratedTextFilters.strippingThinking(response.response)
        guard !cleaned.isEmpty else {
            throw RelayAppError("The model returned an empty reply.")
        }
        return cleaned
    }
}

/// Accumulates streamed pieces on the main actor.
@MainActor
private final class StreamedText {
    private var text = ""

    func append(_ piece: String) -> String {
        text += piece
        return text
    }
}

struct RelayAppError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
