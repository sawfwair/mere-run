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

    /// MLX cannot create a Metal device in the simulator; the on-device lane
    /// exists only on hardware. Constructing a generator in the simulator
    /// aborts the process, so every compute entry point gates on this first.
    static let isSupported: Bool = {
        #if targetEnvironment(simulator)
        return false
        #else
        return true
        #endif
    }()

    /// UI preview in the simulator (`MERERUN_UI_PREVIEW` launch argument):
    /// the on-device surfaces render with mocked install states so the design
    /// can be iterated without hardware. Compute stays gated on hardware.
    static let isPreview = ProcessInfo.processInfo.arguments.contains("MERERUN_UI_PREVIEW")

    /// Whether on-device surfaces appear at all.
    static var showsOnDeviceUI: Bool { isSupported || isPreview }

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
            title: "Klein Nano",
            detail: "Makes images. The fastest one."
        ),
        Model(
            id: "image-bonsai-binary",
            kind: .image,
            title: "Bonsai Binary",
            detail: "Makes images. The smallest download."
        ),
        Model(
            id: "image-bonsai-ternary",
            kind: .image,
            title: "Bonsai Ternary",
            detail: "Makes images. The best-looking of the three."
        ),
    ]

    static let chatModel = Model(
        id: "text-chat-bonsai-27b-1bit",
        kind: .chat,
        title: "Bonsai Chat",
        detail: "A 27-billion-parameter assistant. Your words never leave the phone."
    )

    static func model(withID id: String) -> Model? {
        (imageModels + [chatModel]).first { $0.id == id }
    }

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
    @Published private(set) var lastImageURL: URL?
    @Published var selectedImageModelID = LocalEngine.imageModels[0].id
    @Published private(set) var lastError: String?

    private lazy var imageGenerator = Flux2KleinGeneratoriOS()
    private var chatGenerator: Q35Generator?

    init() {
        refresh()
    }

    func refresh() {
        if Self.isPreview {
            // Mocked variety for design iteration in the simulator.
            if states.isEmpty {
                states = [
                    Self.imageModels[0].id: .ready,
                    Self.imageModels[1].id: .downloading("1204 of 3269 MB"),
                    Self.imageModels[2].id: .notInstalled,
                    Self.chatModel.id: .notInstalled,
                ]
            }
            return
        }
        guard Self.isSupported else { return }
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
        // A locked screen suspends the app and kills the transfer; keep the
        // display on for the duration of the pull.
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        do {
            _ = try await ManagedModelResolver.installManagedModel(
                id: modelID,
                progress: { progress in
                    let label: String
                    switch progress {
                    case .downloadingBytes(let completed, let total):
                        if let total, total > 0 {
                            label = "\(completed / 1_048_576) of \(total / 1_048_576) MB"
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
        guard Self.isSupported else { return }
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
            lastImageURL = result.outputURL
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
        guard Self.isSupported else {
            throw RelayAppError("On-device chat needs a physical iPhone.")
        }
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
