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
        /// Minimum device memory this model can realistically wire; nil = any.
        var minimumMemoryGB: Int?
        /// Shown before download when the upstream license carries terms.
        var licenseNote: String?

        var isCompatible: Bool {
            guard let minimumMemoryGB else { return true }
            return ProcessInfo.processInfo.physicalMemory >= UInt64(minimumMemoryGB) * 1_000_000_000
        }

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

    static let chatModels: [Model] = [
        Model(
            id: "text-chat-lfm25-2.6b-4bit",
            kind: .chat,
            title: "Liquid Chat",
            detail: "A quick assistant sized for a phone. Your words never leave it.",
            licenseNote: "LFM Open License: free for personal use and for companies under USD 10M revenue."
        ),
        Model(
            id: "text-chat-bonsai-27b-1bit",
            kind: .chat,
            title: "Bonsai Chat",
            detail: "A 27-billion-parameter assistant for devices with 12 GB of memory or more.",
            minimumMemoryGB: 12
        ),
    ]

    static var allModels: [Model] { imageModels + chatModels }

    static func model(withID id: String) -> Model? {
        allModels.first { $0.id == id }
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
    @Published var selectedChatModelID: String {
        didSet { UserDefaults.standard.set(selectedChatModelID, forKey: "local.selectedChatModelID") }
    }
    /// Which chat model currently has its weights resident, if any.
    @Published private(set) var warmChatModelID: String?
    /// Human status for the pending bubble ("Loading Liquid Chat…").
    @Published private(set) var chatStatus: String?
    @Published private(set) var lastError: String?

    private lazy var imageGenerator = Flux2KleinGeneratoriOS()
    private var q35Generators: [String: Q35Generator] = [:]
    private var lfm2Generators: [String: LFM2Generator] = [:]

    private enum AnyChatGenerator {
        case q35(Q35Generator)
        case lfm2(LFM2Generator)

        func prepare(modelPath: String, progressHandler: (@Sendable (ChatProgress) -> Void)?) async throws {
            switch self {
            case .q35(let generator): try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
            case .lfm2(let generator): try await generator.prepare(modelPath: modelPath, progressHandler: progressHandler)
            }
        }

        func chat(
            _ request: ChatRequest,
            modelPath: String,
            progressHandler: (@Sendable (ChatProgress) -> Void)?
        ) async throws -> ChatResponse {
            switch self {
            case .q35(let generator): try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
            case .lfm2(let generator): try await generator.chat(request, modelPath: modelPath, progressHandler: progressHandler)
            }
        }

        func unload() async {
            switch self {
            case .q35(let generator): await generator.unload()
            case .lfm2(let generator): await generator.unload()
            }
        }
    }

    private func chatGenerator(for model: Model) -> AnyChatGenerator {
        if model.id.hasPrefix("text-chat-lfm") {
            let generator = lfm2Generators[model.id] ?? LFM2Generator(modelId: model.id)
            lfm2Generators[model.id] = generator
            return .lfm2(generator)
        }
        let generator = q35Generators[model.id] ?? Q35Generator(modelId: model.id)
        q35Generators[model.id] = generator
        return .q35(generator)
    }

    /// Loads the selected chat model's weights ahead of the first message so
    /// it answers immediately. Safe to call repeatedly; the generators cache.
    func warmChat() {
        guard Self.isSupported,
              let model = Self.model(withID: selectedChatModelID),
              warmChatModelID != model.id,
              let root = ManagedModelResolver.resolveInstalledModel(id: model.id) else { return }
        let generator = chatGenerator(for: model)
        chatStatus = "Loading \(model.title)…"
        Task { [weak self] in
            do {
                try await generator.prepare(modelPath: root.path, progressHandler: nil)
                await MainActor.run {
                    self?.warmChatModelID = model.id
                    self?.chatStatus = nil
                }
            } catch {
                await MainActor.run { self?.chatStatus = nil }
            }
        }
    }

    /// Frees resident chat weights (an image generation is about to need the
    /// memory, or the user left the on-device lane).
    func coolChat() {
        let generators = lfm2Generators.values.map(AnyChatGenerator.lfm2)
            + q35Generators.values.map(AnyChatGenerator.q35)
        warmChatModelID = nil
        Task {
            for generator in generators {
                await generator.unload()
            }
        }
    }

    init() {
        let saved = UserDefaults.standard.string(forKey: "local.selectedChatModelID")
        selectedChatModelID = Self.chatModels.first { $0.id == saved }?.id ?? Self.chatModels[0].id
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
                    Self.chatModels[0].id: .notInstalled,
                    Self.chatModels[1].id: .notInstalled,
                ]
            }
            return
        }
        guard Self.isSupported else { return }
        // The app container moves on every update, dangling any absolute
        // symlinks a previous install wrote; heal them before scanning.
        ManagedModelResolver.repairRelocatedInstalls()
        for model in Self.allModels {
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

    func download(_ modelID: String, acceptedTerms: Bool = false) async {
        states[modelID] = .downloading("Starting…")
        lastError = nil
        // A locked screen suspends the app and kills the transfer; keep the
        // display on for the duration of the pull.
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        do {
            _ = try await ManagedModelResolver.installManagedModel(
                id: modelID,
                usageTermsAcknowledged: acceptedTerms,
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

    /// Removes an installed model and reclaims its unshared hub-cache
    /// payloads, mirroring `mere.run model remove`. Blobs shared with another
    /// installed model are preserved.
    func delete(_ modelID: String) {
        guard Self.isSupported, activity == .idle,
              let installURL = ManagedModelResolver.resolveInstalledModel(id: modelID) else { return }
        lastError = nil
        do {
            let storage = try ModelStorageManager()
            let cacheUnits = try storage.cacheUnitsReferenced(by: modelID)
            try FileManager.default.removeItem(at: installURL)
            let plan = try storage.garbageCollectionPlan(limitingTo: cacheUnits)
            _ = try storage.execute(plan)
        } catch {
            lastError = error.localizedDescription
        }
        q35Generators[modelID] = nil
        lfm2Generators[modelID] = nil
        refresh()
        refreshReclaimable()
    }

    /// Bytes a full garbage-collection pass would free: orphaned payloads
    /// (deleted or partially downloaded models) plus incomplete downloads.
    @Published private(set) var reclaimableBytes: Int64 = 0

    func refreshReclaimable() {
        guard Self.isSupported else { return }
        let plan = try? ModelStorageManager().garbageCollectionPlan()
        reclaimableBytes = (plan?.reclaimableBytes ?? 0) + (plan?.incompleteDownloadBytes ?? 0)
    }

    /// Frees everything no installed model references — the way partial
    /// downloads and crash leftovers come back.
    func reclaimSpace() {
        guard Self.isSupported, activity == .idle else { return }
        lastError = nil
        do {
            let storage = try ModelStorageManager()
            let plan = try storage.garbageCollectionPlan()
            _ = try storage.execute(plan)
        } catch {
            lastError = error.localizedDescription
        }
        refresh()
        refreshReclaimable()
    }

    func generateImage(prompt: String, width: Int = 512, height: Int = 512, steps: Int = 4) async {
        guard Self.isSupported else { return }
        let modelID = selectedImageModelID
        guard state(of: modelID) == .ready, activity == .idle else { return }
        if warmChatModelID != nil {
            coolChat()
        }
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
        guard let model = Self.model(withID: selectedChatModelID) else {
            throw RelayAppError("Pick an on-device chat model first.")
        }
        guard let root = ManagedModelResolver.resolveInstalledModel(id: model.id) else {
            throw RelayAppError("Download \(model.title) before chatting on-device.")
        }
        guard activity == .idle else {
            throw RelayAppError("The on-device engine is busy with another run.")
        }
        activity = .chatting
        defer { activity = .idle }

        let sampling = Q35Resources.recommendedSampling(forModelId: model.id)
        let request = ChatRequest(
            messages: messages,
            maxTokens: 1024,
            temperature: sampling?.temperature ?? 0.7,
            topP: sampling?.topP ?? 0.9,
            showThinking: false,
            // The model's native context is 262k; a phone-sized KV budget
            // keeps the 27B's working set inside the app memory ceiling.
            maxContextTokens: 4096
        )

        let accumulated = StreamedText()
        let progressHandler: @Sendable (ChatProgress) -> Void = { progress in
            guard progress.stage == .generating,
                  let piece = progress.message, !piece.isEmpty else { return }
            Task { @MainActor in
                LocalEngine.shared.chatStatus = nil
                let text = accumulated.append(piece)
                let visible = GeneratedTextFilters.strippingThinking(text, streaming: true)
                if !visible.isEmpty { onDelta(visible) }
            }
        }
        if warmChatModelID != model.id {
            chatStatus = "Loading \(model.title)…"
        }
        defer { chatStatus = nil }
        let generator = chatGenerator(for: model)
        let response = try await generator.chat(request, modelPath: root.path, progressHandler: progressHandler)
        warmChatModelID = model.id
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
