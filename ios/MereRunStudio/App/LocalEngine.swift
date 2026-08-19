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
    static let allowsCellularDownloadsKey = "models.allowCellularDownloads"
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

    @Published private(set) var states: [String: ModelState] = [:]
    @Published private(set) var activity: RuntimeResidencyState.Activity = .idle
    @Published private(set) var lastImage: UIImage?
    @Published private(set) var lastImageURL: URL?
    @Published var selectedImageModelID = LocalEngine.imageModels[0].id {
        didSet {
            if oldValue != selectedImageModelID {
                releaseRuntime()
            }
        }
    }
    @Published var selectedChatModelID: String {
        didSet {
            UserDefaults.standard.set(selectedChatModelID, forKey: "local.selectedChatModelID")
            if oldValue != selectedChatModelID {
                releaseRuntime()
            }
        }
    }
    /// Which chat model currently has its weights resident, if any.
    @Published private(set) var warmChatModelID: String?
    /// Human status for the pending bubble ("Loading Liquid Chat…").
    @Published private(set) var chatStatus: String?
    #if DEBUG
    /// Streaming pipeline telemetry for on-device diagnostics.
    @Published var debugStreamInfo = ""
    #endif
    @Published private(set) var lastError: String?

    private var imageGenerator: Flux2KleinGeneratoriOS?
    private var q35Generators: [String: Q35Generator] = [:]
    private var lfm2Generators: [String: LFM2Generator] = [:]
    private let pendingDownloadStore = PendingModelDownloadStore()
    private var activeDownloadIDs: Set<String> = []
    private var warmTask: Task<Void, Never>?
    private var warmRequestID: UUID?
    private var idleEvictionTask: Task<Void, Never>?
    private var residency = RuntimeResidencyState()

    private static let idleResidencySeconds: Duration = .seconds(120)

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
            // Prefix KV cache: each turn re-sends the whole transcript, and
            // without it the entire history prefills again — the spinner the
            // user stares at. With it, only the newest turn prefills.
            let generator = lfm2Generators[model.id]
                ?? LFM2Generator(modelId: model.id, prefixKVCacheEnabled: true)
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
        guard Self.isSupported, activity == .idle,
              warmTask == nil,
              let model = Self.model(withID: selectedChatModelID),
              warmChatModelID != model.id,
              let root = ManagedModelResolver.resolveInstalledModel(id: model.id) else { return }
        idleEvictionTask?.cancel()
        let requestID = UUID()
        warmRequestID = requestID
        let generator = chatGenerator(for: model)
        chatStatus = "Loading \(model.title)…"
        warmTask = Task { [weak self] in
            do {
                try await generator.prepare(modelPath: root.path, progressHandler: nil)
                guard let self else { return }
                if !Task.isCancelled,
                   self.warmRequestID == requestID,
                   self.selectedChatModelID == model.id,
                   self.activity == .idle {
                    self.warmChatModelID = model.id
                    self.chatStatus = nil
                    self.warmTask = nil
                    self.scheduleIdleEviction()
                } else if self.selectedChatModelID != model.id
                            || self.activity != .chatting {
                    await generator.unload()
                    self.removeChatGenerator(modelID: model.id)
                }
            } catch {
                guard let self, self.warmRequestID == requestID else { return }
                self.chatStatus = nil
                self.warmTask = nil
            }
        }
    }

    /// Frees resident chat weights (an image generation is about to need the
    /// memory, or the user left the on-device lane).
    func coolChat() {
        releaseRuntime()
    }

    /// Drops every resident model after backgrounding, memory pressure, model
    /// changes, or the idle timeout. Active inference is allowed to finish and
    /// then releases before another request can begin.
    func releaseRuntime() {
        idleEvictionTask?.cancel()
        warmTask?.cancel()
        guard residency.requestRelease() else { return }
        activity = residency.activity
        performRuntimeRelease()
    }

    private func performRuntimeRelease() {
        Task { [weak self] in
            guard let self else { return }
            await self.unloadRuntimeNow()
            self.residency.completeRelease()
            self.activity = self.residency.activity
        }
    }

    private func unloadRuntimeNow() async {
        idleEvictionTask?.cancel()
        idleEvictionTask = nil
        warmTask?.cancel()
        warmTask = nil
        warmRequestID = nil
        chatStatus = nil
        warmChatModelID = nil

        let chatGenerators = lfm2Generators.values.map(AnyChatGenerator.lfm2)
            + q35Generators.values.map(AnyChatGenerator.q35)
        lfm2Generators.removeAll()
        q35Generators.removeAll()
        let residentImageGenerator = imageGenerator
        imageGenerator = nil

        for generator in chatGenerators {
            await generator.unload()
        }
        await residentImageGenerator?.unload()
    }

    private func unloadChatRuntime() async {
        warmTask?.cancel()
        warmTask = nil
        warmRequestID = nil
        chatStatus = nil
        warmChatModelID = nil
        let generators = lfm2Generators.values.map(AnyChatGenerator.lfm2)
            + q35Generators.values.map(AnyChatGenerator.q35)
        lfm2Generators.removeAll()
        q35Generators.removeAll()
        for generator in generators {
            await generator.unload()
        }
    }

    private func unloadImageRuntime() async {
        let generator = imageGenerator
        imageGenerator = nil
        await generator?.unload()
    }

    private func removeChatGenerator(modelID: String) {
        lfm2Generators[modelID] = nil
        q35Generators[modelID] = nil
        if warmChatModelID == modelID {
            warmChatModelID = nil
        }
    }

    private func finishActivity() {
        if residency.completeActivity() {
            activity = residency.activity
            performRuntimeRelease()
        } else {
            activity = residency.activity
            scheduleIdleEviction()
        }
    }

    private func scheduleIdleEviction() {
        idleEvictionTask?.cancel()
        idleEvictionTask = Task { [weak self] in
            do {
                try await Task.sleep(for: Self.idleResidencySeconds)
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            self?.releaseRuntime()
        }
    }

    init() {
        let saved = UserDefaults.standard.string(forKey: "local.selectedChatModelID")
        selectedChatModelID = Self.chatModels.first { $0.id == saved }?.id ?? Self.chatModels[0].id
        refresh()
        guard Self.isSupported else { return }
        HubBackgroundTransferSession.shared.reconnect()
        resumePendingDownloads()
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

    var isDownloadingModel: Bool {
        states.values.contains { state in
            if case .downloading = state { return true }
            return false
        }
    }

    func download(_ modelID: String, acceptedTerms: Bool = false) async {
        await download(
            PendingModelDownload(
                modelID: modelID,
                usageTermsAcknowledged: acceptedTerms,
                requestedAt: Date(),
                allowsCellular: UserDefaults.standard.bool(forKey: Self.allowsCellularDownloadsKey)
            ),
            persistRequest: true
        )
    }

    func resumePendingDownloads() {
        guard Self.isSupported else { return }
        HubBackgroundTransferSession.shared.reconnect()
        for pending in pendingDownloadStore.all() {
            if ManagedModelResolver.resolveInstalledModel(id: pending.modelID) != nil {
                try? pendingDownloadStore.remove(modelID: pending.modelID)
                continue
            }
            Task { await self.download(pending, persistRequest: false) }
        }
    }

    private func download(
        _ pending: PendingModelDownload,
        persistRequest: Bool
    ) async {
        let modelID = pending.modelID
        guard Self.isSupported,
              Self.model(withID: modelID) != nil,
              !activeDownloadIDs.contains(modelID) else { return }
        activeDownloadIDs.insert(modelID)
        defer { activeDownloadIDs.remove(modelID) }
        states[modelID] = .downloading("Starting…")
        lastError = nil
        do {
            if persistRequest {
                try pendingDownloadStore.save(pending)
            }
            _ = try await ManagedModelResolver.installManagedModel(
                id: modelID,
                usageTermsAcknowledged: pending.usageTermsAcknowledged,
                useBackgroundSession: true,
                backgroundNetworkPolicy: pending.allowsCellular ? .allNetworks : .wifiOnly,
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
            try pendingDownloadStore.remove(modelID: modelID)
            states[modelID] = .ready
            refreshReclaimable()
        } catch {
            try? pendingDownloadStore.remove(modelID: modelID)
            states[modelID] = .notInstalled
            lastError = error.localizedDescription
        }
    }

    /// Removes an installed model and reclaims its unshared hub-cache
    /// payloads, mirroring `mere.run model remove`. Blobs shared with another
    /// installed model are preserved.
    func delete(_ modelID: String) async {
        guard Self.isSupported,
              let installURL = ManagedModelResolver.resolveInstalledModel(id: modelID),
              residency.beginRelease() else { return }
        activity = residency.activity
        await unloadRuntimeNow()
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
        refresh()
        refreshReclaimable()
        residency.completeRelease()
        activity = residency.activity
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
        guard state(of: modelID) == .ready, residency.begin(.generatingImage) else { return }
        activity = residency.activity
        idleEvictionTask?.cancel()
        lastError = nil
        defer { finishActivity() }
        await unloadChatRuntime()
        let generator = imageGenerator ?? Flux2KleinGeneratoriOS()
        imageGenerator = generator
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
            let result = try await generator.generate(request, progressHandler: nil)
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
    ) async throws -> (reply: String, thinking: String?) {
        guard Self.isSupported else {
            throw RelayAppError("On-device chat needs a physical iPhone.")
        }
        guard let model = Self.model(withID: selectedChatModelID) else {
            throw RelayAppError("Pick an on-device chat model first.")
        }
        guard let root = ManagedModelResolver.resolveInstalledModel(id: model.id) else {
            throw RelayAppError("Download \(model.title) before chatting on-device.")
        }
        guard residency.begin(.chatting) else {
            throw RelayAppError("The on-device engine is busy with another run.")
        }
        activity = residency.activity
        idleEvictionTask?.cancel()
        warmTask?.cancel()
        warmTask = nil
        warmRequestID = nil
        defer { finishActivity() }
        await unloadImageRuntime()

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
            #if DEBUG
            Task { @MainActor in
                LocalEngine.shared.debugRecord(stage: progress.stage, piece: progress.message)
            }
            #endif
            guard progress.stage == .generating,
                  let piece = progress.message, !piece.isEmpty else { return }
            Task { @MainActor in
                LocalEngine.shared.chatStatus = nil
                let text = accumulated.append(piece)
                let visible = GeneratedTextFilters.strippingThinking(text, streaming: true)
                onDelta(visible.isEmpty ? text : visible)
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
        // The stream is reasoning followed by the answer, with no markers;
        // whatever precedes the clean answer in the raw text is the thinking.
        let raw = accumulated.current
        var thinking: String?
        if let answerRange = raw.range(of: cleaned) {
            let head = String(raw[..<answerRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            thinking = head.isEmpty ? nil : head
        } else if raw != cleaned {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            thinking = trimmed.isEmpty || trimmed == cleaned ? nil : trimmed
        }
        return (cleaned, thinking)
    }
}

#if DEBUG
extension LocalEngine {
    private static var debugRaw = 0
    private static var debugGenerating = 0
    private static var debugFirstPiece = ""

    func debugRecord(stage: ChatStage, piece: String?) {
        Self.debugRaw += 1
        if stage == .generating, let piece, !piece.isEmpty {
            Self.debugGenerating += 1
            if Self.debugFirstPiece.count < 60 {
                Self.debugFirstPiece += piece
            }
        }
        debugStreamInfo = "raw \(Self.debugRaw) gen \(Self.debugGenerating) first: \(Self.debugFirstPiece.prefix(60))"
    }
}
#endif

/// Accumulates streamed pieces on the main actor.
@MainActor
private final class StreamedText {
    private var text = ""

    func append(_ piece: String) -> String {
        text += piece
        return text
    }

    var current: String { text }
}

struct RelayAppError: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
