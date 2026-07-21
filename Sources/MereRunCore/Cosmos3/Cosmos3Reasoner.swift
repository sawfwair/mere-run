import Foundation
@preconcurrency import Hub
import MediaIO
import MLX
import MLXRandom
@preconcurrency import Tokenizers

public enum Cosmos3ReasonerError: LocalizedError, Sendable {
    case missingModelFiles([URL])
    case conflictingMedia
    case missingMedia(URL)
    case emptyVideo(URL)
    case invalidTokenLayout(expected: Int, actual: Int)
    case invalidGenerationLimit(Int)

    public var errorDescription: String? {
        switch self {
        case .missingModelFiles(let urls):
            return "Cosmos3 reasoner resources are incomplete: "
                + urls.map(\.path).joined(separator: ", ")
        case .conflictingMedia:
            return "Cosmos3 reasoner accepts one image or one video per request."
        case .missingMedia(let url):
            return "Cosmos3 reasoner media does not exist: \(url.path)"
        case .emptyVideo(let url):
            return "Cosmos3 reasoner video has no decodable frames: \(url.path)"
        case .invalidTokenLayout(let expected, let actual):
            return "Cosmos3 reasoner prompt has \(actual) visual placeholders for \(expected) visual embeddings."
        case .invalidGenerationLimit(let value):
            return "Cosmos3 reasoner max new tokens must be positive; received \(value)."
        }
    }
}

public struct Cosmos3ReasonerRequest: Hashable, Sendable {
    public let prompt: String
    public let imageURL: URL?
    public let videoURL: URL?
    public let maxNewTokens: Int
    public let temperature: Float
    public let topP: Float
    public let seed: UInt64
    public let maximumVideoFrames: Int

    public init(
        prompt: String,
        imageURL: URL? = nil,
        videoURL: URL? = nil,
        maxNewTokens: Int = 256,
        temperature: Float = 0.2,
        topP: Float = 0.9,
        seed: UInt64 = 0,
        maximumVideoFrames: Int = 32
    ) throws {
        guard maxNewTokens > 0 else {
            throw Cosmos3ReasonerError.invalidGenerationLimit(maxNewTokens)
        }
        guard imageURL == nil || videoURL == nil else {
            throw Cosmos3ReasonerError.conflictingMedia
        }
        self.prompt = prompt
        self.imageURL = imageURL
        self.videoURL = videoURL
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topP = topP
        self.seed = seed
        self.maximumVideoFrames = max(1, maximumVideoFrames)
    }
}

public struct Cosmos3ReasonerResult: Hashable, Sendable {
    public let text: String
    public let generatedTokenIDs: [Int]
    public let promptTokenCount: Int
    public let visualTokenCount: Int

    public init(
        text: String,
        generatedTokenIDs: [Int],
        promptTokenCount: Int,
        visualTokenCount: Int
    ) {
        self.text = text
        self.generatedTokenIDs = generatedTokenIDs
        self.promptTokenCount = promptTokenCount
        self.visualTokenCount = visualTokenCount
    }
}

public struct Cosmos3ReasonerProcessedVision {
    public let patches: MLXArray
    public let grids: [Cosmos3ReasonerVisionGrid]
    public let tokenID: Int

    public init(
        patches: MLXArray,
        grids: [Cosmos3ReasonerVisionGrid],
        tokenID: Int
    ) {
        self.patches = patches
        self.grids = grids
        self.tokenID = tokenID
    }
}

public enum Cosmos3ReasonerProcessor {
    public static func targetSize(
        width: Int,
        height: Int,
        factor: Int = 32,
        minimumPixels: Int,
        maximumPixels: Int
    ) -> (width: Int, height: Int) {
        precondition(width > 0 && height > 0 && factor > 0)
        func rounded(_ value: Double) -> Int {
            Int((value / Double(factor)).rounded()) * factor
        }
        func floored(_ value: Double) -> Int {
            Int(floor(value / Double(factor))) * factor
        }
        func ceiled(_ value: Double) -> Int {
            Int(ceil(value / Double(factor))) * factor
        }
        var targetHeight = max(factor, rounded(Double(height)))
        var targetWidth = max(factor, rounded(Double(width)))
        if targetHeight * targetWidth > maximumPixels {
            let beta = sqrt(Double(width * height) / Double(maximumPixels))
            targetHeight = max(factor, floored(Double(height) / beta))
            targetWidth = max(factor, floored(Double(width) / beta))
        } else if targetHeight * targetWidth < minimumPixels {
            let beta = sqrt(Double(minimumPixels) / Double(width * height))
            targetHeight = max(factor, ceiled(Double(height) * beta))
            targetWidth = max(factor, ceiled(Double(width) * beta))
        }
        return (targetWidth, targetHeight)
    }

    public static func image(
        _ image: MediaImage,
        configuration: Cosmos3ReasonerConfiguration
    ) throws -> Cosmos3ReasonerProcessedVision {
        let factor = configuration.vision.patchSize
            * configuration.vision.spatialMergeSize
        let size = targetSize(
            width: image.width,
            height: image.height,
            factor: factor,
            minimumPixels: 65_536,
            maximumPixels: 16_777_216
        )
        let resized = try MediaImageIO.resized(
            image,
            width: size.width,
            height: size.height
        )
        let patches = patchify(
            resized,
            patchSize: configuration.vision.patchSize
        )
        return Cosmos3ReasonerProcessedVision(
            patches: patches,
            grids: [Cosmos3ReasonerVisionGrid(
                time: 1,
                height: size.height / configuration.vision.patchSize,
                width: size.width / configuration.vision.patchSize
            )],
            tokenID: configuration.imageTokenID
        )
    }

    public static func video(
        _ frames: [MediaImage],
        configuration: Cosmos3ReasonerConfiguration
    ) throws -> Cosmos3ReasonerProcessedVision {
        precondition(!frames.isEmpty)
        let factor = configuration.vision.patchSize
            * configuration.vision.spatialMergeSize
        let perFrameMaximum = max(4_096, 25_165_824 / frames.count)
        let size = targetSize(
            width: frames[0].width,
            height: frames[0].height,
            factor: factor,
            minimumPixels: 4_096,
            maximumPixels: perFrameMaximum
        )
        let patchArrays = try frames.map {
            patchify(
                try MediaImageIO.resized(
                    $0,
                    width: size.width,
                    height: size.height
                ),
                patchSize: configuration.vision.patchSize
            )
        }
        return Cosmos3ReasonerProcessedVision(
            patches: MLX.concatenated(patchArrays, axis: 0),
            grids: [Cosmos3ReasonerVisionGrid(
                time: frames.count,
                height: size.height / configuration.vision.patchSize,
                width: size.width / configuration.vision.patchSize
            )],
            tokenID: configuration.videoTokenID
        )
    }

    public static func positionIDs(
        tokenIDs: [Int],
        imageGrids: [Cosmos3ReasonerVisionGrid],
        videoGrids: [Cosmos3ReasonerVisionGrid],
        configuration: Cosmos3ReasonerConfiguration
    ) -> MLXArray {
        var expandedVideoGrids: [Cosmos3ReasonerVisionGrid] = []
        for grid in videoGrids {
            expandedVideoGrids.append(contentsOf: (0..<grid.time).map { _ in
                Cosmos3ReasonerVisionGrid(
                    time: 1,
                    height: grid.height,
                    width: grid.width
                )
            })
        }
        var positions: [[Float]] = [[], [], []]
        var start = 0
        var imageIndex = 0
        var videoIndex = 0

        while start < tokenIDs.count {
            let nextImage = tokenIDs[start...].firstIndex(of: configuration.imageTokenID)
            let nextVideo = tokenIDs[start...].firstIndex(of: configuration.videoTokenID)
            let tokenIndex: Int
            let grid: Cosmos3ReasonerVisionGrid
            if let nextImage, nextVideo == nil || nextImage < nextVideo!,
               imageIndex < imageGrids.count {
                tokenIndex = nextImage
                grid = imageGrids[imageIndex]
                imageIndex += 1
            } else if let nextVideo, videoIndex < expandedVideoGrids.count {
                tokenIndex = nextVideo
                grid = expandedVideoGrids[videoIndex]
                videoIndex += 1
            } else {
                appendTextPositions(
                    count: tokenIDs.count - start,
                    to: &positions
                )
                break
            }

            appendTextPositions(count: tokenIndex - start, to: &positions)
            let base = (positions.flatMap { $0 }.max() ?? -1) + 1
            let mergedHeight = grid.height / configuration.projector.spatialMergeSize
            let mergedWidth = grid.width / configuration.projector.spatialMergeSize
            for time in 0..<grid.time {
                for row in 0..<mergedHeight {
                    for column in 0..<mergedWidth {
                        positions[0].append(base + Float(time))
                        positions[1].append(base + Float(row))
                        positions[2].append(base + Float(column))
                    }
                }
            }
            start = tokenIndex + grid.time * mergedHeight * mergedWidth
        }
        return MLX.stacked(positions.map { MLXArray($0) }, axis: 0)
    }

    private static func appendTextPositions(
        count: Int,
        to positions: inout [[Float]]
    ) {
        guard count > 0 else { return }
        let base = (positions.flatMap { $0 }.max() ?? -1) + 1
        let values = (0..<count).map { base + Float($0) }
        for axis in positions.indices {
            positions[axis].append(contentsOf: values)
        }
    }

    private static func patchify(
        _ image: MediaImage,
        patchSize: Int
    ) -> MLXArray {
        let chw = MediaImageIO.rgbCHWFloat(
            image,
            normalizedToMinusOneToOne: true
        )
        let gridHeight = image.height / patchSize
        let gridWidth = image.width / patchSize
        var values: [Float] = []
        values.reserveCapacity(
            gridHeight * gridWidth * patchSize * patchSize * 3
        )
        for patchRow in 0..<gridHeight {
            for patchColumn in 0..<gridWidth {
                for row in 0..<patchSize {
                    for column in 0..<patchSize {
                        let pixel = (patchRow * patchSize + row) * image.width
                            + patchColumn * patchSize + column
                        for channel in 0..<3 {
                            values.append(chw[channel * image.width * image.height + pixel])
                        }
                    }
                }
            }
        }
        return MLXArray(values).reshaped(
            gridHeight * gridWidth,
            patchSize * patchSize * 3
        )
    }
}

public final class Cosmos3Reasoner: @unchecked Sendable {
    private var loadedRootURL: URL?
    private var tokenizer: (any Tokenizer)?
    private var transformer: Cosmos3OmniTransformerModel?
    private var vision: Cosmos3ReasonerVisionModel?
    private var configuration: Cosmos3ReasonerConfiguration?

    public init() {}

    public var isWarm: Bool {
        tokenizer != nil && transformer != nil && vision != nil
    }

    public func prepare(
        resources: Cosmos3Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws {
        if loadedRootURL != nil, loadedRootURL != resources.rootURL {
            unload()
        }
        let missing = resources.validate() + resources.validateReasoner()
        guard missing.isEmpty else {
            throw Cosmos3ReasonerError.missingModelFiles(missing)
        }
        loadedRootURL = resources.rootURL
        if configuration == nil {
            configuration = try resources.loadReasonerConfiguration()
        }
        if tokenizer == nil {
            progress?("Loading Cosmos3-Edge reasoner tokenizer")
            tokenizer = try await AutoTokenizer.from(
                modelFolder: resources.rootURL,
                hubApi: .shared
            )
        }
        if transformer == nil {
            transformer = try Cosmos3ModelLoader.loadTransformer(
                resources: resources,
                progress: progress
            )
        }
        if vision == nil {
            vision = try Cosmos3ModelLoader.loadReasonerVision(
                resources: resources,
                progress: progress
            )
        }
    }

    public func unload() {
        tokenizer = nil
        transformer = nil
        vision = nil
        configuration = nil
        loadedRootURL = nil
        Memory.clearCache()
    }

    public func generate(
        request: Cosmos3ReasonerRequest,
        resources: Cosmos3Resources,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> Cosmos3ReasonerResult {
        try await prepare(resources: resources, progress: progress)
        MLXRandom.seed(request.seed)
        let prepared = try prepareInput(request: request)
        var tokenIDs = prepared.tokenIDs
        var generated: [Int] = []
        generated.reserveCapacity(request.maxNewTokens)
        let eos = tokenizer!.eosTokenId ?? 11

        for index in 0..<request.maxNewTokens {
            try Task.checkCancellation()
            progress?("Reasoning token \(index + 1)/\(request.maxNewTokens)")
            let ids = MLXArray(tokenIDs.map(Int32.init))
            var embeddings = transformer!.embedText(tokenIDs: ids)
            if let visualEmbeddings = prepared.visualEmbeddings {
                let indexes = tokenIDs.enumerated().compactMap {
                    $0.element == prepared.visualTokenID ? $0.offset : nil
                }
                guard indexes.count == visualEmbeddings.dim(0) else {
                    throw Cosmos3ReasonerError.invalidTokenLayout(
                        expected: visualEmbeddings.dim(0),
                        actual: indexes.count
                    )
                }
                let indexArray = MLXArray(indexes)
                embeddings = embeddings.at[indexArray].add(
                    visualEmbeddings - embeddings[indexArray]
                )
            }
            let positions = Cosmos3ReasonerProcessor.positionIDs(
                tokenIDs: tokenIDs,
                imageGrids: prepared.imageGrids,
                videoGrids: prepared.videoGrids,
                configuration: configuration!
            )
            let hidden = transformer!.reasonerHidden(
                inputEmbeddings: embeddings,
                positionIDs: positions
            )
            let logits = transformer!.languageModelHead(hidden[hidden.dim(0) - 1])
            let next: Int
            if request.temperature <= 0 {
                next = Int(MLX.argMax(logits).item(Int32.self))
            } else {
                next = topPSample(
                    logits: logits,
                    temperature: request.temperature,
                    topP: request.topP
                )
            }
            tokenIDs.append(next)
            generated.append(next)
            if next == eos { break }
            Memory.clearCache()
        }
        return Cosmos3ReasonerResult(
            text: tokenizer!.decode(tokens: generated, skipSpecialTokens: true)
                .trimmingCharacters(in: .whitespacesAndNewlines),
            generatedTokenIDs: generated,
            promptTokenCount: prepared.promptTokenCount,
            visualTokenCount: prepared.visualEmbeddings?.dim(0) ?? 0
        )
    }

    private func prepareInput(
        request: Cosmos3ReasonerRequest
    ) throws -> (
        tokenIDs: [Int],
        promptTokenCount: Int,
        visualEmbeddings: MLXArray?,
        visualTokenID: Int,
        imageGrids: [Cosmos3ReasonerVisionGrid],
        videoGrids: [Cosmos3ReasonerVisionGrid]
    ) {
        var processed: Cosmos3ReasonerProcessedVision?
        var imageGrids: [Cosmos3ReasonerVisionGrid] = []
        var videoGrids: [Cosmos3ReasonerVisionGrid] = []
        var content = request.prompt
        if let imageURL = request.imageURL {
            try requireMedia(imageURL)
            let value = try Cosmos3ReasonerProcessor.image(
                MediaImageIO.decode(imageURL),
                configuration: configuration!
            )
            processed = value
            imageGrids = value.grids
            let count = value.grids.reduce(0) {
                $0 + $1.patchCount
                    / (
                        configuration!.projector.spatialMergeSize
                            * configuration!.projector.spatialMergeSize
                    )
            }
            content = "<|vision_start|>"
                + String(repeating: "<|image_pad|>", count: count)
                + "<|vision_end|>\n"
                + request.prompt
        } else if let videoURL = request.videoURL {
            try requireMedia(videoURL)
            let decoded = try decodeVideo(
                videoURL,
                maximumFrames: request.maximumVideoFrames
            )
            let value = try Cosmos3ReasonerProcessor.video(
                decoded.frames,
                configuration: configuration!
            )
            processed = value
            videoGrids = value.grids
            let perFrame = value.grids[0].height * value.grids[0].width
                / (
                    configuration!.projector.spatialMergeSize
                        * configuration!.projector.spatialMergeSize
                )
            var placeholder = ""
            for frame in decoded.frames.indices {
                placeholder += String(
                    format: "<%.1f seconds>",
                    Double(frame) / decoded.fps
                )
                placeholder += "<|vision_start|>"
                    + String(repeating: "<|video_pad|>", count: perFrame)
                    + "<|vision_end|>"
            }
            content = placeholder + "\n" + request.prompt
        }
        let ids = try tokenizer!.applyChatTemplate(
            messages: [["role": "user", "content": content]],
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil
        )
        let visualEmbeddings: MLXArray?
        if let processed {
            let encoded = vision!(
                patches: processed.patches.asType(.bfloat16),
                grids: processed.grids
            ).embeddings
            eval(encoded)
            visualEmbeddings = encoded
        } else {
            visualEmbeddings = nil
        }
        return (
            ids,
            ids.count,
            visualEmbeddings,
            processed?.tokenID ?? configuration!.imageTokenID,
            imageGrids,
            videoGrids
        )
    }

    private func requireMedia(_ url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Cosmos3ReasonerError.missingMedia(url)
        }
    }

    private func decodeVideo(
        _ url: URL,
        maximumFrames: Int
    ) throws -> (frames: [MediaImage], fps: Double) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-cosmos3-reasoner-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let sequence = try MediaVideoIO.extractFrames(
            from: url,
            into: directory,
            endFrame: maximumFrames - 1
        )
        guard !sequence.frameURLs.isEmpty else {
            throw Cosmos3ReasonerError.emptyVideo(url)
        }
        return (
            try sequence.frameURLs.prefix(maximumFrames).map(MediaImageIO.decode),
            max(sequence.fps, 1)
        )
    }
}
