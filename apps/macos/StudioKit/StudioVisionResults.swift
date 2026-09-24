import CoreGraphics
import Foundation

// The result documents of the Vision tasks moving onto the Analyze canvas (`vision pose`,
// `vision flow`), decoded here so the overlay and flow renderers in StudioUI draw from typed
// values. `StudioFaceOverlayResult` lives beside them in `StudioSpecialistResults.swift`.

// MARK: - vision pose

/// What `mere.run vision pose --json-output` writes: the picture's stored size, the coordinate
/// space the points are in, and one subject per body, hand, or face with its named landmarks.
package struct StudioPoseOverlayResult: Decodable, Equatable {
    package struct Subject: Decodable, Equatable {
        package struct Point: Decodable, Equatable {
            package let name: String
            package let x: Double
            package let y: Double
            package let confidence: Double
        }

        /// "body", "hand", or "face".
        package let kind: String
        package let index: Int
        package let points: [Point]
    }

    package let imageWidth: Int
    package let imageHeight: Int
    /// "normalized" (origin top-left) or "normalized-bottom-left".
    package let coordinateSpace: String
    package let subjects: [Subject]

    package static func load(from url: URL) -> StudioPoseOverlayResult? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Self.self, from: data)
    }

    /// Where a landmark sits in the picture's stored pixels, origin top-left, whichever space
    /// the document declares.
    package func storedPoint(_ point: Subject.Point) -> CGPoint {
        let normalizedY = coordinateSpace == "normalized-bottom-left" ? 1 - point.y : point.y
        return CGPoint(x: point.x * Double(max(1, imageWidth)), y: normalizedY * Double(max(1, imageHeight)))
    }

    /// "2 subjects · 51 landmarks"
    package var summary: String {
        let subjectCount = subjects.count == 1 ? "1 subject" : "\(subjects.count) subjects"
        let landmarks = subjects.reduce(0) { $0 + $1.points.count }
        return "\(subjectCount) · \(landmarks) landmarks"
    }
}

// MARK: - vision flow

/// What `mere.run vision flow` writes: a Middlebury `.flo` file — the magic float `202021.25`,
/// the width and height as 32-bit integers, then one (dx, dy) float pair per pixel, row-major.
package struct StudioFlowField: Equatable {
    package let width: Int
    package let height: Int
    package let vectors: [SIMD2<Float>]

    package init(width: Int, height: Int, vectors: [SIMD2<Float>]) {
        self.width = width
        self.height = height
        self.vectors = vectors
    }

    package static func load(url: URL) throws -> StudioFlowField {
        try decode(Data(contentsOf: url))
    }

    package static func decode(_ data: Data) throws -> StudioFlowField {
        guard data.count >= 12 else { throw CocoaError(.fileReadCorruptFile) }
        func uint32(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { raw in
                UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            }
        }
        let magic = Float(bitPattern: uint32(0))
        let width = Int(Int32(bitPattern: uint32(4)))
        let height = Int(Int32(bitPattern: uint32(8)))
        guard abs(magic - 202_021.25) < 0.01, width > 0, height > 0,
              data.count >= 12 + width * height * 8 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var vectors: [SIMD2<Float>] = []
        vectors.reserveCapacity(width * height)
        var offset = 12
        for _ in 0..<(width * height) {
            vectors.append(SIMD2(Float(bitPattern: uint32(offset)), Float(bitPattern: uint32(offset + 4))))
            offset += 8
        }
        return StudioFlowField(width: width, height: height, vectors: vectors)
    }

    /// "640×360 dense flow"
    package var summary: String {
        "\(width)×\(height) dense flow"
    }

    /// How much motion the field holds, for the result panel's rows.
    package struct Statistics: Equatable {
        package let meanMagnitude: Double
        package let maximumMagnitude: Double
        /// The share of pixels that moved more than a hundredth of a pixel.
        package let movingFraction: Double
    }

    package var statistics: Statistics {
        var total = 0.0
        var maximum = 0.0
        var moving = 0
        for vector in vectors {
            let magnitude = Double(hypot(vector.x, vector.y))
            guard magnitude.isFinite else { continue }
            total += magnitude
            maximum = max(maximum, magnitude)
            if magnitude > 0.01 { moving += 1 }
        }
        let count = Double(max(1, vectors.count))
        return Statistics(meanMagnitude: total / count, maximumMagnitude: maximum, movingFraction: Double(moving) / count)
    }
}

// MARK: - vision face embed

/// What `mere.run vision face embed --json-output` writes (`FaceEmbeddingOutput`): the picture,
/// the model, and the one face it embedded with its normalized ArcFace vector.
package struct StudioFaceEmbeddingDocument: Decodable, Equatable {
    package let image: String
    package let modelID: String
    package let face: StudioFaceOverlayResult.Record

    /// "Face 2 embedded · 512 dimensions"
    package var summary: String {
        "Face \(face.index + 1) embedded · \(face.embedding?.count ?? 0) dimensions"
    }
}

// MARK: - vision face compare

/// What `mere.run vision face compare --json-output` writes (`FaceComparisonOutput`): which face
/// of each picture was compared and their cosine similarity.
package struct StudioFaceComparisonDocument: Decodable, Equatable {
    package let modelID: String
    package let referenceImage: String
    package let referenceFaceIndex: Int
    package let candidateImage: String
    package let candidateFaceIndex: Int
    package let cosineSimilarity: Double

    /// "Similarity 0.83"
    package var summary: String {
        String(format: "Similarity %.2f", cosineSimilarity)
    }
}

// MARK: - vision face batch

/// What `mere.run vision face batch --jsonl-output` writes: one `FaceBatchOutput` object per
/// line, each naming its picture and carrying the detection result or the error for it.
package struct StudioFaceBatchDocument: Equatable {
    package struct Entry: Decodable, Equatable {
        package let ok: Bool
        package let image: String
        package let result: StudioFaceOverlayResult?
        package let error: String?
    }

    package let entries: [Entry]

    package init(entries: [Entry]) {
        self.entries = entries
    }

    /// Reads the JSONL, or nil when any line is not a batch entry.
    package static func decode(_ data: Data) -> StudioFaceBatchDocument? {
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        let lines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !lines.isEmpty else { return nil }
        let decoder = JSONDecoder()
        var entries: [Entry] = []
        for line in lines {
            guard let entry = try? decoder.decode(Entry.self, from: Data(line.utf8)) else { return nil }
            entries.append(entry)
        }
        return StudioFaceBatchDocument(entries: entries)
    }

    package var faceCount: Int {
        entries.reduce(0) { $0 + ($1.result?.faces.count ?? 0) }
    }

    package var failureCount: Int {
        entries.filter { !$0.ok }.count
    }

    /// "3 images · 5 faces", with " · 1 failed" when a picture could not be read.
    package var summary: String {
        let images = entries.count == 1 ? "1 image" : "\(entries.count) images"
        let faces = faceCount == 1 ? "1 face" : "\(faceCount) faces"
        var parts = [images, faces]
        if failureCount > 0 { parts.append("\(failureCount) failed") }
        return parts.joined(separator: " · ")
    }
}

// MARK: - vision depth

/// What `mere.run vision depth` writes beside the depth EXR and its preview
/// (`MarigoldV2DepthManifest`): the picture's size, the size it was inferred at, the checkpoint,
/// and the decoded range. The result panel reads it; the preview PNG is what the canvas shows.
package struct StudioDepthManifest: Decodable, Equatable {
    package struct Statistics: Decodable, Equatable {
        package let rawMinimum: Float
        package let rawMaximum: Float
    }

    package struct Model: Decodable, Equatable {
        package let modelID: String
    }

    package let width: Int
    package let height: Int
    package let inferenceWidth: Int
    package let inferenceHeight: Int
    package let checkpoint: String
    package let parameterization: String
    package let semantics: String
    package let depthStatistics: Statistics
    package let model: Model

    /// "960 × 720 · inferred at 1024 × 768 · log-stage2"
    package var summary: String {
        "\(width) × \(height) · inferred at \(inferenceWidth) × \(inferenceHeight) · \(checkpoint)"
    }
}

// MARK: - Picking a face by clicking it

/// The `--face-index` family: which picture each flag counts faces in, and the Face detection
/// run whose boxes the picker draws over it.
package enum StudioFacePick {
    /// The positional the flag's face is in: the candidate index names a comparison's second
    /// picture, the rest the first.
    package static func argumentIndex(forFlag flag: String) -> Int? {
        switch flag {
        case "--face-index", "--reference-face-index": return 0
        case "--candidate-face-index": return 1
        default: return nil
        }
    }

    /// The newest finished Face detection document for `imagePath` among the Library's rows, or
    /// nil when nobody has detected faces in that picture yet.
    package static func detectionDocumentURL(for imagePath: String, in items: [StudioLibraryItem]) -> URL? {
        let trimmed = imagePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let input = URL(fileURLWithPath: NSString(string: trimmed).expandingTildeInPath)
        let detection = items
            .filter { $0.templateID == .visionFaceDetect && $0.status == .completed && StudioInputIdentity.matches(item: $0, input: input) }
            .max { $0.createdAt < $1.createdAt }
        return detection.flatMap { item in
            item.allArtifactURLs.first { $0.pathExtension.lowercased() == "json" && FileManager.default.fileExists(atPath: $0.path) }
        }
    }
}

// MARK: - What a directory output holds

/// The files a depth or geometry run wrote into its output directory, sorted into what the
/// canvas can show: preview PNGs, the point clouds Quick Look renders, review clips, and the
/// manifests. Read from the row's artifacts, descending one level into a directory output and
/// into the `frames` and `views` folders the exporters use.
package struct StudioVisionRunArtifacts: Equatable {
    package let previews: [URL]
    package let scenes: [URL]
    package let clips: [URL]
    package let documents: [URL]

    package init(previews: [URL] = [], scenes: [URL] = [], clips: [URL] = [], documents: [URL] = []) {
        self.previews = previews
        self.scenes = scenes
        self.clips = clips
        self.documents = documents
    }

    package var isEmpty: Bool {
        previews.isEmpty && scenes.isEmpty && clips.isEmpty
    }

    package static func read(item: StudioLibraryItem, fileManager: FileManager = .default) -> StudioVisionRunArtifacts {
        var files: [URL] = []
        var seen = Set<String>()
        func add(_ url: URL) {
            guard seen.insert(url.standardizedFileURL.path).inserted else { return }
            files.append(url)
        }
        func descend(_ directory: URL, nested: Bool) {
            let contents = (try? fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
            )) ?? []
            for url in contents.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                    if nested, ["frames", "views"].contains(url.lastPathComponent) { descend(url, nested: false) }
                } else {
                    add(url)
                }
            }
        }
        for url in item.allArtifactURLs {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else { continue }
            if isDirectory.boolValue {
                descend(url, nested: true)
            } else {
                add(url)
            }
        }
        var previews: [URL] = []
        var scenes: [URL] = []
        var clips: [URL] = []
        var documents: [URL] = []
        for url in files {
            switch StudioOutputFileKind.classify(url) {
            case .image where url.pathExtension.lowercased() == "png": previews.append(url)
            case .model3D: scenes.append(url)
            case .video: clips.append(url)
            case .text where url.pathExtension.lowercased() == "json": documents.append(url)
            default: break
            }
        }
        // A GLB carries the colors; the PLY is the same cloud for other tools.
        scenes.sort { lhs, rhs in
            let order = ["glb", "usdz", "obj", "ply"]
            let left = order.firstIndex(of: lhs.pathExtension.lowercased()) ?? order.count
            let right = order.firstIndex(of: rhs.pathExtension.lowercased()) ?? order.count
            return left == right ? lhs.lastPathComponent < rhs.lastPathComponent : left < right
        }
        return StudioVisionRunArtifacts(previews: previews, scenes: scenes, clips: clips, documents: documents)
    }
}
