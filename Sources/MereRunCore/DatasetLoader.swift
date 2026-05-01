import Foundation

public enum DatasetLoader {
    public struct ImageCaptionPair: Hashable, Sendable {
        public let imageURL: URL
        public let caption: String

        public init(imageURL: URL, caption: String) {
            self.imageURL = imageURL
            self.caption = caption
        }
    }

    public struct EditImagePair: Hashable, Sendable {
        public let inputImageURL: URL
        public let outputImageURL: URL
        public let caption: String

        public init(inputImageURL: URL, outputImageURL: URL, caption: String) {
            self.inputImageURL = inputImageURL
            self.outputImageURL = outputImageURL
            self.caption = caption
        }
    }

    public enum TrainingDataset: Hashable, Sendable {
        case textToImage([ImageCaptionPair])
        case edit([EditImagePair])
    }

    public static func loadTrainingDataset(
        from directory: URL,
        excludePreviewImages: Bool = false,
        fileManager: FileManager = .default
    ) throws -> TrainingDataset {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            throw DatasetLoaderError.datasetDirectoryNotFound(directory)
        }

        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)

        let imageExts: Set<String> = ["png", "jpg", "jpeg", "webp"]
        let images = contents
            .filter { imageExts.contains($0.pathExtension.lowercased()) }
            .filter { entry in
                guard excludePreviewImages else { return true }
                return !entry.deletingPathExtension().lastPathComponent.hasPrefix("preview")
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        let outputImages = images.filter { $0.deletingPathExtension().lastPathComponent.hasSuffix("_out") }
        if !outputImages.isEmpty {
            let outputBases = Set(outputImages.map { outputImage in
                String(outputImage.deletingPathExtension().lastPathComponent.dropLast(4))
            })

            for image in images {
                let stem = image.deletingPathExtension().lastPathComponent
                if stem.hasSuffix("_out") {
                    continue
                }
                if stem.hasSuffix("_in") {
                    let base = String(stem.dropLast(3))
                    guard outputBases.contains(base) else {
                        throw DatasetLoaderError.strayEditInputImage(image)
                    }
                    continue
                }
                throw DatasetLoaderError.mixedDatasetModes(directory)
            }

            let imagesByStem = Dictionary(uniqueKeysWithValues: images.map { ($0.deletingPathExtension().lastPathComponent, $0) })
            var pairs: [EditImagePair] = []
            pairs.reserveCapacity(outputImages.count)

            for outputImage in outputImages {
                let base = String(outputImage.deletingPathExtension().lastPathComponent.dropLast(4))
                let inputStem = "\(base)_in"
                guard let inputImage = imagesByStem[inputStem] else {
                    throw DatasetLoaderError.missingEditInputImage(outputImage)
                }
                let captionURL = directory.appendingPathComponent("\(inputStem).txt")
                let caption = try loadCaption(from: captionURL, fileManager: fileManager)
                pairs.append(EditImagePair(inputImageURL: inputImage, outputImageURL: outputImage, caption: caption))
            }

            return .edit(pairs)
        }

        var pairs: [ImageCaptionPair] = []
        pairs.reserveCapacity(images.count)
        for image in images {
            let captionURL = image.deletingPathExtension().appendingPathExtension("txt")
            let caption = try loadCaption(from: captionURL, fileManager: fileManager)
            pairs.append(ImageCaptionPair(imageURL: image, caption: caption))
        }
        return .textToImage(pairs)
    }

    public static func loadImageCaptionPairs(
        from directory: URL,
        excludePreviewImages: Bool = false,
        fileManager: FileManager = .default
    ) throws -> [(imageURL: URL, caption: String)] {
        let dataset = try loadTrainingDataset(
            from: directory,
            excludePreviewImages: excludePreviewImages,
            fileManager: fileManager
        )
        switch dataset {
        case .textToImage(let pairs):
            return pairs.map { (imageURL: $0.imageURL, caption: $0.caption) }
        case .edit:
            throw DatasetLoaderError.editDatasetNotSupportedByTextToImageAPI(directory)
        }
    }

    private static func loadCaption(from captionURL: URL, fileManager: FileManager) throws -> String {
        guard fileManager.fileExists(atPath: captionURL.path) else {
            throw DatasetLoaderError.missingCaptionFile(captionURL)
        }
        let captionData = try Data(contentsOf: captionURL)
        let caption = String(decoding: captionData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !caption.isEmpty else {
            throw DatasetLoaderError.emptyCaptionFile(captionURL)
        }
        return caption
    }
}

public enum DatasetLoaderError: Error, LocalizedError, Sendable {
    case datasetDirectoryNotFound(URL)
    case missingCaptionFile(URL)
    case emptyCaptionFile(URL)
    case mixedDatasetModes(URL)
    case strayEditInputImage(URL)
    case missingEditInputImage(URL)
    case editDatasetNotSupportedByTextToImageAPI(URL)

    public var errorDescription: String? {
        switch self {
        case .datasetDirectoryNotFound(let url):
            return "Dataset directory not found: \(url.path)"
        case .missingCaptionFile(let url):
            return "Missing caption file: \(url.path)"
        case .emptyCaptionFile(let url):
            return "Empty caption file: \(url.path)"
        case .mixedDatasetModes(let url):
            return "Data folder mixes edit-style and txt2img images: \(url.path). Use only *_out/*_in pairs or only standard image+caption pairs."
        case .strayEditInputImage(let url):
            return "Found input image without matching output: \(url.lastPathComponent). Remove the stray *_in.* or add the corresponding *_out.*."
        case .missingEditInputImage(let url):
            let stem = url.deletingPathExtension().lastPathComponent
            let base = String(stem.dropLast(4))
            return "Missing input image for '\(url.lastPathComponent)'. Expected '\(base)_in.*' in \(url.deletingLastPathComponent().path)."
        case .editDatasetNotSupportedByTextToImageAPI(let url):
            return "Edit-style dataset detected in \(url.path). This command currently supports only image + .txt caption pairs."
        }
    }
}
