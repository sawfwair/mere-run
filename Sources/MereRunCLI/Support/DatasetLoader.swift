import ArgumentParser
import Foundation
import MereRunCore

enum DatasetLoader {
    static func loadTrainingDataset(
        from directory: URL,
        excludePreviewImages: Bool = false
    ) throws -> MereRunCore.DatasetLoader.TrainingDataset {
        do {
            return try MereRunCore.DatasetLoader.loadTrainingDataset(
                from: directory,
                excludePreviewImages: excludePreviewImages
            )
        } catch let error as DatasetLoaderError {
            throw ValidationError(error.localizedDescription)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    static func loadImageCaptionPairs(
        from directory: URL,
        excludePreviewImages: Bool = false
    ) throws -> [(imageURL: URL, caption: String)] {
        do {
            return try MereRunCore.DatasetLoader.loadImageCaptionPairs(
                from: directory,
                excludePreviewImages: excludePreviewImages
            )
        } catch let error as DatasetLoaderError {
            throw ValidationError(error.localizedDescription)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }
}
