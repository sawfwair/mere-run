import Foundation
import XCTest
@testable import MereRunCore

final class DatasetLoaderTests: MereRunCoreTestCase {

    func testDiscoversTextToImageDataset() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        try writeImageAndCaption(in: temp, stem: "001", imageExt: "png", caption: "first prompt")
        try writeImageAndCaption(in: temp, stem: "002", imageExt: "jpg", caption: "second prompt")

        let dataset = try DatasetLoader.loadTrainingDataset(from: temp)
        guard case .textToImage(let pairs) = dataset else {
            XCTFail("Expected text-to-image dataset")
            return
        }

        XCTAssertEqual(pairs.count, 2)
        XCTAssertEqual(pairs[0].imageURL.lastPathComponent, "001.png")
        XCTAssertEqual(pairs[0].caption, "first prompt")
        XCTAssertEqual(pairs[1].imageURL.lastPathComponent, "002.jpg")
        XCTAssertEqual(pairs[1].caption, "second prompt")
    }

    func testDiscoversEditDataset() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        try writeFile(temp.appendingPathComponent("cat_in.png"))
        try writeFile(temp.appendingPathComponent("cat_out.png"))
        try writeText(temp.appendingPathComponent("cat_in.txt"), "edit cat")

        try writeFile(temp.appendingPathComponent("dog_in.webp"))
        try writeFile(temp.appendingPathComponent("dog_out.jpg"))
        try writeText(temp.appendingPathComponent("dog_in.txt"), "edit dog")

        let dataset = try DatasetLoader.loadTrainingDataset(from: temp)
        guard case .edit(let pairs) = dataset else {
            XCTFail("Expected edit dataset")
            return
        }

        XCTAssertEqual(pairs.count, 2)
        XCTAssertEqual(pairs[0].inputImageURL.lastPathComponent, "cat_in.png")
        XCTAssertEqual(pairs[0].outputImageURL.lastPathComponent, "cat_out.png")
        XCTAssertEqual(pairs[0].caption, "edit cat")
        XCTAssertEqual(pairs[1].inputImageURL.lastPathComponent, "dog_in.webp")
        XCTAssertEqual(pairs[1].outputImageURL.lastPathComponent, "dog_out.jpg")
        XCTAssertEqual(pairs[1].caption, "edit dog")
    }

    func testRejectsMixedDatasetModes() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        try writeFile(temp.appendingPathComponent("sample_in.png"))
        try writeFile(temp.appendingPathComponent("sample_out.png"))
        try writeText(temp.appendingPathComponent("sample_in.txt"), "edit sample")

        try writeImageAndCaption(in: temp, stem: "plain", imageExt: "png", caption: "plain prompt")

        XCTAssertThrowsError(try DatasetLoader.loadTrainingDataset(from: temp)) { error in
            guard case DatasetLoaderError.mixedDatasetModes = error else {
                XCTFail("Unexpected error: \(error)")
                return
            }
        }
    }

    func testLegacyTextToImageAPIRejectsEditDataset() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        try writeFile(temp.appendingPathComponent("subject_in.png"))
        try writeFile(temp.appendingPathComponent("subject_out.png"))
        try writeText(temp.appendingPathComponent("subject_in.txt"), "edit subject")

        XCTAssertThrowsError(try DatasetLoader.loadImageCaptionPairs(from: temp)) { error in
            guard case DatasetLoaderError.editDatasetNotSupportedByTextToImageAPI = error else {
                XCTFail("Unexpected error: \(error)")
                return
            }
        }
    }

    func testExcludesPreviewImagesFromTextToImageWhenRequested() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        try writeImageAndCaption(in: temp, stem: "001", imageExt: "png", caption: "first prompt")
        try writeImageAndCaption(in: temp, stem: "preview01", imageExt: "png", caption: "preview prompt")

        let dataset = try DatasetLoader.loadTrainingDataset(from: temp, excludePreviewImages: true)
        guard case .textToImage(let pairs) = dataset else {
            XCTFail("Expected text-to-image dataset")
            return
        }

        XCTAssertEqual(pairs.map { $0.imageURL.lastPathComponent }, ["001.png"])
    }

    func testExcludesPreviewImagesFromEditDetectionWhenRequested() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        try writeFile(temp.appendingPathComponent("cat_in.png"))
        try writeFile(temp.appendingPathComponent("cat_out.png"))
        try writeText(temp.appendingPathComponent("cat_in.txt"), "edit cat")

        // Preview assets should be ignored from training-set auto-discovery when requested.
        try writeFile(temp.appendingPathComponent("preview01.png"))
        try writeText(temp.appendingPathComponent("preview01.txt"), "preview prompt")

        let dataset = try DatasetLoader.loadTrainingDataset(from: temp, excludePreviewImages: true)
        guard case .edit(let pairs) = dataset else {
            XCTFail("Expected edit dataset")
            return
        }

        XCTAssertEqual(pairs.count, 1)
        XCTAssertEqual(pairs[0].inputImageURL.lastPathComponent, "cat_in.png")
        XCTAssertEqual(pairs[0].outputImageURL.lastPathComponent, "cat_out.png")
    }

    private func writeImageAndCaption(in directory: URL, stem: String, imageExt: String, caption: String) throws {
        try writeFile(directory.appendingPathComponent("\(stem).\(imageExt)"))
        try writeText(directory.appendingPathComponent("\(stem).txt"), caption)
    }

    private func writeFile(_ url: URL) throws {
        try TestFileSystem.writeFile(url, contents: Data([0]))
    }

    private func writeText(_ url: URL, _ value: String) throws {
        try TestFileSystem.writeFile(url, contents: Data(value.utf8))
    }
}
