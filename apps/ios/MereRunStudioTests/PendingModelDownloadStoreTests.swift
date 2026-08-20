import Foundation
import XCTest

final class PendingModelDownloadStoreTests: XCTestCase {
    private var suiteName: String!
    private var defaults: UserDefaults!

    override func setUpWithError() throws {
        try super.setUpWithError()
        suiteName = "PendingModelDownloadStoreTests.\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        suiteName = nil
        try super.tearDownWithError()
    }

    func testSaveReplacesSameModelAndPreservesTerms() throws {
        let store = PendingModelDownloadStore(defaults: defaults, key: "downloads")
        let first = PendingModelDownload(
            modelID: "image-bonsai-binary",
            usageTermsAcknowledged: false,
            requestedAt: Date(timeIntervalSince1970: 1)
        )
        let replacement = PendingModelDownload(
            modelID: first.modelID,
            usageTermsAcknowledged: true,
            requestedAt: Date(timeIntervalSince1970: 2),
            allowsCellular: true
        )

        try store.save(first)
        try store.save(replacement)

        XCTAssertEqual(store.all(), [replacement])
    }

    func testRemoveClearsOnlyCompletedModel() throws {
        let store = PendingModelDownloadStore(defaults: defaults, key: "downloads")
        let image = PendingModelDownload(
            modelID: "image-bonsai-binary",
            usageTermsAcknowledged: false,
            requestedAt: Date(timeIntervalSince1970: 1)
        )
        let chat = PendingModelDownload(
            modelID: "text-chat-lfm25-2.6b-4bit",
            usageTermsAcknowledged: true,
            requestedAt: Date(timeIntervalSince1970: 2)
        )
        try store.save(image)
        try store.save(chat)

        try store.remove(modelID: image.modelID)

        XCTAssertEqual(store.all(), [chat])
    }

    func testLegacyRecordDefaultsToWifiOnly() throws {
        let legacy = """
        [{"modelID":"image-bonsai-binary","usageTermsAcknowledged":false,"requestedAt":0}]
        """
        defaults.set(try XCTUnwrap(legacy.data(using: .utf8)), forKey: "downloads")

        let download = try XCTUnwrap(PendingModelDownloadStore(defaults: defaults, key: "downloads").all().first)

        XCTAssertFalse(download.allowsCellular)
    }
}
