import Foundation

struct PendingModelDownload: Codable, Equatable, Sendable {
    let modelID: String
    let usageTermsAcknowledged: Bool
    let requestedAt: Date
    let allowsCellular: Bool

    init(
        modelID: String,
        usageTermsAcknowledged: Bool,
        requestedAt: Date,
        allowsCellular: Bool = false
    ) {
        self.modelID = modelID
        self.usageTermsAcknowledged = usageTermsAcknowledged
        self.requestedAt = requestedAt
        self.allowsCellular = allowsCellular
    }

    private enum CodingKeys: String, CodingKey {
        case modelID
        case usageTermsAcknowledged
        case requestedAt
        case allowsCellular
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try container.decode(String.self, forKey: .modelID)
        usageTermsAcknowledged = try container.decode(Bool.self, forKey: .usageTermsAcknowledged)
        requestedAt = try container.decode(Date.self, forKey: .requestedAt)
        allowsCellular = try container.decodeIfPresent(Bool.self, forKey: .allowsCellular) ?? false
    }
}

struct PendingModelDownloadStore {
    static let defaultKey = "local.pendingModelDownloads"

    private let defaults: UserDefaults
    private let key: String

    init(
        defaults: UserDefaults = .standard,
        key: String = PendingModelDownloadStore.defaultKey
    ) {
        self.defaults = defaults
        self.key = key
    }

    func all() -> [PendingModelDownload] {
        guard let data = defaults.data(forKey: key),
              let downloads = try? JSONDecoder().decode([PendingModelDownload].self, from: data) else {
            return []
        }
        return downloads.sorted { $0.requestedAt < $1.requestedAt }
    }

    func save(_ download: PendingModelDownload) throws {
        var downloads = all().filter { $0.modelID != download.modelID }
        downloads.append(download)
        try persist(downloads)
    }

    func remove(modelID: String) throws {
        try persist(all().filter { $0.modelID != modelID })
    }

    private func persist(_ downloads: [PendingModelDownload]) throws {
        if downloads.isEmpty {
            defaults.removeObject(forKey: key)
            return
        }
        defaults.set(try JSONEncoder().encode(downloads), forKey: key)
    }
}
