import Foundation

public struct ManagedModelUsageTerm: Codable, Hashable, Sendable {
    public let component: String
    public let license: String
    public let summary: String
    public let sourceRepoId: String
    public let sourceRevision: String
    public let licenseURL: String

    public init(
        component: String,
        license: String,
        summary: String,
        sourceRepoId: String,
        sourceRevision: String,
        licenseURL: String
    ) {
        self.component = component
        self.license = license
        self.summary = summary
        self.sourceRepoId = sourceRepoId
        self.sourceRevision = sourceRevision
        self.licenseURL = licenseURL
    }

    private enum CodingKeys: String, CodingKey {
        case component
        case license
        case summary
        case sourceRepoId = "source_repo_id"
        case sourceRevision = "source_revision"
        case licenseURL = "license_url"
    }
}

public struct ManagedModelUsageRestriction: Hashable, Sendable {
    public let summary: String
    public let terms: [ManagedModelUsageTerm]

    public var licenseURL: String {
        terms.first?.licenseURL ?? ""
    }

    public init(summary: String, terms: [ManagedModelUsageTerm]) {
        self.summary = summary
        self.terms = terms
    }
}
