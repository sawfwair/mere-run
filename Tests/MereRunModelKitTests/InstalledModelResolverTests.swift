import Foundation
import XCTest
import MereRunModelKit

final class InstalledModelResolverTests: XCTestCase {
    private var root: URL!
    private let id = ManagedModelID.kleinNano

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func install(_ path: String, id: ManagedModelID? = nil, acknowledged: Bool? = nil) throws -> URL {
        let directory = root.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if let id {
            try MereRunModelManifest(id: id.rawValue, usageTermsAcknowledged: acknowledged).write(to: directory)
        }
        return directory.standardizedFileURL
    }

    func testPrimaryThenBindingThenSearchRootWithRuntimeValidation() throws {
        let primary = try install("models/\(id.rawValue)", id: id)
        let binding = try install("external", id: id)
        let search = try install("search/\(id.rawValue)", id: id)
        let resolver = InstalledModelResolver(locations: ModelLocationSnapshot(
            primaryRoot: root.appendingPathComponent("models"),
            searchRoots: [root.appendingPathComponent("search")],
            bindings: [.init(modelID: id.rawValue, path: binding.path)]
        ))
        for (accepted, expectedKind) in [(primary, ModelLocationKind.primaryStore),
                                        (binding, .registeredBinding), (search, .registeredSearchRoot)] {
            var visited: [URL] = []
            let result = try resolver.resolve(id, descriptor: { InstalledModelDescriptor(id: $0) }) { modelID, url in
                XCTAssertEqual(modelID, self.id)
                visited.append(url)
                return url == accepted
            }
            XCTAssertEqual(result.candidate.rootURL, accepted)
            XCTAssertEqual(result.candidate.kind, expectedKind)
            XCTAssertEqual(visited, Array([primary, binding, search].prefix(visited.count)))
            XCTAssertEqual(visited.last, accepted)
        }
    }

    func testFallbackRetainsRequestedIdentityAndReportsInstalledIdentity() throws {
        let fallback = ManagedModelID.kleinMax
        let fallbackRoot = try install(fallback.rawValue, id: fallback)
        let resolver = InstalledModelResolver(locations: ModelLocationSnapshot(primaryRoot: root))
        let result = try resolver.resolve(id, descriptor: {
            InstalledModelDescriptor(id: $0, fallbackIDs: $0 == self.id ? [fallback] : [])
        }) { validatedID, url in
            XCTAssertEqual(validatedID, fallback)
            XCTAssertEqual(url, fallbackRoot)
            return true
        }
        XCTAssertEqual(result.requestedModelID, id)
        XCTAssertEqual(result.installedModelID, fallback)
    }

    func testMissingErrorIncludesPrimaryAndFallbackLocationsInOrder() throws {
        let fallback = ManagedModelID.kleinMax
        let locations = ModelLocationSnapshot(primaryRoot: root, searchRoots: [root.appendingPathComponent("search")])
        let resolver = InstalledModelResolver(locations: locations)
        XCTAssertThrowsError(try resolver.resolve(id, descriptor: {
            InstalledModelDescriptor(id: $0, upstreamRepoID: "example/model", fallbackIDs: [fallback])
        }, validateRuntime: { _, _ in XCTFail("Missing directory reached runtime validation"); return true })) {
            guard let error = $0 as? InstalledModelResolver.ResolverError else { return XCTFail("Unexpected error: \($0)") }
            XCTAssertEqual(error.modelID, self.id)
            XCTAssertEqual(error.upstreamRepoID, "example/model")
            XCTAssertEqual(error.searchedPaths, [self.id, fallback].flatMap { locations.candidates(for: $0.rawValue).map(\.rootURL) })
        }
    }

    func testInvalidManifestCannotBypassIdentityThroughBinding() throws {
        let binding = try install("external", id: .kleinMax)
        let resolver = InstalledModelResolver(locations: ModelLocationSnapshot(
            primaryRoot: root, bindings: [.init(modelID: id.rawValue, path: binding.path)]
        ))
        for contents in [try Data(contentsOf: MereRunModelManifest.url(in: binding)), Data("invalid JSON".utf8)] {
            try contents.write(to: MereRunModelManifest.url(in: binding))
            XCTAssertThrowsError(try resolver.resolve(id, descriptor: { InstalledModelDescriptor(id: $0) },
                validateRuntime: { _, _ in XCTFail("Invalid metadata reached runtime validation"); return true }))
        }
    }

    func testManifestlessDirectoriesRequireExplicitBinding() throws {
        _ = try install(id.rawValue)
        let external = try install("search/\(id.rawValue)")
        let unbound = InstalledModelResolver(locations: ModelLocationSnapshot(
            primaryRoot: root, searchRoots: [root.appendingPathComponent("search")]
        ))
        XCTAssertThrowsError(try unbound.resolve(id, descriptor: { InstalledModelDescriptor(id: $0) },
            validateRuntime: { _, _ in XCTFail("Unbound directory reached runtime validation"); return true }))
        let bound = InstalledModelResolver(locations: ModelLocationSnapshot(
            primaryRoot: root, bindings: [.init(modelID: id.rawValue, path: external.path)]
        ))
        let result = try bound.resolve(id, descriptor: { InstalledModelDescriptor(id: $0) }, validateRuntime: { _, _ in true })
        XCTAssertEqual(result.candidate.rootURL, external)
    }

    func testRestrictedBindingRequiresRegistryOrManifestAcknowledgement() throws {
        let external = try install("external")
        for manifestPresent in [false, true] {
            for registryAcknowledged in [false, true] {
                for manifestAcknowledged in [false, true] {
                    if manifestPresent {
                        try MereRunModelManifest(id: id.rawValue, usageTermsAcknowledged: manifestAcknowledged).write(to: external)
                    }
                    let resolver = InstalledModelResolver(locations: ModelLocationSnapshot(
                        primaryRoot: root,
                        bindings: [.init(modelID: id.rawValue, path: external.path, usageTermsAcknowledged: registryAcknowledged)]
                    ))
                    var validated = false
                    let result = try? resolver.resolve(id, descriptor: {
                        InstalledModelDescriptor(id: $0, requiresUsageTermsAcknowledgement: true)
                    }, validateRuntime: { _, _ in validated = true; return true })
                    let accepted = registryAcknowledged || (manifestPresent && manifestAcknowledged)
                    XCTAssertEqual(result != nil, accepted)
                    XCTAssertEqual(validated, accepted)
                }
            }
        }
    }

    func testDescriptorCannotSubstituteAnotherModelIdentity() throws {
        _ = try install(id.rawValue, id: .kleinMax)
        let resolver = InstalledModelResolver(locations: ModelLocationSnapshot(primaryRoot: root))
        XCTAssertThrowsError(try resolver.resolve(id, descriptor: { _ in InstalledModelDescriptor(id: .kleinMax) },
            validateRuntime: { _, _ in XCTFail("Mismatched descriptor reached runtime validation"); return true }))
    }
}
