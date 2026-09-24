import Foundation
import XCTest
@testable import MereRunCore

final class MereRunModelPathsTests: XCTestCase {
    private let legacyModelsDirEnvironmentKey = ["ZE", "RO", "MODELS", "DIR"].joined(separator: "_")
    private let legacyApplicationSupportName = ["Ze", "ro"].joined()
    private var registrationDomain: [String: Any] = [:]

    override func setUp() {
        super.setUp()
        registrationDomain = UserDefaults.standard.volatileDomain(forName: UserDefaults.registrationDomain)
    }

    override func tearDown() {
        UserDefaults.standard.setVolatileDomain(registrationDomain, forName: UserDefaults.registrationDomain)
        MereRunModelPaths.setProcessModelsDirOverride(nil)
        unsetenv(MereRunModelPaths.modelsDirEnvironmentKey)
        unsetenv(legacyModelsDirEnvironmentKey)
        UserDefaults.standard.removeObject(forKey: MereRunModelPaths.modelStorageActivePathDefaultsKey)
        super.tearDown()
    }

    func testResolveModelDirReturnsPrimaryWhenPrimaryValid() throws {
        let fm = FileManager.default
        let modelId = "mererun-test-primary-\(UUID().uuidString)"
        let primary = MereRunModelPaths.modelDir(modelId)
        let marker = primary.appendingPathComponent("marker.txt")

        defer {
            try? fm.removeItem(at: primary)
        }

        try fm.createDirectory(at: primary, withIntermediateDirectories: true)
        try Data("ok".utf8).write(to: marker)

        let resolved = MereRunModelPaths.resolveModelDir(modelId) { candidate in
            fm.fileExists(atPath: candidate.appendingPathComponent("marker.txt").path)
        }

        XCTAssertEqual(resolved.standardizedFileURL, primary.standardizedFileURL)
    }

    func testResolveModelDirFallsBackToPrimaryWhenNoCandidateValid() {
        let modelId = "mererun-test-fallback-\(UUID().uuidString)"
        let primary = MereRunModelPaths.modelDir(modelId)

        let resolved = MereRunModelPaths.resolveModelDir(modelId) { _ in false }

        XCTAssertEqual(resolved.standardizedFileURL, primary.standardizedFileURL)
    }

    func testModelStoreResolutionUsesProcessOverride() throws {
        let fm = FileManager.default
        let customRoot = try makeTemporaryDirectory(named: "mererun-models-override")
        defer { try? fm.removeItem(at: customRoot) }

        MereRunModelPaths.setProcessModelsDirOverride(customRoot)

        let resolution = MereRunModelPaths.modelStoreResolution()

        XCTAssertEqual(resolution.source, .processOverride)
        XCTAssertEqual(resolution.activeModelsDir.standardizedFileURL, customRoot.standardizedFileURL)
        XCTAssertFalse(resolution.isFallbackToDefault)
        XCTAssertEqual(
            MereRunModelPaths.modelDir("mererun-test").standardizedFileURL,
            customRoot.appendingPathComponent("mererun-test", isDirectory: true).standardizedFileURL
        )
    }

    func testProcessOverrideCanOptIntoRegisteredLocations() throws {
        let customRoot = try makeTemporaryDirectory(named: "mererun-models-with-locations")
        defer { try? FileManager.default.removeItem(at: customRoot) }

        MereRunModelPaths.setProcessModelsDirOverride(
            customRoot,
            includeRegisteredLocations: true
        )

        XCTAssertTrue(MereRunModelPaths.includesRegisteredModelLocations(environment: [:]))
    }

    func testModelStoreResolutionUsesMissingProcessOverrideWithoutFallback() throws {
        let temp = try makeTemporaryDirectory(named: "mererun-models-missing-process-parent")
        defer { try? FileManager.default.removeItem(at: temp) }
        let missingRoot = temp.appendingPathComponent("models", isDirectory: true)

        MereRunModelPaths.setProcessModelsDirOverride(missingRoot)

        let resolution = MereRunModelPaths.modelStoreResolution()

        XCTAssertEqual(resolution.source, .processOverride)
        XCTAssertEqual(resolution.activeModelsDir.standardizedFileURL.path, missingRoot.standardizedFileURL.path)
        XCTAssertEqual(resolution.configuredModelsDir?.standardizedFileURL.path, missingRoot.standardizedFileURL.path)
        XCTAssertFalse(resolution.isFallbackToDefault)
    }

    func testModelStoreResolutionUsesEnvironmentOverride() throws {
        let fm = FileManager.default
        let envRoot = try makeTemporaryDirectory(named: "mererun-models-env")
        defer { try? fm.removeItem(at: envRoot) }

        setenv(MereRunModelPaths.modelsDirEnvironmentKey, envRoot.path, 1)

        let resolution = MereRunModelPaths.modelStoreResolution()

        XCTAssertEqual(resolution.source, .environment)
        XCTAssertEqual(resolution.activeModelsDir.standardizedFileURL, envRoot.standardizedFileURL)
        XCTAssertFalse(resolution.isFallbackToDefault)
    }

    func testModelStoreResolutionUsesMissingEnvironmentOverrideWithoutFallback() throws {
        let temp = try makeTemporaryDirectory(named: "mererun-models-missing-env-parent")
        defer { try? FileManager.default.removeItem(at: temp) }
        let missingRoot = temp.appendingPathComponent("models", isDirectory: true)

        setenv(MereRunModelPaths.modelsDirEnvironmentKey, missingRoot.path, 1)

        let resolution = MereRunModelPaths.modelStoreResolution()

        XCTAssertEqual(resolution.source, .environment)
        XCTAssertEqual(resolution.activeModelsDir.standardizedFileURL.path, missingRoot.standardizedFileURL.path)
        XCTAssertEqual(resolution.configuredModelsDir?.standardizedFileURL.path, missingRoot.standardizedFileURL.path)
        XCTAssertFalse(resolution.isFallbackToDefault)
    }

    func testModelStoreResolutionFallsBackWhenPersistedPathUnavailable() {
        let missingPath = "/Volumes/mererun-missing-\(UUID().uuidString)"
        let defaults = makeDefaults([MereRunModelPaths.modelStorageActivePathDefaultsKey: missingPath])

        let resolution = MereRunModelPaths.modelStoreResolution(defaults: defaults)

        XCTAssertEqual(resolution.source, .persisted)
        XCTAssertTrue(resolution.isFallbackToDefault)
        XCTAssertEqual(
            resolution.activeModelsDir.standardizedFileURL,
            MereRunModelPaths.defaultModelsDir.standardizedFileURL
        )
        XCTAssertEqual(
            resolution.configuredModelsDir?.standardizedFileURL.path,
            URL(fileURLWithPath: missingPath).standardizedFileURL.path
        )
    }

    func testModelStoreResolutionIgnoresLegacyEnvironmentVariable() {
        let resolution = MereRunModelPaths.modelStoreResolution(
            environment: [legacyModelsDirEnvironmentKey: "/Volumes/legacy-models"],
            defaults: makeDefaults()
        )

        XCTAssertEqual(resolution.source, .default)
        XCTAssertEqual(
            resolution.activeModelsDir.standardizedFileURL,
            MereRunModelPaths.defaultModelsDir.standardizedFileURL
        )
        XCTAssertNil(resolution.configuredModelsDir)
    }

    func testResolveModelDirDoesNotSearchLegacyApplicationSupportRootWhenActiveRootMisses() throws {
        let fm = FileManager.default
        let modelId = "mererun-test-no-legacy-fallback-\(UUID().uuidString)"
        let activeRoot = try makeTemporaryDirectory(named: "mererun-models-active")
        MereRunModelPaths.setProcessModelsDirOverride(activeRoot)

        let legacyModelRoot = try fm
            .url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent(legacyApplicationSupportName, isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelId, isDirectory: true)
        let marker = legacyModelRoot.appendingPathComponent("marker.txt")

        defer {
            try? fm.removeItem(at: activeRoot)
            try? fm.removeItem(at: legacyModelRoot)
        }

        try fm.createDirectory(at: legacyModelRoot, withIntermediateDirectories: true)
        try Data("ok".utf8).write(to: marker)

        let resolved = MereRunModelPaths.resolveModelDir(modelId) { candidate in
            fm.fileExists(atPath: candidate.appendingPathComponent("marker.txt").path)
        }

        XCTAssertEqual(
            resolved.standardizedFileURL,
            activeRoot.appendingPathComponent(modelId, isDirectory: true).standardizedFileURL
        )
    }

    /// Defaults holding `values` only in this process's registration domain, which is never
    /// written to disk. A throwaway suite leaves an empty plist in ~/Library/Preferences even
    /// after its persistent domain is removed. tearDown puts the registration domain back.
    private func makeDefaults(_ values: [String: Any] = [:]) -> UserDefaults {
        UserDefaults.standard.register(defaults: values)
        return UserDefaults(suiteName: "MereRunModelPathsTests")!
    }

    private func makeTemporaryDirectory(named prefix: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(prefix)-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
