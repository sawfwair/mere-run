import XCTest
@testable import MereRunCore

final class ManagedModelCatalogTests: XCTestCase {
    func testEveryCanonicalManagedModelIDHasCatalogSpec() {
        for modelID in ModelResolver.ModelID.allCases {
            let spec = ManagedModelCatalog.spec(for: modelID.rawValue)
            XCTAssertNotNil(spec, "Missing catalog spec for \(modelID.rawValue)")
        }
    }

    func testEveryCatalogSpecMapsToManifestTemplate() {
        for spec in ManagedModelCatalog.allSpecs {
            guard let modelID = ModelResolver.ModelID(rawValue: spec.id) else {
                XCTFail("Catalog spec does not map to a canonical ModelID: \(spec.id)")
                continue
            }

            let manifest = MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
            XCTAssertEqual(manifest.id, spec.id)
        }
    }

    func testAllRuntimeAutoDownloadSpecsHaveManagedSource() {
        for spec in ManagedModelCatalog.allSpecs where spec.runtimeAutoDownloadAllowed {
            XCTAssertTrue(
                spec.archiveSource != nil || spec.hubFallback != nil,
                "Runtime auto-download model \(spec.id) has no configured managed source."
            )
        }
    }
}
