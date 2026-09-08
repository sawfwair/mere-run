import XCTest
import MereRunCore

final class ModelManagementCompatibilityTests: XCTestCase {
    func testCoreRetainsModelTypesAndRuntimeTemplates() {
        let id: ModelResolver.ModelID = .kleinNano
        let portableID: ManagedModelID = id
        let manifest: MereRunCore.MereRunModelManifest = .template(for: id)
        XCTAssertEqual(manifest.id, portableID.rawValue)
        XCTAssertEqual(manifest.engine, .flux2Klein)
    }
}
