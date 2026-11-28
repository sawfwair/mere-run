import Foundation
import MereRunCore
import MereRunUnlimitedOCRModel

extension UnlimitedOCRResources {
    package static func configuration(root: URL) throws -> UnlimitedOCRConfiguration {
        try UnlimitedOCRConfiguration.load(Data(contentsOf: root.appendingPathComponent("config.json")))
    }
}
