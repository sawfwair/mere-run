import Foundation
import MereRunCore

package enum MereRunVideoRuntime {
    package static func registerImageBackends() {
        ImageGenerationOperation.registerCosmos3SuperGeneratorFactory {
            Cosmos3SuperImageGenerator()
        }
    }
}
