import Foundation
import MereRunCore

extension YuE2Resources {
    func configuration() throws -> YuE2Configuration {
        let value = try JSONDecoder().decode(
            YuE2Configuration.self, from: Data(contentsOf: rootURL.appendingPathComponent("config.json"))
        )
        try value.validateReleased()
        return value
    }

    func vaeConfiguration() throws -> YuE2VAEConfiguration {
        let value = try JSONDecoder().decode(
            YuE2VAEConfiguration.self, from: Data(contentsOf: vaeURL.appendingPathComponent("config.json"))
        )
        try value.validateReleased()
        return value
    }
}
