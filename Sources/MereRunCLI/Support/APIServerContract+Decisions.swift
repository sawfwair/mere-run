import Foundation
import MereRunCore

struct LayaAPIRequest: Decodable, Sendable {
    let model: String
    let request: LayaDecisionRequest

    private enum CodingKeys: String, CodingKey { case model }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        request = try LayaDecisionRequest(from: decoder)
    }

    func validate() throws {
        guard LayaCatalog.modelIDs.contains(model) else {
            throw APIRequestValidationError.invalidField("model", "select an installed managed Laya model")
        }
        do {
            try request.validate()
        } catch {
            throw APIRequestValidationError.invalidField("request", error.localizedDescription)
        }
    }
}
