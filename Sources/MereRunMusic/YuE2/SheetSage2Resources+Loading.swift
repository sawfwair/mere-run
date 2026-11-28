import Foundation
import MLX
import MereRunCore
import MereRunSheetSageModel

extension SheetSage2Resources {
    func load() throws -> SheetSage2Model {
        let config = try JSONDecoder().decode(
            SheetSage2Configuration.self,
            from: Data(contentsOf: rootURL.appendingPathComponent("config.json"))
        )
        try config.validate()
        let parentURL = rootURL.appendingPathComponent("mert2/model.safetensors")
        guard try ModelArtifactPin.fileSHA256(parentURL) == "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d" else {
            throw SheetSage2Error.invalid("MERT2 parent does not match the adapter's pinned checkpoint.")
        }
        return try SheetSage2Model(
            configuration: config,
            parent: SafetensorsStreamingLoader.loadArrays(url: parentURL),
            adapter: SafetensorsStreamingLoader.loadArrays(url: rootURL.appendingPathComponent("model.safetensors"))
        )
    }
}
