import Foundation

public struct IdolResources: Sendable {
    public static let repository = "speridlabs/idol"
    public static let revision = "2731c2291fc65a797c5cb627a5e5526fc1490902"
    public static let referenceRevision = "4702a7f00850e34b6e83a2170cc123c3e9570744"
    public static let pins: [ModelArtifactPin] = [
        .init(filename: "birefnet.safetensors", byteCount: 444466856,
              sha256: "44bbc97b041f879975d0d6d538b8e66a25eb04d1f49aab40697544cdd97eaa8e"),
        .init(filename: "denoiser.safetensors", byteCount: 1482258308,
              sha256: "ded2cacf671bf3978ebc62729087abdbd318ef4752e6805b4c9fc8ffa51fe716"),
        .init(filename: "decoder.safetensors", byteCount: 203171696,
              sha256: "9da4d7676bfcadb1c3fdfbff796e6257dadf82d26ec05be7717c79b2b13a1c4e"),
        .init(filename: "decoder_gs.safetensors", byteCount: 949083036,
              sha256: "3087eb8ee5763206a53e4078acf2a1bf382dab5147d407ec13a2cf39c04cc544")
    ]
    public let root: URL
    public let dinoRoot: URL
    public let vaeRoot: URL
    public init(root: URL, dinoRoot: URL, vaeRoot: URL) {
        self.root = root; self.dinoRoot = dinoRoot; self.vaeRoot = vaeRoot
    }

    public func configuration() throws -> IdolConfiguration {
        let configuration = try JSONDecoder().decode(IdolConfiguration.self, from: Data(contentsOf: root.appending(path: "config.json")))
        try configuration.validate()
        guard configuration.image_size == 1024, configuration.denoiser.args.q_token_length == 8192,
              configuration.denoiser.args.model_channels == 1024,
              configuration.denoiser.args.cond_channels == 1280, configuration.denoiser.args.cond2_channels == 128,
              configuration.image_encoders.dinov3 == "facebook/dinov3-vith16plus-pretrain-lvd1689m",
              configuration.image_encoders.flux2_vae == "black-forest-labs/FLUX.2-dev" else {
            throw IdolError.invalidConfiguration("Expected the released IDOL configuration and image encoders.")
        }
        return configuration
    }

    public func verify() throws -> [ModelArtifactPin] {
        _ = try configuration()
        var verified: [ModelArtifactPin] = []
        for pin in Self.pins { _ = try pin.verify(in: root); verified.append(pin) }
        for (prefix, directory, filenames) in [
            ("idol", root, ["config.json"]),
            ("dinov3", dinoRoot, ["config.json", "model.safetensors"]),
            ("flux2-vae", vaeRoot, ["config.json", "diffusion_pytorch_model.safetensors"])
        ] {
            for name in filenames {
                let file = directory.appending(path: name)
                verified.append(ModelArtifactPin(filename: prefix + "/" + name,
                    byteCount: try ModelArtifactPin.fileByteCount(file), sha256: try ModelArtifactPin.fileSHA256(file)))
            }
        }
        return verified
    }
}
