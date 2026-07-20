import ArgumentParser
import Foundation
import MereRunCore

struct AdapterList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List cataloged LoRA adapters and their install state."
    )

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func run() throws {
        let output = ManagedAdapterListOutput(
            schemaVersion: 1,
            adapterStore: MereRunModelPaths.adaptersDir.path,
            adapters: ManagedAdapterCatalog.allSpecs.map(ManagedAdapterListItem.init)
        )
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            print(String(decoding: try encoder.encode(output), as: UTF8.self))
            return
        }

        for adapter in output.adapters {
            let state = adapter.installed ? "installed" : "not installed"
            print("\(adapter.id)\t\(adapter.version)\t\(adapter.baseModelID)\t\(state)")
        }
    }
}

struct ManagedAdapterListOutput: Codable, Equatable {
    let schemaVersion: Int
    let adapterStore: String
    let adapters: [ManagedAdapterListItem]
}

struct ManagedAdapterListItem: Codable, Equatable {
    let id: String
    let title: String
    let version: String
    let summary: String
    let baseModelID: String
    let format: String
    let license: String
    let upstreamRevision: String?
    let releaseManifestURL: String
    let downloadURL: String
    let filename: String
    let byteCount: Int64
    let sha256: String
    let installed: Bool
    let path: String?

    init(_ spec: ManagedAdapterSpec) {
        let installed = spec.isInstalled()
        self.id = spec.id
        self.title = spec.title
        self.version = spec.version
        self.summary = spec.summary
        self.baseModelID = spec.baseModelID
        self.format = spec.format
        self.license = spec.license
        self.upstreamRevision = spec.upstreamRevision
        self.releaseManifestURL = spec.releaseManifestURL.absoluteString
        self.downloadURL = spec.downloadURL.absoluteString
        self.filename = spec.artifact.filename
        self.byteCount = spec.artifact.byteCount
        self.sha256 = spec.artifact.sha256
        self.installed = installed
        self.path = installed ? spec.installedFileURL().path : nil
    }
}
