import Foundation
import MereRunCore

enum ManagedAdapterArgumentResolver {
    static func resolve(
        _ reference: String?,
        baseModelID: String,
        adaptersRoot: URL = MereRunModelPaths.adaptersDir,
        fileManager: FileManager = .default,
        requireInstalled: Bool = true
    ) throws -> String? {
        guard let reference = reference?.trimmingCharacters(in: .whitespacesAndNewlines),
              !reference.isEmpty else {
            return nil
        }
        if !requireInstalled, let spec = ManagedAdapterCatalog.spec(for: reference) {
            guard spec.baseModelID == baseModelID else {
                throw ManagedAdapterResolutionError.incompatibleBaseModel(
                    adapterID: spec.id,
                    expected: spec.baseModelID,
                    actual: baseModelID
                )
            }
            return spec.installedFileURL(adaptersRoot: adaptersRoot).path
        }
        if let installed = try ManagedAdapterCatalog.resolveInstalledReference(
            reference,
            baseModelID: baseModelID,
            adaptersRoot: adaptersRoot,
            fileManager: fileManager
        ) {
            return installed.path
        }
        return URL(fileURLWithPath: reference).standardizedFileURL.path
    }
}
