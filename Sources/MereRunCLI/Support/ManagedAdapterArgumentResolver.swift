import Foundation
import MereRunCore

enum ManagedAdapterArgumentResolver {
    static func resolve(
        _ reference: String?,
        baseModelID: String,
        adaptersRoot: URL = MereRunModelPaths.adaptersDir,
        fileManager: FileManager = .default
    ) throws -> String? {
        guard let reference = reference?.trimmingCharacters(in: .whitespacesAndNewlines),
              !reference.isEmpty else {
            return nil
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
