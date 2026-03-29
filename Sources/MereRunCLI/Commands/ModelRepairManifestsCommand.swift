import ArgumentParser
import Foundation
import MereRunCore

struct ModelRepairManifests: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "repair-manifests",
        abstract: "Write missing mererun_model.json for known models in the local mere.run model store."
    )

    @Flag(name: [.long], help: "Print what would change without writing files.")
    var dryRun: Bool = false

    func run() throws {
        let fm = FileManager.default

        let modelDirs = resolveCandidateModelDirs()
        if modelDirs.isEmpty {
            throw ValidationError("Could not locate any model directories to repair.")
        }

        var wroteCount = 0
        var alreadyCount = 0
        var skippedCount = 0

        for modelId in ModelResolver.ModelID.allCases {
            var foundAnyModel = false
            for modelsDir in modelDirs {
                let modelRoot = modelsDir.appendingPathComponent(modelId.rawValue, isDirectory: true)
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: modelRoot.path, isDirectory: &isDir), isDir.boolValue else {
                    continue
                }
                foundAnyModel = true

                let manifestURL = MereRunModelManifest.url(in: modelRoot)
                if fm.fileExists(atPath: manifestURL.path) {
                    alreadyCount += 1
                    print("[ok] \(modelId.rawValue): \(manifestURL.path)")
                    continue
                }

                if dryRun {
                    wroteCount += 1
                    print("[dry-run] would write \(modelId.rawValue) -> \(manifestURL.path)")
                    continue
                }

                do {
                    _ = try MereRunModelManifest.writeTemplateIfKnown(modelId: modelId.rawValue, to: modelRoot)
                    wroteCount += 1
                    print("[wrote] \(modelId.rawValue): \(manifestURL.path)")
                } catch {
                    skippedCount += 1
                    print("[skip] \(modelId.rawValue): \(error.localizedDescription)")
                }
            }

            if !foundAnyModel {
                skippedCount += 1
                print("[skip] \(modelId.rawValue): model directory not found")
            }
        }

        print("")
        print("Summary: wrote=\(wroteCount) already=\(alreadyCount) skipped=\(skippedCount)")
        if dryRun {
            print("Run again without `--dry-run` to apply.")
        }
    }

    private func resolveCandidateModelDirs() -> [URL] {
        [MereRunModelPaths.modelsDir]
    }
}
