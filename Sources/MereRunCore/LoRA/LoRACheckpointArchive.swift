import Foundation
#if os(Linux)
import Glibc
#endif

public enum LoRACheckpointArchive {
    public static func archiveURL(for checkpointURL: URL) -> URL {
        checkpointURL.deletingPathExtension().appendingPathExtension("zip")
    }

    @discardableResult
    public static func createZipBundle(
        primaryFile: URL,
        additionalFiles: [URL] = []
    ) throws -> URL? {
        let archiveURL = archiveURL(for: primaryFile)
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent(
            "mererun-lora-archive-\(UUID().uuidString)",
            isDirectory: true
        )
        try fm.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }
        let bundleDir = tempRoot.appendingPathComponent(
            primaryFile.deletingPathExtension().lastPathComponent,
            isDirectory: true
        )
        try fm.createDirectory(at: bundleDir, withIntermediateDirectories: true)

        var staged: [URL] = [primaryFile]
        staged.append(contentsOf: additionalFiles)
        for file in staged {
            guard fm.fileExists(atPath: file.path) else { continue }
            let destination = bundleDir.appendingPathComponent(file.lastPathComponent, isDirectory: false)
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            try fm.copyItem(at: file, to: destination)
        }

        if fm.fileExists(atPath: archiveURL.path) {
            try fm.removeItem(at: archiveURL)
        }

        #if os(macOS)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-c", "-k", "--sequesterRsrc", "--keepParent", bundleDir.path, archiveURL.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "LoRACheckpointArchive",
                code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: "Failed to create checkpoint archive with ditto (status \(process.terminationStatus))."]
            )
        }
        #elseif os(Linux)
        let command = [
            "cd \(shellQuote(tempRoot.path))",
            "zip -qry \(shellQuote(archiveURL.path)) \(shellQuote(bundleDir.lastPathComponent))"
        ].joined(separator: " && ")
        guard system(command) == 0 else {
            throw NSError(
                domain: "LoRACheckpointArchive",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create checkpoint archive with zip."]
            )
        }
        #else
        return nil
        #endif

        return archiveURL
    }

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }
}
