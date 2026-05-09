import Foundation

enum CLIBootstrapInstallError: LocalizedError {
    case missingBinary(String)

    var errorDescription: String? {
        switch self {
        case .missingBinary(let path):
            return "Bundled CLI binary not found at \(path)."
        }
    }
}

enum CLIBootstrapInstallOutcome: Equatable {
    case installed(URL)
    case alreadyInstalled(URL)
    case skippedNoBundledCLI
    case failed(String)
}

enum CLIBootstrapInstaller {
    static func installBundledCLIIfNeeded(
        fileManager fm: FileManager = .default,
        bundle: Bundle = .main
    ) -> CLIBootstrapInstallOutcome {
        if let installed = CLIResolver.existingInstalledCLI(fileManager: fm) {
            return .alreadyInstalled(installed)
        }

        guard let payloadURL = bundledPayloadURL(fileManager: fm, bundle: bundle) else {
            return .skippedNoBundledCLI
        }

        let destinationURL = preferredAutomaticInstallURL(fileManager: fm)
        do {
            try installBundledCLI(from: payloadURL, to: destinationURL, fileManager: fm)
            return .installed(destinationURL)
        } catch {
            return .failed(error.localizedDescription)
        }
    }

    static func installBundledCLI(
        from payloadURL: URL,
        to destinationURL: URL,
        fileManager fm: FileManager = .default
    ) throws {
        let binaryURL = payloadURL.appendingPathComponent("mere.run", isDirectory: false)
        guard fm.isExecutableFile(atPath: binaryURL.path) else {
            throw CLIBootstrapInstallError.missingBinary(binaryURL.path)
        }

        let destinationDirectoryURL = destinationURL.deletingLastPathComponent()
        try fm.createDirectory(at: destinationDirectoryURL, withIntermediateDirectories: true)
        try copyReplacingItem(at: binaryURL, to: destinationURL, fileManager: fm)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: destinationURL.path)

        for assetURL in supportAssetURLs(in: payloadURL, fileManager: fm) {
            try copyReplacingItem(
                at: assetURL,
                to: destinationDirectoryURL.appendingPathComponent(assetURL.lastPathComponent),
                fileManager: fm
            )
        }
    }

    static func bundledPayloadURL(
        fileManager fm: FileManager = .default,
        bundle: Bundle = .main
    ) -> URL? {
        guard let resourceURL = bundle.resourceURL else {
            return nil
        }

        let payloadURL = resourceURL.appendingPathComponent("mere.run", isDirectory: true)
        let binaryURL = payloadURL.appendingPathComponent("mere.run", isDirectory: false)
        return fm.isExecutableFile(atPath: binaryURL.path) ? payloadURL : nil
    }

    static func preferredAutomaticInstallURL(fileManager fm: FileManager = .default) -> URL {
        preferredAutomaticInstallURL(
            fileManager: fm,
            installDirectoryCandidates: [
                URL(fileURLWithPath: "/usr/local/bin", isDirectory: true),
                URL(fileURLWithPath: "/opt/homebrew/bin", isDirectory: true),
            ]
        )
    }

    static func preferredAutomaticInstallURL(
        fileManager fm: FileManager = .default,
        installDirectoryCandidates: [URL]
    ) -> URL {
        for candidate in installDirectoryCandidates where candidate.isWritableDirectory(fileManager: fm) {
            return candidate.appendingPathComponent("mere.run", isDirectory: false)
        }
        return fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".local", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("mere.run", isDirectory: false)
    }

    private static func supportAssetURLs(in payloadURL: URL, fileManager fm: FileManager) -> [URL] {
        let contents = (try? fm.contentsOfDirectory(
            at: payloadURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        return contents
            .filter { url in
                url.pathExtension == "framework"
                    || url.pathExtension == "bundle"
                    || url.lastPathComponent == "Resources"
                    || url.lastPathComponent == "mlx.metallib"
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func copyReplacingItem(at sourceURL: URL, to destinationURL: URL, fileManager fm: FileManager) throws {
        if fm.fileExists(atPath: destinationURL.path) {
            try fm.removeItem(at: destinationURL)
        }
        try fm.copyItem(at: sourceURL, to: destinationURL)
    }
}

private extension URL {
    func isWritableDirectory(fileManager fm: FileManager) -> Bool {
        var isDirectory: ObjCBool = false
        return fm.fileExists(atPath: path, isDirectory: &isDirectory)
            && isDirectory.boolValue
            && fm.isWritableFile(atPath: path)
    }
}
