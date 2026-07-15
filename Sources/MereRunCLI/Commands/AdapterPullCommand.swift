import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import MereRunCore

struct AdapterPull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download and verify a cataloged LoRA adapter."
    )

    @Argument(help: "Canonical adapter id (for example: mere-platform-assistant).")
    var target: String

    @Flag(name: [.long], help: "Re-download even if the adapter is already installed and verified.")
    var force: Bool = false

    @Flag(name: [.short, .long], help: "Suppress progress output.")
    var quiet: Bool = false

    func run() async throws {
        guard let spec = ManagedAdapterCatalog.spec(for: target) else {
            throw ValidationError("Unknown canonical adapter id: \(target)")
        }
        let destination = spec.installedFileURL()
        let fileManager = FileManager.default

        if fileManager.fileExists(atPath: destination.path), !force {
            do {
                let verified = try spec.verifiedInstalledFileURL()
                if !quiet {
                    stderr("[\(spec.id)] already installed and verified (use --force to re-download)")
                }
                print(verified.path)
                return
            } catch {
                throw ValidationError(
                    "Existing adapter \(spec.id) failed verification: \(error.localizedDescription) Use --force to replace it."
                )
            }
        }

        let directory = spec.installDirectory()
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let partial = directory.appendingPathComponent(
            ".\(spec.artifact.filename).partial.\(UUID().uuidString)",
            isDirectory: false
        )
        defer { try? fileManager.removeItem(at: partial) }

        if !quiet {
            stderr("[\(spec.id)] downloading \(spec.downloadURL.absoluteString)")
        }
        var request = URLRequest(url: spec.downloadURL)
        request.httpMethod = "GET"
        let temporary = try await downloadWithTransientRetries(request, adapterID: spec.id)
        try fileManager.moveItem(at: temporary, to: partial)

        let partialPin = ModelArtifactPin(
            filename: partial.lastPathComponent,
            byteCount: spec.artifact.byteCount,
            sha256: spec.artifact.sha256
        )
        do {
            _ = try partialPin.verify(in: directory)
        } catch {
            throw ValidationError("Downloaded adapter failed verification: \(error.localizedDescription)")
        }

        if fileManager.fileExists(atPath: destination.path) {
            _ = try fileManager.replaceItemAt(destination, withItemAt: partial)
        } else {
            try fileManager.moveItem(at: partial, to: destination)
        }
        let verified = try spec.verifiedInstalledFileURL()
        if !quiet {
            stderr("[\(spec.id)] verified SHA-256 \(spec.artifact.sha256)")
        }
        print(verified.path)
    }

    private func stderr(_ message: String) {
        CLIStderr.write(message + "\n")
    }

    private func downloadWithTransientRetries(
        _ request: URLRequest,
        adapterID: String
    ) async throws -> URL {
        let maximumAttempts = 3
        for attempt in 1...maximumAttempts {
            let temporary: URL
            let response: URLResponse
            do {
                (temporary, response) = try await URLSession.shared.download(for: request)
            } catch {
                guard attempt < maximumAttempts else { throw error }
                if !quiet {
                    stderr("[\(adapterID)] transient download error; retrying (\(attempt + 1)/\(maximumAttempts))")
                }
                try await Task.sleep(nanoseconds: UInt64(attempt) * 250_000_000)
                continue
            }

            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if status == 200 {
                return temporary
            }
            try? FileManager.default.removeItem(at: temporary)
            guard (500...599).contains(status), attempt < maximumAttempts else {
                throw ValidationError("Adapter download failed (HTTP \(status)).")
            }
            if !quiet {
                stderr("[\(adapterID)] server returned HTTP \(status); retrying (\(attempt + 1)/\(maximumAttempts))")
            }
            try await Task.sleep(nanoseconds: UInt64(attempt) * 250_000_000)
        }
        throw ValidationError("Adapter download failed after \(maximumAttempts) attempts.")
    }
}
