import ArgumentParser
import Foundation
import MereRunCore
import MereRunRelayKit

// MereRunRelayKit owns the relay client, executor profiles, and workflow wire
// types; it is platform-neutral, so the CLI supplies its storage locations
// here and converts RelayClientError into ValidationError at the command
// boundary to keep messages, usage output, and exit codes unchanged.

func mapRelayErrors<T>(_ body: () throws -> T) throws -> T {
    do {
        return try body()
    } catch let error as RelayClientError {
        throw ValidationError(error.message)
    }
}

func mapRelayErrors<T>(_ body: () async throws -> T) async throws -> T {
    do {
        return try await body()
    } catch let error as RelayClientError {
        throw ValidationError(error.message)
    }
}

extension WorkflowExecutorProfileStore {
    static var fileURL: URL {
        MereRunModelPaths.applicationSupportBase.appendingPathComponent("executors.json")
    }

    static func load(fileManager: FileManager = .default) throws -> WorkflowExecutorProfiles {
        try mapRelayErrors { try load(from: fileURL, fileManager: fileManager) }
    }

    static func save(_ value: WorkflowExecutorProfiles, fileManager: FileManager = .default) throws {
        try mapRelayErrors { try save(value, to: fileURL, fileManager: fileManager) }
    }

    static func require(_ reference: String) throws -> WorkflowExecutorProfile {
        try mapRelayErrors { try require(reference, profilesURL: fileURL) }
    }
}

extension RelayAuthentication {
    static func defaultTokenFile(profileName: String) -> URL {
        defaultTokenFile(
            profileName: profileName,
            applicationSupportBase: MereRunModelPaths.applicationSupportBase
        )
    }
}

extension WorkflowExecutorProbe {
    static func local(fileManager: FileManager = .default) -> WorkflowExecutorProbe {
        #if os(Linux)
        let platform = "linux"
        let backend = ProcessInfo.processInfo.environment["MERERUN_LINUX_ACCEL"] ?? "cpu"
        let memoryBytes = backend == "cuda"
            ? linuxNVIDIAMemoryBytes() ?? ProcessInfo.processInfo.physicalMemory
            : ProcessInfo.processInfo.physicalMemory
        #else
        let platform = "macos"
        let backend = "metal"
        let memoryBytes = ProcessInfo.processInfo.physicalMemory
        #endif
        #if arch(arm64)
        let architecture = "arm64"
        #elseif arch(x86_64)
        let architecture = "x86_64"
        #else
        let architecture = "unknown"
        #endif
        let fileSystemAttributes = try? FileManager.default.attributesOfFileSystem(
            forPath: MereRunModelPaths.applicationSupportBase.path
        )
        let disk = (fileSystemAttributes?[.systemFreeSize] as? NSNumber)?.int64Value
        let graphProviders = WorkflowGraphProviderRegistry.discoveredCatalog().providers
        return WorkflowExecutorProbe(
            schemaVersion: 1,
            workerVersion: MereRunCLIVersion.current,
            contractVersions: [WorkflowJobManifest.contractVersion],
            platform: platform,
            architecture: architecture,
            acceleratorBackend: backend,
            memoryBytes: memoryBytes,
            systemMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            logicalCPUCores: ProcessInfo.processInfo.processorCount,
            availableDiskBytes: disk ?? nil,
            networkAccess: true,
            nodeKinds: Array(Set(
                WorkflowNodeRegistry.entries.map(\.kind) + graphProviders.flatMap { $0.nodes.map(\.kind) }
            )).sorted(),
            installedModelIDs: ModelInventory.snapshot(
                mode: .verified,
                fileManager: fileManager
            ).rows.filter(\.isInstalled).map(\.id).sorted(),
            availableSecretNames: availableWorkflowSecretNames(),
            providers: graphProviders.map(\.requirement)
        )
    }

    private static func availableWorkflowSecretNames(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String] {
        let prefix = "MERERUN_SECRET_"
        return environment.keys.compactMap { key -> String? in
            guard key.hasPrefix(prefix), environment[key]?.isEmpty == false else { return nil }
            return key.dropFirst(prefix.count).lowercased().replacingOccurrences(of: "_", with: "-")
        }.sorted()
    }

    #if os(Linux)
    private static func linuxNVIDIAMemoryBytes() -> UInt64? {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
        process.standardOutput = output
        process.standardError = Pipe()
        guard (try? process.run()) != nil else { return nil }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return parseNVIDIAMemoryBytes(String(decoding: data, as: UTF8.self))
    }
    #endif

    static func parseNVIDIAMemoryBytes(_ output: String) -> UInt64? {
        output.split(whereSeparator: \Character.isNewline)
            .compactMap { line in
                line.split(whereSeparator: \Character.isWhitespace).first.flatMap { UInt64($0) }
            }
            .max()
            .map { $0 * 1_024 * 1_024 }
    }
}
