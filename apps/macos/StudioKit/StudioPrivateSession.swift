import Combine
import Foundation
import MereRunContract

/// One temporary workspace. Its Library never reads or writes the saved Library files.
@MainActor
package final class StudioPrivateSession {
    package let directory: URL
    package let library: StudioLibraryStore
    package let savedLibrary: StudioLibraryStore
    package var requestIDs = Set<UUID>()
    package private(set) var keptIDs = Set<UUID>()
    private var completionSubscription: AnyCancellable?

    private static let directoryPrefix = "mere-run-private-"

    package init(savedLibrary: StudioLibraryStore, controller: MereRunController, directory: URL? = nil) throws {
        self.savedLibrary = savedLibrary
        // The owner's process id tells a later launch whether this workspace was abandoned.
        self.directory = directory ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("\(Self.directoryPrefix)\(getpid())-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: self.directory, withIntermediateDirectories: true,
                                              attributes: [.posixPermissions: 0o700])
        // Deleted rows can be undone in memory. Their files stay inside this workspace until
        // cleanup, rather than escaping to the system Trash.
        library = StudioLibraryStore(libraryURL: self.directory.appendingPathComponent("library.json"),
            trashItem: { _ in nil }, persists: false)
        library.observe(controller: controller)
    }

    /// A crash or force quit never reaches `removeFiles`. Workspaces of other running instances
    /// stay; a reused process id only postpones removal to a later launch.
    package static func removeAbandonedWorkspaces(
        in parent: URL = FileManager.default.temporaryDirectory,
        isRunning: (pid_t) -> Bool = { kill($0, 0) == 0 || errno == EPERM }
    ) {
        let contents = (try? FileManager.default.contentsOfDirectory(at: parent, includingPropertiesForKeys: nil)) ?? []
        for url in contents where url.lastPathComponent.hasPrefix(directoryPrefix) {
            let owner = url.lastPathComponent.dropFirst(directoryPrefix.count).prefix { $0 != "-" }
            guard let pid = pid_t(owner), !isRunning(pid) else { continue }
            try? FileManager.default.removeItem(at: url)
        }
    }

    package func prepareDirectories(for request: StudioRunRequest) throws {
        let root = directory.appendingPathComponent(request.id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        guard let form = request.execution?.form, let capability = request.templateID.capability else { return }
        for option in capability.options where StudioTaskSessions.privateOutputFlags(for: capability).contains(option.flag) {
            let path = form.text(option.flag)
            guard path.hasPrefix(root.path + "/") else { continue }
            let url = URL(fileURLWithPath: path)
            let parent = option.kind == .directory ? url : url.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        }
    }

    /// Copy first, then publish a durable row. A failure leaves the temporary result intact.
    package func keep(_ item: StudioLibraryItem) throws {
        guard item.status != .running && item.status != .queued else {
            throw StudioValidationError(message: "Wait for this run to finish before keeping it.")
        }
        let destination = StudioOutputLocation.appOutputsRoot().appendingPathComponent("Kept", isDirectory: true)
            .appendingPathComponent(item.id.uuidString, isDirectory: true)
        let files = referencedFiles(of: item).filter { $0.standardizedFileURL.path.hasPrefix(directory.path + "/") }
            .sorted { $0.path.count < $1.path.count }
        var copied: [String] = []
        for file in files where !copied.contains(where: { file.path == $0 || file.path.hasPrefix($0 + "/") }) {
            let relative = String(file.standardizedFileURL.path.dropFirst(directory.path.count + 1))
            let target = destination.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.copyItem(at: file, to: target)
            }
            copied.append(file.path)
        }
        let encoded = try JSONEncoder.mereRunApp.encode(item)
        let json = try JSONDecoder.mereRunApp.decode(StudioLibraryJSON.self, from: encoded)
        let relocated = json.relocatingPrivatePaths(from: directory, to: destination)
        let kept = try JSONDecoder.mereRunApp.decode(StudioLibraryItem.self, from: JSONEncoder.mereRunApp.encode(relocated))
        savedLibrary.upsert(kept)
        if let error = savedLibrary.lastPersistenceError { throw StudioValidationError(message: error) }
        keptIDs.insert(item.id)
    }

    /// Everything the row's paths can name: its results, every input the run read, and the
    /// declared sidecars the run actually wrote. Keep rewrites all of those paths.
    private func referencedFiles(of item: StudioLibraryItem) -> [URL] {
        var paths = item.allArtifactURLs.map(\.path) + StudioLibraryLineage.inputPaths(of: item)
            + (item.messages ?? []).compactMap(\.imagePath)
        if let arguments = item.commandArguments, let capability = item.templateID?.capability {
            let form = StudioConsoleCommand.seed(capability: capability, arguments: arguments)
            paths += StudioTaskSessions.privateOutputFlags(for: capability).map { form.text($0) }
                .filter { FileManager.default.fileExists(atPath: $0) }
        }
        return paths.map { URL(fileURLWithPath: $0) }
    }

    /// Cancel private jobs, then delete only this workspace after every writer has stopped.
    package func retire(controller: MereRunController) {
        for job in controller.jobs.all where job.request.requestID.map(requestIDs.contains) == true && job.state.isActive {
            controller.jobs.cancel(job.id)
        }
        completionSubscription = controller.jobs.completions.sink { [weak self, weak controller] _ in
            Task { @MainActor in
                guard let self, let controller else { return }
                self.cleanIfFinished(controller: controller)
            }
        }
        cleanIfFinished(controller: controller)
    }

    private func cleanIfFinished(controller: MereRunController) {
        guard !controller.jobs.all.contains(where: {
            $0.request.requestID.map(requestIDs.contains) == true && $0.state.isActive
        }) else { return }
        guard removeFiles() else {
            controller.status = library.lastPersistenceError ?? "Could not remove private files."
            return
        }
        controller.jobs.discardFinished(requestIDs: requestIDs)
        completionSubscription = nil
        controller.retiredPrivateSessions.removeAll { $0 === self }
    }

    @discardableResult
    package func removeFiles() -> Bool {
        guard FileManager.default.fileExists(atPath: directory.path) else { return true }
        do { try FileManager.default.removeItem(at: directory); return true }
        catch { library.lastPersistenceError = "Could not remove private files: \(error.localizedDescription)"; return false }
    }
}

extension StudioTaskSessions {
    static func privateOutputFlags(for capability: MereRunCommandCapability) -> Set<String> {
        StudioTaskSchema.outputFlags(for: capability).union(StudioTaskSchema.chosenOutputFlags)
            .union(["--run-dir", "--mask-output-dir"])
    }

    /// Relocate every declared output, including explicit destinations and durable run records.
    package func privatizing(_ request: StudioRunRequest, source: StudioScopeSource) -> StudioRunRequest {
        guard let directory = privateOutputDirectory, let capability = request.templateID.capability else { return request }
        var execution = request.execution ?? StudioExecution(templateID: request.templateID,
            arguments: request.template.arguments(from: request.draft, source: source))
        guard let form = execution.form else { return request }
        let root = directory.appendingPathComponent(request.id.uuidString, isDirectory: true)
        for option in capability.options where Self.privateOutputFlags(for: capability).contains(option.flag)
            && (option.kind == .file || option.kind == .directory) {
            let path = form.text(option.flag)
            guard !path.isBlank, !path.hasPrefix(root.path + "/") else { continue }
            let name = URL(fileURLWithPath: path).lastPathComponent
            let output = root.appendingPathComponent(String(option.flag.dropFirst(2)), isDirectory: true)
                .appendingPathComponent(name.isEmpty ? "output" : name)
            execution = execution.replacing(option.flag, with: output.path)
        }
        return StudioRunRequest(id: request.id, mode: request.mode, templateID: request.templateID,
            template: request.template, draft: execution.project(onto: request.draft), createdAt: request.createdAt,
            conversationID: request.conversationID, execution: execution, parentID: request.parentID)
    }
}

extension MereRunController {
    package func setPrivateMode(_ enabled: Bool, savedLibrary: StudioLibraryStore) throws {
        guard enabled != (privateSession != nil) else { return }
        if enabled {
            let session = try StudioPrivateSession(savedLibrary: savedLibrary, controller: self)
            taskSessions.beginPrivateSession(directory: session.directory)
            privateSession = session
        } else if let session = privateSession {
            privateSession = nil
            taskSessions.endPrivateSession()
            retiredPrivateSessions.append(session)
            session.retire(controller: self)
        }
        clearPrivatePresentation()
    }

    package func discardPrivateWork(preserving templateID: CommandTemplateID? = nil) throws {
        guard let previous = privateSession else { return }
        let next = try StudioPrivateSession(savedLibrary: previous.savedLibrary, controller: self)
        privateSession = next
        taskSessions.beginPrivateSession(directory: next.directory)
        if let templateID {
            taskSessions.setTaskDraft(StudioTaskDraft(templateID: templateID), for: templateID.studioTask)
        }
        retiredPrivateSessions.append(previous)
        previous.retire(controller: self)
        clearPrivatePresentation()
    }

}

private extension StudioLibraryJSON {
    func relocatingPrivatePaths(from source: URL, to destination: URL) -> Self {
        switch self {
        case .string(let text):
            return .string(text.replacingOccurrences(of: source.path + "/", with: destination.path + "/"))
        case .array(let values): return .array(values.map { $0.relocatingPrivatePaths(from: source, to: destination) })
        case .object(let values):
            return .object(Dictionary(uniqueKeysWithValues: values.map { key, value in
                (key.replacingOccurrences(of: source.path + "/", with: destination.path + "/"),
                 value.relocatingPrivatePaths(from: source, to: destination))
            }))
        default: return self
        }
    }
}
