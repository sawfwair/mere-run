import Combine
import Foundation
import MereRunContract

/// One temporary workspace. Its Library never reads or writes the saved Library files.
@MainActor
package final class StudioPrivateSession {
    package enum Phase: Equatable {
        case active
        case waitingForJobs(Set<JobID>)
        case removingFiles
        case removalFailed(String)
        /// Shutdown can confirm the native writers stopped before their main-actor callbacks
        /// arrive. Keep owning those completions even though the files have been removed.
        case waitingForCompletions(Set<JobID>)
        case removed
    }

    package let directory: URL
    package let library: StudioLibraryStore
    package let savedLibrary: StudioLibraryStore
    package private(set) var requestIDs = Set<UUID>()
    package private(set) var keptIDs = Set<UUID>()
    package private(set) var phase: Phase = .active
    private var completionSubscription: AnyCancellable?
    private let removeDirectory: (URL) throws -> Void

    private static let directoryPrefix = "mere-run-private-"

    package init(
        savedLibrary: StudioLibraryStore, controller: MereRunController, directory: URL? = nil,
        removeDirectory: @escaping (URL) throws -> Void = { try FileManager.default.removeItem(at: $0) }
    ) throws {
        self.savedLibrary = savedLibrary
        self.removeDirectory = removeDirectory
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

    /// A crash or force quit never finishes retirement. Workspaces of other running instances
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
        try requireActive()
        requestIDs.insert(request.id)
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
        try requireActive()
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
        guard phase == .active else { return }
        controller.retiredPrivateSessions.append(self)
        let jobs = activeJobs(controller: controller)
        phase = .waitingForJobs(Set(jobs.map(\.id)))
        // Subscribe before cancelling: queued jobs complete synchronously. Completions run
        // after all finished-event observers, including the controller's privacy filtering.
        completionSubscription = controller.jobs.completions.sink { [weak self, weak controller] result in
            guard let self, let controller, let id = result.requestID, self.requestIDs.contains(id) else { return }
            self.advanceRetirement(controller: controller)
        }
        // Remove queued work first so freeing a running slot cannot launch another private job.
        for job in jobs.filter(\.state.isQueued) + jobs.filter({ !$0.state.isQueued }) {
            controller.jobs.cancel(job.id)
        }
        advanceRetirement(controller: controller)
    }

    /// Only the process runner can confirm native shutdown while main-actor delivery is blocked
    /// by application termination. Unconfirmed writers continue to own their files.
    package func confirmStoppedJobs(_ ids: Set<JobID>, controller: MereRunController) {
        guard case .waitingForJobs(let pending) = phase else { return }
        phase = .waitingForJobs(pending.subtracting(ids))
        advanceRetirement(controller: controller)
    }

    package func retryRemoval(controller: MereRunController) {
        guard case .removalFailed = phase else { return }
        removeFiles(controller: controller)
    }

    private func activeJobs(controller: MereRunController) -> [Job] {
        controller.jobs.all.filter { $0.request.requestID.map(requestIDs.contains) == true && $0.state.isActive }
    }

    private func advanceRetirement(controller: MereRunController) {
        switch phase {
        case .waitingForJobs(let pending):
            let remaining = pending.filter { controller.jobs.job($0)?.state.isActive == true }
            phase = .waitingForJobs(remaining)
            if remaining.isEmpty { removeFiles(controller: controller) }
        case .waitingForCompletions(let pending):
            let remaining = pending.filter { controller.jobs.job($0)?.state.isActive == true }
            phase = .waitingForCompletions(remaining)
            if remaining.isEmpty { finishRetirement(controller: controller) }
        case .active, .removingFiles, .removalFailed, .removed:
            break
        }
    }

    private func finishRetirement(controller: MereRunController) {
        phase = .removed
        controller.jobs.discardFinished(requestIDs: requestIDs)
        completionSubscription = nil
        controller.retiredPrivateSessions.removeAll { $0 === self }
    }

    private func removeFiles(controller: MereRunController) {
        phase = .removingFiles
        do {
            if FileManager.default.fileExists(atPath: directory.path) { try removeDirectory(directory) }
            library.lastPersistenceError = nil
            phase = .waitingForCompletions(Set(activeJobs(controller: controller).map(\.id)))
            advanceRetirement(controller: controller)
        } catch {
            let message = "Could not remove private files: \(error.localizedDescription)"
            phase = .removalFailed(message)
            library.lastPersistenceError = message
            controller.status = message
        }
    }

    private func requireActive() throws {
        guard phase == .active else {
            throw StudioValidationError(message: "This private workspace has closed.")
        }
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
            session.retire(controller: self)
        }
        clearPrivatePresentation()
        if case .removalFailed(let message)? = retiredPrivateSessions.last?.phase { status = message }
    }

    package func discardPrivateWork(preserving templateID: CommandTemplateID? = nil) throws {
        guard let previous = privateSession else { return }
        let next = try StudioPrivateSession(savedLibrary: previous.savedLibrary, controller: self)
        privateSession = next
        taskSessions.beginPrivateSession(directory: next.directory)
        if let templateID {
            taskSessions.setTaskDraft(StudioTaskDraft(templateID: templateID), for: templateID.studioTask)
        }
        previous.retire(controller: self)
        clearPrivatePresentation()
        if case .removalFailed(let message) = previous.phase { status = message }
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
