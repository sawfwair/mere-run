import Foundation
import MereRunContract

/// The checks a command needs beyond what its contract can say — the ones a page used to make
/// before Run — applied to the console form every task-draft and Command-view run validates
/// through (`StudioConsoleCommand.validationMessage`), so the workspace's banner and the Command
/// view's agree, and a run the CLI would refuse never reaches it.
package enum StudioCommandChecks {
    /// What the catalog's own validation and this check both say when InstantMesh has the wrong
    /// number of views.
    package static let instantMeshViewCountMessage = "Add exactly 4 or 6 ordered source views."

    /// The reason `draft` cannot run `capability` yet, or nil.
    package static func message(for capability: MereRunCommandCapability, draft: StudioConsoleDraft) -> String? {
        if let problem = renoiseMessage(for: capability, draft: draft) { return problem }
        switch capability.id {
        case MereRunCapabilityCatalog.imageReconstruct3DMultiview.id:
            return instantMeshMessage(draft: draft)
        case MereRunCapabilityCatalog.visionGeometryMultiview.id:
            return geometryMultiviewMessage(draft: draft)
        case MereRunCapabilityCatalog.speechDiarize.id:
            return diarizeMessage(draft: draft)
        default:
            return nil
        }
    }

    /// Woosh's renoise is one amount or one amount per step; the CLI rejects anything else
    /// (`parseRenoiseSchedule`), so the run is refused with the same objection first.
    private static func renoiseMessage(for capability: MereRunCommandCapability, draft: StudioConsoleDraft) -> String? {
        guard capability.options.contains(where: { $0.flag == "--renoise" }) else { return nil }
        let argument = draft.text("--renoise")
        guard !argument.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        let schedule = StudioRenoise(mode: StudioRenoise.inferredMode(argument: argument), argument: argument)
        let templateID = CommandTemplateID.allCases.first { $0.capability?.id == capability.id }
        return schedule.problems(steps: StudioRenoise.stepCount(in: draft, templateID: templateID)).first
    }

    /// Multi-view geometry solves relative cameras between views; one picture is the
    /// single-view command's job, so Studio asks for two, as its page did.
    private static func geometryMultiviewMessage(draft: StudioConsoleDraft) -> String? {
        let views = draft.arguments.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        return views.count < 2 ? "Add at least two ordered views." : nil
    }

    /// `speech diarize` rejects a streaming input buffer for any model but Nemotron 3, which
    /// the Sortformer default is; the message names the fix before the CLI does.
    private static func diarizeMessage(draft: StudioConsoleDraft) -> String? {
        let latency = draft.text("--latency")
        let model = draft.text("--model")
        guard !latency.isEmpty, latency != "offline", !model.localizedCaseInsensitiveContains("nemotron") else { return nil }
        return "Input buffer latency applies to Nemotron 3 only; choose Offline for \(model.isEmpty ? "Sortformer" : model)."
    }

    /// InstantMesh reconstructs from exactly four or six ordered views, and a supplied camera
    /// file must be one the CLI reads (`InstantMeshCameraDocument`) with one camera per view.
    private static func instantMeshMessage(draft: StudioConsoleDraft) -> String? {
        let views = StudioAttachmentSlot.separatedPaths(draft.text("--view"))
        guard views.count == 4 || views.count == 6 else { return instantMeshViewCountMessage }
        let camerasPath = draft.text("--cameras").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !camerasPath.isEmpty else { return nil }
        return cameraDocument(at: URL(fileURLWithPath: NSString(string: camerasPath).expandingTildeInPath))
            .problem(viewCount: views.count)
    }

    // MARK: Camera file cache

    /// A camera file as last read: its document, or why it would not read. Validation runs on
    /// every render of the Command view and the composer, so the file is read again only when
    /// its modification date changes.
    private enum CameraFile {
        case document(StudioInstantMeshCameraDocument)
        case unreadable(String)

        func problem(viewCount: Int) -> String? {
            switch self {
            case .document(let document): return document.problems(viewCount: viewCount).first
            case .unreadable(let reason): return reason
            }
        }
    }

    private static let cameraFileLock = NSLock()
    nonisolated(unsafe) private static var cameraFiles: [String: (modified: Date?, file: CameraFile)] = [:]

    private static func cameraDocument(at url: URL) -> CameraFile {
        let modified = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        cameraFileLock.lock()
        defer { cameraFileLock.unlock() }
        if let known = cameraFiles[url.path], known.modified == modified, modified != nil {
            return known.file
        }
        let file: CameraFile
        do {
            file = .document(try StudioInstantMeshCameraDocument.importing(Data(contentsOf: url)))
        } catch {
            file = .unreadable("The camera file at \(url.lastPathComponent) could not be read: \(error.localizedDescription)")
        }
        cameraFiles[url.path] = (modified, file)
        return file
    }
}
