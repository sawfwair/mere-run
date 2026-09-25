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
        case MereRunCapabilityCatalog.imageRunPlan.id:
            return runPlanMessage(draft: draft)
        case MereRunCapabilityCatalog.imageValidate.id:
            return validateMessage(draft: draft)
        default:
            return nil
        }
    }

    /// `image run-plan` refuses `--materialize` beside `--preflight`; a check does not write the
    /// run folder.
    private static func runPlanMessage(draft: StudioConsoleDraft) -> String? {
        !draft.text("--preflight").isBlank && !draft.text("--materialize").isBlank
            ? "Choose Preflight or Materialize, not both: a preflight does not write the run folder." : nil
    }

    /// `image validate --compare` compares against a reference folder, and without
    /// `--reference-dir` compares nothing.
    private static func validateMessage(draft: StudioConsoleDraft) -> String? {
        !draft.text("--compare").isBlank && draft.text("--reference-dir").isBlank
            ? "Choose the reference folder to compare against." : nil
    }

    /// Woosh's renoise is one amount or one amount per step; the CLI rejects anything else
    /// (`parseRenoiseSchedule`), so the run is refused with the same objection first.
    private static func renoiseMessage(for capability: MereRunCommandCapability, draft: StudioConsoleDraft) -> String? {
        guard capability.options.contains(where: { $0.flag == "--renoise" }) else { return nil }
        let templateID = CommandTemplateID.allCases.first { $0.capability?.id == capability.id }
        return StudioRenoise.problems(
            argument: draft.text("--renoise"), steps: StudioRenoise.stepCount(in: draft, templateID: templateID)
        ).first
    }

    /// Multi-view geometry solves relative cameras between views; one picture is the
    /// single-view command's job, so Studio asks for two, as its page did. A supplied camera
    /// file must be one the CLI reads (`DepthAnything3CameraDocument`) with one camera per view,
    /// each sized like its picture; the page refused a run whose cameras did not match rather
    /// than letting the model estimate them.
    private static func geometryMultiviewMessage(draft: StudioConsoleDraft) -> String? {
        let views = draft.arguments.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard views.count >= 2 else { return "Add at least two ordered views." }
        return cameraFileProblem(in: draft, importing: StudioGeometryCameraDocument.importing) { document in
            document.problems(views: views.map { path in
                let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
                return StudioCameraView(name: url.lastPathComponent, pixelSize: StudioPixelSize.of(url))
            })
        }
    }

    /// InstantMesh reconstructs from exactly four or six ordered views, and a supplied camera
    /// file must be one the CLI reads (`InstantMeshCameraDocument`) with one camera per view.
    private static func instantMeshMessage(draft: StudioConsoleDraft) -> String? {
        let views = StudioAttachmentSlot.separatedPaths(draft.text("--view"))
        guard views.count == 4 || views.count == 6 else { return instantMeshViewCountMessage }
        return cameraFileProblem(in: draft, importing: StudioInstantMeshCameraDocument.importing) { document in
            document.problems(viewCount: views.count)
        }
    }

    // MARK: Camera files

    /// The first objection to the draft's `--cameras` file: why it will not read as `Document`,
    /// else the first of `problems`. Nil when the draft names no file.
    private static func cameraFileProblem<Document>(
        in draft: StudioConsoleDraft,
        importing: (Data) throws -> Document,
        problems: (Document) -> [String]
    ) -> String? {
        let path = draft.text("--cameras").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        do {
            return problems(try importing(cameraFileContents(at: url))).first
        } catch {
            return "The camera file at \(url.lastPathComponent) could not be read: \(error.localizedDescription)"
        }
    }

    /// A camera file's bytes as last read. Validation runs on every render of the Command view
    /// and the composer, so the file is read again only when its modification date changes.
    private static let cameraFileLock = NSLock()
    nonisolated(unsafe) private static var cameraFiles: [String: (modified: Date, contents: Data)] = [:]

    private static func cameraFileContents(at url: URL) throws -> Data {
        let modified = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
        cameraFileLock.lock()
        defer { cameraFileLock.unlock() }
        if let modified, let known = cameraFiles[url.path], known.modified == modified {
            return known.contents
        }
        let contents = try Data(contentsOf: url)
        if let modified { cameraFiles[url.path] = (modified, contents) }
        return contents
    }
}
