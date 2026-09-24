import Foundation
import MereRunContract

/// The checks a command needs beyond what its contract can say — the ones a page used to make
/// before Run — applied to the console form every task-draft and Command-view run validates
/// through (`StudioConsoleCommand.validationMessage`), so the workspace's banner and the Command
/// view's agree, and an incomplete InstantMesh run never reaches the CLI.
package enum StudioCommandChecks {
    /// What the catalog's own validation and this check both say when InstantMesh has the wrong
    /// number of views.
    package static let instantMeshViewCountMessage = "Add exactly 4 or 6 ordered source views."

    /// The reason `draft` cannot run `capability` yet, or nil.
    package static func message(for capability: MereRunCommandCapability, draft: StudioConsoleDraft) -> String? {
        switch capability.id {
        case MereRunCapabilityCatalog.imageReconstruct3DMultiview.id:
            return instantMeshMessage(draft: draft)
        default:
            return nil
        }
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
