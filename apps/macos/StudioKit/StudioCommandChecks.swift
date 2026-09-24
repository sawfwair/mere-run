import Foundation
import MereRunContract

/// The checks a command needs beyond what its contract can say — the ones a page used to make
/// before Run — applied to the console form every task-draft and Command-view run validates
/// through (`StudioConsoleCommand.validationMessage`), so the workspace's banner and the Command
/// view's agree, and an incomplete InstantMesh run never reaches the CLI.
package enum StudioCommandChecks {
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
        guard views.count == 4 || views.count == 6 else {
            return "Add exactly 4 or 6 ordered source views."
        }
        let camerasPath = draft.text("--cameras").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !camerasPath.isEmpty else { return nil }
        let url = URL(fileURLWithPath: NSString(string: camerasPath).expandingTildeInPath)
        do {
            let cameras = try StudioInstantMeshCameraDocument.importing(Data(contentsOf: url))
            return cameras.problems(viewCount: views.count).first
        } catch {
            return "The camera file at \(url.lastPathComponent) could not be read: \(error.localizedDescription)"
        }
    }
}
