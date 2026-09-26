@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
import AppKit
import Foundation
import SwiftUI
import XCTest

/// Offscreen snapshots of the composer's Recent prompts button, the inspectors' Defaults
/// section, and the Keyboard Shortcuts window, light and dark. Skipped unless
/// `MERERUN_STUDIO_SNAPSHOT_DIR` names a directory, like `StudioSnapshotTests`; everything the
/// views read lives in a temporary folder, and no CLI process starts.
@MainActor
final class StudioHistoryDefaultsSnapshotTests: XCTestCase {
    private var outputDirectory: URL!
    private var root: URL!
    private var controller: MereRunController!

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard let path = ProcessInfo.processInfo.environment["MERERUN_STUDIO_SNAPSHOT_DIR"],
              !path.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw XCTSkip("Set MERERUN_STUDIO_SNAPSHOT_DIR to a directory to write Studio snapshots.")
        }
        outputDirectory = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        root = FileManager.default.temporaryDirectory.appendingPathComponent("history-defaults-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(),
            resolvesCLIOnInit: false, taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
        controller.modelIdentities.use(nil)
    }

    override func tearDownWithError() throws {
        controller?.terminateAllProcesses()
        controller = nil
        if let root { try? FileManager.default.removeItem(at: root) }
        try super.tearDownWithError()
    }

    private func write<Content: View>(_ view: Content, size: CGSize, appearance: StudioSnapshotAppearance, name: String) throws {
        let rep = try StudioSnapshotRenderer.render(view, size: size, appearance: appearance, settle: 1.5)
        XCTAssertGreaterThan(StudioSnapshotRenderer.nonBlankCoverage(of: rep), 0.05, "\(name) rendered blank")
        let data = try XCTUnwrap(rep.representation(using: .png, properties: [:]))
        try data.write(to: outputDirectory.appendingPathComponent("\(name).png"), options: .atomic)
    }

    /// The Image composer over a Library with three earlier prompts, so the clock sits beside
    /// Run; then the same composer with no history, where it does not.
    func testRecentPromptsButtonSnapshots() throws {
        let prompts = ["a ceramic mug in soft morning light", "a lighthouse at dusk, long exposure", "a fox in fresh snow"]
        let items = prompts.enumerated().map { index, prompt in
            StudioLibraryItem(
                id: UUID(), mode: .createImage, prompt: prompt, inputURL: nil, outputURL: nil,
                createdAt: StudioSnapshotRenderer.referenceDate.addingTimeInterval(Double(-index) * 600),
                updatedAt: StudioSnapshotRenderer.referenceDate, status: .completed, exitCode: 0,
                commandPreview: "mere.run image generate", outputText: nil
            )
        }
        let size = CGSize(width: 760, height: 190)
        for appearance in StudioSnapshotAppearance.allCases {
            for (name, history) in [("recent-prompts", items), ("recent-prompts-empty", [])] {
                let view = ComposerHost()
                    .environmentObject(controller)
                    .environment(\.studioLibraryItems, history)
                    .frame(width: size.width, height: size.height)
                    .background(MereRunTheme.background)
                try write(view, size: size, appearance: appearance, name: "\(name)-\(appearance.rawValue)")
            }
        }
    }

    /// The Defaults section in its three states — nothing saved with changes to save, saved and
    /// current, and at the app's defaults — then the whole Enhance inspector with it at the end.
    func testInspectorDefaultsSnapshots() throws {
        let states: [(String, StudioPageDefaultsStatus)] = [
            ("Changed, nothing saved", StudioPageDefaultsStatus(hasSaved: false, canSave: true, canRestore: true)),
            ("Saved and current", StudioPageDefaultsStatus(hasSaved: true, canSave: false, canRestore: true)),
            ("At the app's defaults", StudioPageDefaultsStatus(hasSaved: false, canSave: false, canRestore: false)),
        ]
        let width = StudioLayoutPolicy.inspectorWidth
        for appearance in StudioSnapshotAppearance.allCases {
            let view = VStack(spacing: 0) {
                ForEach(states, id: \.0) { state in
                    StudioInspectorDefaultsSection(defaults: StudioInspectorPageDefaults(status: state.1, save: {}, restore: {}))
                    Divider()
                }
            }
            .frame(width: width)
            .background(MereRunTheme.background)
            try write(view, size: CGSize(width: width, height: 390), appearance: appearance,
                      name: "inspector-defaults-\(appearance.rawValue)")
        }

        let sessions = controller.taskSessions
        var draft = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        draft.form["--dtype"] = .text("float16")
        sessions.setTaskDraft(draft, for: .audioEnhance)
        for appearance in StudioSnapshotAppearance.allCases {
            let inspector = InspectorHost(task: .audioEnhance)
                .environmentObject(controller)
                .environment(\.studioTaskSessions, sessions)
                .frame(width: width, height: 1_100)
            try write(inspector, size: CGSize(width: width, height: 1_100), appearance: appearance,
                      name: "task-inspector-defaults-\(appearance.rawValue)")
        }
    }

    /// Help ▸ Keyboard Shortcuts, tall enough to show every section.
    func testKeyboardShortcutsWindowSnapshots() throws {
        let size = CGSize(width: 460, height: 1_320)
        for appearance in StudioSnapshotAppearance.allCases {
            try write(StudioKeyboardShortcutsView().frame(width: size.width, height: size.height), size: size,
                      appearance: appearance, name: "keyboard-shortcuts-\(appearance.rawValue)")
        }
    }
}

/// The Image composer with a draft of its own and a focus state to hand it.
private struct ComposerHost: View {
    @State private var draft: StudioDraft = {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        return draft
    }()
    @FocusState private var focused: Bool

    var body: some View {
        VStack {
            Spacer(minLength: 0)
            StudioComposer(
                mode: .createImage, draft: $draft, isRunning: false, queuedCount: 0, readiness: .ready,
                modelInventory: [], promptFocus: $focused, onRun: {}, onStop: {}, onShowModels: {},
                onRecallPrompt: { draft.prompt = $0 }
            )
        }
    }
}

/// A task inspector over the draft the sessions hold, so Save and Restore act on it.
private struct InspectorHost: View {
    let task: StudioTask
    @Environment(\.studioTaskSessions) private var sessions

    var body: some View {
        if let sessions, let draft = sessions.taskDraft(for: task) {
            StudioTaskInspector(
                task: task,
                draft: Binding(get: { draft }, set: { sessions.setTaskDraft($0, for: task) }),
                modelInventory: [], readiness: .ready, onShowModels: {}, onClose: {}
            )
        }
    }
}
