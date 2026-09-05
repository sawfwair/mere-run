import AppKit
import MereRunContract
import StudioKit
import SwiftUI

/// The Command Console's window identity, shared by the scene that declares it and by every
/// surface that opens it: the toolbar, View ▸ Command Console, readiness "Details", Library
/// "Edit command", and the adapter fallbacks. There is no docked or detached variant any more.
package enum StudioConsoleWindow {
    package static let id = "console"
    package static let title = "Command Console"
}

/// The Command Console: every capability the CLI declares, as three panes that share the window's
/// width — the catalog, the capability's form, and the run's output.
///
/// The middle pane is `ContractForm` over `MereRunCapabilityCatalog`, so the console has no
/// per-command view of its own: a capability's options come from the contract, in the contract's
/// groups, with the control each option's `kind` and `range` call for, and the "Will run" block
/// shows the argv `StudioConsoleCommand` builds from the same values. A capability nobody has
/// designed a surface for is reachable here the day the contract declares it.
///
/// While this window is key the menu bar acts on the console's own Run/Stop and on the Studio
/// window's navigation.
package struct StudioConsoleView: View {
    package init() {}

    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @EnvironmentObject private var navigation: NavigationModel

    @State private var draft = StudioConsoleDraft()
    @State private var requestID: UUID?
    @StateObject private var jobs = StudioJobMonitor()

    private var job: Job? {
        _ = jobs.generation
        return requestID.flatMap(jobs.job(requestID:))
    }

    private func stop() { if let job { jobs.cancel(job) } }

    package var body: some View {
        HSplitView {
            StudioConsoleCatalog()
                .frame(minWidth: 220, idealWidth: 268, maxWidth: 360)

            StudioConsoleForm(draft: $draft, run: run, stop: stop, canStop: job?.state.isActive == true)
                .frame(minWidth: 420, idealWidth: 560, maxWidth: .infinity)
                .layoutPriority(1)

            Group {
                if let job { StudioConsoleLog(job: job) }
                else { ContentUnavailableView("Run output", systemImage: "terminal", description: Text("Your Console run will appear here.")) }
            }
                .frame(minWidth: 320, idealWidth: 440, maxWidth: .infinity)
        }
        .background(MereRunTheme.background.ignoresSafeArea())
        .foregroundStyle(MereRunTheme.textPrimary)
        .onAppear {
            jobs.attach(controller.jobs)
            library.observe(controller: controller)
            navigation.isConsoleOpen = true
            reseed()
        }
        .onDisappear { navigation.isConsoleOpen = false }
        // The console's values are read out of the template's own argv, so selecting a template,
        // syncing the composer into it, or opening a Library row all arrive the same way.
        .onChange(of: controller.selectedTemplate.id) { reseed() }
        .onChange(of: controller.draft) { reseed() }
        .onChange(of: controller.consoleSeedArguments) { reseed() }
        .focusedSceneValue(\.studioActions, sceneActions)
    }

    private var capability: MereRunCommandCapability? {
        controller.selectedTemplate.id.capability
    }

    private func reseed() {
        if let arguments = controller.consoleSeedArguments, let capability {
            draft = StudioConsoleCommand.seed(capability: capability, arguments: arguments)
        } else {
            draft = StudioConsoleCommand.seed(template: controller.selectedTemplate, draft: controller.draft)
        }
    }

    private var canRun: Bool {
        StudioConsoleRun(template: controller.selectedTemplate, draft: draft, seed: controller.draft)?
            .validationMessage == nil
    }

    /// Records the run in the Library — with the exact argv, so "Edit command" reopens this
    /// command and not the draft it was seeded from — and hands it to the inference lane.
    private func run() {
        let template = controller.selectedTemplate
        guard let launch = StudioConsoleRun(template: template, draft: draft, seed: controller.draft),
              launch.validationMessage == nil else {
            return
        }
        let arguments = launch.arguments
        let commandDraft = launch.commandDraft
        let request = StudioRunRequest(
            mode: template.libraryMode,
            templateID: template.id,
            template: template,
            draft: commandDraft,
            execution: StudioExecution(templateID: template.id, arguments: arguments)
        )
        let status: StudioLibraryStatus = controller.isRunning || controller.queuedRunCount > 0
            ? .queued
            : .running
        library.start(
            request: request,
            commandPreview: controller.commandPreview(arguments: arguments, masksSecrets: true),
            status: status,
            arguments: arguments
        )
        requestID = request.id
        controller.runConsole(
            template: template,
            draft: commandDraft,
            arguments: arguments,
            requestID: request.id
        )
    }

    private var sceneActions: StudioSceneActions {
        StudioSceneActions(
            destination: navigation.destination,
            showLibrary: Binding(
                get: { navigation.showLibrary },
                set: { navigation.showLibrary = $0 }
            ),
            canShowLibrary: navigation.destination.task.isPromptTask,
            showInspector: .constant(false),
            canShowInspector: false,
            showCommand: .constant(false),
            canShowCommand: false,
            open: { navigation.open(destination: $0) },
            openDomain: { navigation.open(domain: $0) },
            newChat: {},
            canNewChat: false,
            runComposer: run,
            canRun: canRun,
            stop: stop,
            canStop: job?.state.isActive == true,
            openConsole: {},
            showGuide: { navigation.showGuide = true },
            importReceipt: {}
        )
    }
}

// MARK: - The catalog

/// Every template in the catalog, grouped by category. Selecting one reseeds the form from that
/// template's own default command.
private struct StudioConsoleCatalog: View {
    @EnvironmentObject private var controller: MereRunController

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                StudioWordmark()
                Text("Every command, from the contract")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
            }
            .padding(.horizontal, 18)
            .padding(.top, 20)
            .padding(.bottom, 14)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(CommandCategory.allCases) { category in
                        let templates = CommandCatalog.templates(in: category)
                        if !templates.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                MereEyebrow(category.rawValue)
                                    .padding(.horizontal, 18)
                                    .padding(.bottom, 2)
                                ForEach(templates) { template in
                                    row(template)
                                }
                            }
                        }
                    }
                }
                .padding(.bottom, 18)
            }
        }
        .background(MereRunTheme.background)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Command catalog")
    }

    private func row(_ template: CommandTemplate) -> some View {
        let selected = template.id == controller.selectedTemplate.id
        return Button {
            controller.select(template)
        } label: {
            HStack(spacing: 10) {
                Image(systemName: template.systemImage)
                    .font(.callout.weight(.medium))
                    .frame(width: 16, height: 16)
                    .foregroundStyle(selected ? MereRunTheme.onAccent : MereRunTheme.textMuted)
                Text(template.title)
                    .font(.system(size: 13, weight: selected ? .semibold : .medium))
                    .foregroundStyle(selected ? MereRunTheme.onAccent : MereRunTheme.textPrimary)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .frame(height: 30)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.md)
                    .fill(selected ? MereRunTheme.accent : Color.clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.md))
            .padding(.horizontal, 10)
        }
        .buttonStyle(.plain)
        .help(template.subtitle)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

// MARK: - The form

/// The selected capability's options, grouped as the contract groups them, over the argv the
/// same values build.
private struct StudioConsoleForm: View {
    @EnvironmentObject private var controller: MereRunController
    @Binding var draft: StudioConsoleDraft
    let run: () -> Void
    let stop: () -> Void
    let canStop: Bool

    @State private var copied = false

    private var template: CommandTemplate { controller.selectedTemplate }
    private var capability: MereRunCommandCapability? { template.id.capability }

    private var launch: StudioConsoleRun? {
        StudioConsoleRun(template: template, draft: draft, seed: controller.draft)
    }

    private var displayCommand: String {
        controller.commandPreview(arguments: launch?.arguments ?? [], masksSecrets: true)
    }

    private var validationMessage: String? { launch?.validationMessage }

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                VStack(spacing: 0) {
                    if let url = template.externalURL {
                        externalSection(url)
                    } else if let capability {
                        ForEach(StudioConsoleCommand.groups(for: capability)) { group in
                            groupView(group, capability: capability)
                        }
                        extraArgumentsSection(lines: 1...4)
                    } else {
                        extraArgumentsSection(lines: 6...20)
                    }
                }
            }
            if template.externalURL == nil {
                footer
            }
        }
        .background(MereRunTheme.background)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Command form")
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: template.systemImage)
                .font(.body.weight(.semibold))
                .foregroundStyle(MereRunTheme.accent)
                .frame(width: 22, height: 22)
            VStack(alignment: .leading, spacing: 1) {
                Text(template.title)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(MereRunTheme.textPrimary)
                Text(template.subtitle)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            if let capability {
                Text(capability.command.joined(separator: " "))
                    .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                    .foregroundStyle(MereRunTheme.textSecondary)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 16)
        .frame(height: 52)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
        }
    }

    private func groupView(_ group: StudioConsoleGroup, capability: MereRunCommandCapability) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            MereEyebrow(group.title)
            ContractForm(
                fields: group.fields,
                dependencies: StudioConsoleCommand.dependencies(for: capability, draft: draft),
                draft: $draft,
                labelStyle: .flag
            ) { _ in
                // The console has no composite editors: it draws every option the contract
                // declares, one row per flag, and the designed surfaces own the rest.
                EmptyView()
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
        }
    }

    private func extraArgumentsSection(lines: ClosedRange<Int>) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            MereEyebrow(capability == nil ? "Arguments" : "Extra arguments")
            StudioInspectorTextField(
                placeholder: "--flag value",
                text: $draft.extraArguments,
                lines: lines,
                isMonospaced: true
            )
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// A template that hands off to another product: there is no command to build, only a place
    /// to go.
    private func externalSection(_ url: URL) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            MereEyebrow("Destination")
            Text(url.absoluteString)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(MereRunTheme.textPrimary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("Open") { NSWorkspace.shared.open(url) }
                .buttonStyle(.merePrimary)
                .keyboardShortcut(.return, modifiers: .command)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                MereEyebrow("Will run")
                Spacer()
                Button {
                    copyCommand()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: copied ? "checkmark" : "doc.on.doc")
                            .font(.caption.weight(.medium))
                        Text(copied ? "Copied" : "Copy")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(MereRunTheme.textMuted)
                }
                .buttonStyle(.plain)
                .help("Copy the command")
                .accessibilityLabel("Copy the command")
            }

            Text(StudioCommandPreviewFormatter.wrapped(displayCommand, width: 64))
                .font(.system(size: 11.5, design: .monospaced))
                .lineSpacing(3)
                .foregroundStyle(MereRunTheme.textPrimary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                        .fill(MereRunTheme.surfaceRaised)
                }
                .accessibilityLabel("Command preview")
                .accessibilityValue(displayCommand)

            if let validationMessage {
                MereBanner(severity: .warning, text: validationMessage)
            }

            HStack(spacing: 8) {
                Spacer()
                Button("Stop", action: stop)
                    .buttonStyle(.mereSecondary)
                    .disabled(!canStop)
                Button(controller.isRunning ? "Queue" : "Run", action: run)
                    .buttonStyle(.merePrimary)
                    .disabled(validationMessage != nil)
                    .keyboardShortcut(.return, modifiers: .command)
                    .help(validationMessage ?? "Run this command (⌘↩)")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .overlay(alignment: .top) {
            Rectangle().fill(MereRunTheme.border.opacity(0.53)).frame(height: 1)
        }
    }

    private func copyCommand() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(displayCommand, forType: .string)
        copied = true
        Task {
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            copied = false
        }
    }
}
