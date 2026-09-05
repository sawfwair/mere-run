import AppKit
import StudioKit
import SwiftUI

/// The current task's complete, editable command. The composer and this form share execution state.
struct StudioTaskCommandView: View {
    @EnvironmentObject private var controller: MereRunController
    let template: CommandTemplate
    let seed: CommandDraft
    @Binding var form: StudioConsoleDraft
    let onRun: () -> Void
    let onClose: () -> Void
    var canRun = true

    private var launch: StudioConsoleRun? { StudioConsoleRun(template: template, draft: form, seed: seed) }
    private var preview: String { controller.commandPreview(arguments: launch?.arguments ?? [], masksSecrets: true) }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Command", systemImage: "terminal").font(.headline)
                Spacer()
                Button(action: onClose) { Image(systemName: "xmark").frame(width: 28, height: 28) }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Close Command")
            }
            .padding(16)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(template.title).font(.headline)
                    if let capability = template.id.capability {
                        ForEach(StudioConsoleCommand.groups(for: capability)) { group in
                            VStack(alignment: .leading, spacing: 8) {
                                MereEyebrow(group.title)
                                ContractForm(fields: group.fields,
                                             dependencies: StudioConsoleCommand.dependencies(for: capability, draft: form),
                                             draft: $form, labelStyle: .flag) { _ in EmptyView() }
                            }
                        }
                    }
                    TextField("Extra arguments", text: $form.extraArguments, axis: .vertical)
                        .font(.system(.body, design: .monospaced))
                        .mereField()
                }
                .padding(16)
            }
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    MereEyebrow("Will run")
                    Spacer()
                    Button("Copy", systemImage: "doc.on.doc") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(preview, forType: .string)
                    }.buttonStyle(.plain)
                }
                ScrollView(.horizontal) {
                    Text(StudioCommandPreviewFormatter.wrapped(preview, width: 48))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }.frame(maxHeight: 110)
                if let message = launch?.validationMessage {
                    Text(message).font(.callout).foregroundStyle(MereRunTheme.red)
                }
                Button("Run command", systemImage: "play.fill", action: onRun)
                    .buttonStyle(.merePrimary)
                    .disabled(!canRun || launch == nil || launch?.validationMessage != nil)
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
            .padding(16)
        }
        .background(MereRunTheme.background)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Editable task command")
    }
}

private struct StudioTaskCommandRegistration: ViewModifier {
    @Environment(\.studioTaskSessions) private var sessions
    @Environment(\.studioTaskScope) private var scope
    let templateID: CommandTemplateID
    let draft: CommandDraft

    private var hasOverrides: Bool {
        let state = sessions?.value(for: templateID.studioTask.rawValue + ".commandOverride",
                                   default: Optional<StudioTaskCommandState>.none)
        let source = CommandCatalog.template(id: templateID)?.arguments(from: draft) ?? []
        guard let state, state.templateID == templateID, let capability = templateID.capability else { return false }
        let baseline = StudioConsoleCommand.seed(capability: capability, arguments: source)
        return StudioConsoleCommand.arguments(for: capability, draft: state.resolved(source: source))
            != StudioConsoleCommand.arguments(for: capability, draft: baseline)
    }

    func body(content: Content) -> some View {
        content.safeAreaInset(edge: .bottom, spacing: 0) {
            if hasOverrides {
                HStack {
                    Label("Using edited Command settings", systemImage: "terminal")
                    Spacer(minLength: 8)
                    Button("Use form settings") {
                        sessions?.set(Optional<StudioTaskCommandState>.none,
                                      for: templateID.studioTask.rawValue + ".commandOverride")
                    }.buttonStyle(.plain).foregroundStyle(MereRunTheme.accent)
                }
                .font(.callout).padding(12)
                .background(MereRunTheme.surfaceRaised)
            }
        }
        .onChange(of: draft, initial: true) { _, value in
            sessions?.set(value, for: scope + ".commandDraft")
            sessions?.set(templateID, for: scope + ".commandTemplate")
        }
        .onChange(of: templateID) { _, value in
            sessions?.set(draft, for: scope + ".commandDraft")
            sessions?.set(value, for: scope + ".commandTemplate")
        }
    }
}

extension View {
    func studioTaskCommand(_ templateID: CommandTemplateID, draft: CommandDraft) -> some View {
        modifier(StudioTaskCommandRegistration(templateID: templateID, draft: draft))
    }
}
