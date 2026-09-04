import AppKit
import SwiftUI

/// The Command view column: the raw form of the current prompt task, rendered from the same
/// draft the composer and inspector edit — every option the contract declares, grouped, with
/// the value the argv carries — and, at the bottom, exactly what Run will launch.
///
/// Values are shown read-only until every task renders its own editable contract form; editing
/// stays with the chips and the inspector, which write the same draft this column reads.
struct StudioCommandView: View {
    let mode: StudioMode
    let template: CommandTemplate
    let draft: CommandDraft
    /// The masked, shell-quoted command line the run would launch.
    let displayCommand: String
    let canRun: Bool
    let onRun: () -> Void
    let onClose: () -> Void

    @State private var copied = false

    static let width = StudioLayoutPolicy.commandWidth

    private var groups: [StudioCommandRowGroupRows] {
        StudioCommandRows.groups(template: template, draft: draft)
    }

    private var commandPathTitle: String {
        (template.id.capabilityID ?? template.id.rawValue).replacingOccurrences(of: ".", with: " ")
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                VStack(spacing: 0) {
                    ForEach(groups) { group in
                        groupView(group)
                    }
                }
            }
            footer
        }
        .frame(width: Self.width)
        .background(MereRunTheme.background)
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(MereRunTheme.border.opacity(0.53))
                .frame(width: 1)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Command view")
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text("Command · \(commandPathTitle)")
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(MereRunTheme.textPrimary)
                .lineLimit(1)
            Spacer(minLength: 0)
            Text("Same draft as the inspector")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 22, height: 22)
            }
            .buttonStyle(.mereIcon(tint: MereRunTheme.textMuted))
            .help("Hide Command view (⌥⌘C)")
            .accessibilityLabel("Hide Command view")
        }
        .padding(.horizontal, 16)
        .frame(height: 44)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
        }
    }

    private func groupView(_ group: StudioCommandRowGroupRows) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            MereEyebrow(group.group.title)
            ForEach(group.rows) { row in
                rowView(row)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(alignment: .bottom) {
            Rectangle().fill(MereRunTheme.border.opacity(0.4)).frame(height: 1)
        }
    }

    private func rowView(_ row: StudioCommandRow) -> some View {
        HStack(alignment: .center, spacing: 10) {
            Text(row.flag)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(width: 150, alignment: .leading)
                .help(row.label)
            switch row.value {
            case .toggle(let on):
                StudioCommandToggle(isOn: on)
                    .accessibilityLabel(row.label)
                    .accessibilityValue(on ? "On" : "Off")
                Spacer(minLength: 0)
            case .text(let text):
                valueField(text, placeholder: false)
            case .positional(let text):
                valueField(text, placeholder: false)
            }
        }
        .frame(minHeight: 28)
        .accessibilityElement(children: .combine)
    }

    private func valueField(_ text: String, placeholder: Bool) -> some View {
        Text(text)
            .font(.system(size: 12, design: .monospaced))
            .foregroundStyle(text.isEmpty ? MereRunTheme.textMuted : MereRunTheme.textPrimary)
            .lineLimit(2)
            .truncationMode(text.contains("/") ? .middle : .tail)
            .textSelection(.enabled)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .frame(maxWidth: .infinity, minHeight: 26, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                    .fill(MereRunTheme.surface)
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.base)
                            .strokeBorder(MereRunTheme.border.opacity(0.8), lineWidth: 1)
                    }
            }
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
                            .font(.system(size: 11, weight: .medium))
                        Text(copied ? "Copied" : "Copy")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(MereRunTheme.textMuted)
                }
                .buttonStyle(.plain)
                .help("Copy the command")
                .accessibilityLabel("Copy the command")
            }

            Text(StudioCommandPreviewFormatter.wrapped(displayCommand, width: 56))
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

            HStack(spacing: 8) {
                Spacer()
                Button("Open in Terminal") { openInTerminal() }
                    .buttonStyle(.mereSecondary)
                    .help("Copies the command and opens Terminal")
                Button("Run", action: onRun)
                    .buttonStyle(.merePrimary)
                    .disabled(!canRun)
                    .help(canRun ? "Run this command (⌘↩)" : "The task is not ready to run")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .overlay(alignment: .top) {
            Rectangle().fill(MereRunTheme.border.opacity(0.53)).frame(height: 1)
        }
    }

    private func copyCommand() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(displayCommand, forType: .string)
        copied = true
        Task {
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            copied = false
        }
    }

    /// Puts the command on the clipboard and brings Terminal forward; the app never scripts
    /// another application, so the user pastes and runs it themselves.
    private func openInTerminal() {
        copyCommand()
        guard let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Terminal") else { return }
        NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
    }
}

/// A 28×16 switch drawn to the board's spec; state only, the draft is edited elsewhere.
struct StudioCommandToggle: View {
    let isOn: Bool

    var body: some View {
        ZStack(alignment: isOn ? .trailing : .leading) {
            Capsule()
                .fill(isOn ? MereRunTheme.accent : MereRunTheme.surfaceRaised)
            Circle()
                .fill(isOn ? MereRunTheme.onAccent : MereRunTheme.surface)
                .frame(width: 12, height: 12)
                .padding(2)
        }
        .frame(width: 28, height: 16)
        .overlay {
            Capsule().strokeBorder(MereRunTheme.border.opacity(isOn ? 0 : 0.8), lineWidth: 1)
        }
    }
}
