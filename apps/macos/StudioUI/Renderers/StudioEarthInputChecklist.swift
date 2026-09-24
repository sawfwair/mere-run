import StudioKit
import SwiftUI

/// The Earth tasks' input column: the attached bundle's name, then one row per tensor its command
/// requires, ticked when the file's header carries it with the tensor's dtype and shape, and the
/// command's own refusal in words when one is missing — so a malformed bundle is caught before
/// the run rather than by a validation error after it. With no file attached, the same rows name
/// what the bundle must hold.
struct StudioEarthInputChecklist: View {
    let requirement: StudioEarthInputRequirement
    let url: URL?

    /// The attached file's header, read once per file. `.unreadable` when the file is not a
    /// safetensors bundle.
    @State private var header: Loaded = .none

    private enum Loaded: Equatable {
        case none
        case header(StudioSafetensorsHeader)
        case unreadable
    }

    private var check: StudioEarthInputCheck? {
        if case .header(let header) = header { return requirement.check(header) }
        return nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            heading
            hairline
            if let check {
                ForEach(check.required) { row in checklistRow(row, required: true) }
                if !check.oneOf.isEmpty {
                    groupTitle("At least one of")
                    ForEach(check.oneOf) { row in checklistRow(row, required: false) }
                }
                footer(check)
            } else {
                ForEach(requirement.required, id: \.self) { name in hintRow(name) }
                if !requirement.oneOf.isEmpty {
                    groupTitle("At least one of")
                    ForEach(requirement.oneOf) { option in hintRow(option.title) }
                }
                if header == .unreadable {
                    note("Not a safetensors file.", tone: MereRunTheme.textPrimary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .task(id: fileIdentity) { await load() }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(url == nil ? "Required tensors" : "Attached bundle")
    }

    /// The file's path with its modification date, so a bundle written over in place reloads.
    private var fileIdentity: String {
        guard let url else { return "" }
        let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        return "\(url.path)#\(modified.timeIntervalSinceReferenceDate)"
    }

    // MARK: Rows

    private var heading: some View {
        HStack(spacing: 10) {
            Image(systemName: "square.stack.3d.up")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(MereRunTheme.accent)
            VStack(alignment: .leading, spacing: 2) {
                Text(url?.lastPathComponent ?? "Tile bundle")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(MereRunTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(subtitle)
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    private var subtitle: String {
        switch header {
        case .header(let header):
            return StudioTensorHeader.safetensors(header).summary
        case .unreadable, .none:
            return requirement.hint
        }
    }

    /// The tensor's name with its shape at the trailing edge; in a column too narrow for both on
    /// one line, the shape moves under the name rather than truncating either. Half a pair is
    /// marked as such, with the half the command will ask for.
    private func checklistRow(_ row: StudioEarthInputCheck.Row, required: Bool) -> some View {
        let name = Text(row.title)
            .font(.system(size: 12.5, weight: .medium, design: .monospaced))
            .foregroundStyle(MereRunTheme.textPrimary)
        let detail = Text(rowDetail(row, required: required))
            .font(.system(size: 11.5, weight: .medium, design: row.detail == nil ? .default : .monospaced))
            .foregroundStyle(row.isPresent ? MereRunTheme.textMuted : MereRunTheme.textSecondary)
        return HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: row.isPresent ? "checkmark.circle.fill" : (row.isIncomplete ? "exclamationmark.circle" : "circle.dashed"))
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(row.isPresent ? MereRunTheme.accent : MereRunTheme.textMuted)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    name.lineLimit(1)
                    Spacer(minLength: 8)
                    detail.lineLimit(1)
                }
                VStack(alignment: .leading, spacing: 3) {
                    name
                    detail
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .overlay(alignment: .bottom) { hairline }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(row.title), \(rowDetail(row, required: required))")
    }

    /// "F32 [1, 4, 2] · F32 [1, 4]", "Needs S1_DESC_DOY", "Missing", or "Not present".
    private func rowDetail(_ row: StudioEarthInputCheck.Row, required: Bool) -> String {
        if let detail = row.detail { return detail }
        if row.isIncomplete { return "Needs \(row.missing.joined(separator: " and "))" }
        return required ? "Missing" : "Not present"
    }

    private func hintRow(_ title: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "circle.dashed")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(MereRunTheme.textMuted)
            Text(title)
                .font(.system(size: 12.5, weight: .medium, design: .monospaced))
                .foregroundStyle(MereRunTheme.textSecondary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .overlay(alignment: .bottom) { hairline }
        .accessibilityElement(children: .combine)
    }

    private func groupTitle(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(MereRunTheme.textMuted)
            .textCase(.uppercase)
            .padding(.horizontal, 14)
            .padding(.top, 10)
            .padding(.bottom, 4)
            .accessibilityAddTraits(.isHeader)
    }

    @ViewBuilder
    private func footer(_ check: StudioEarthInputCheck) -> some View {
        if let message = check.message {
            note(message, tone: MereRunTheme.textPrimary)
        } else {
            note("All required tensors present.", tone: MereRunTheme.textMuted)
        }
    }

    private func note(_ text: String, tone: Color) -> some View {
        Text(text)
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(tone)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
    }

    private var hairline: some View {
        Rectangle()
            .fill(MereRunTheme.border.opacity(0.27))
            .frame(height: 1)
    }

    // MARK: Loading

    private func load() async {
        guard let url else {
            header = .none
            return
        }
        let loaded = await Task.detached(priority: .userInitiated) {
            StudioSafetensorsHeader.load(from: url)
        }.value
        guard !Task.isCancelled else { return }
        header = loaded.map(Loaded.header) ?? .unreadable
    }
}
