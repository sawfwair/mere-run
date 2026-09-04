import AppKit
import StudioKit
import SwiftUI

/// Decodes `mere.run model location list --json`. Keys mirror the CLI's snake_case contract.
struct StudioModelLocationSnapshot: Decodable, Equatable {
    struct Root: Decodable, Equatable, Identifiable {
        let path: String
        let available: Bool

        var id: String { path }
    }

    struct Binding: Decodable, Equatable, Identifiable {
        let modelID: String
        let path: String
        let available: Bool
        let usageTermsAcknowledged: Bool

        var id: String { "\(modelID):\(path)" }

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case path
            case available
            case usageTermsAcknowledged = "usage_terms_acknowledged"
        }
    }

    let primaryStore: Root
    let searchRoots: [Root]
    let bindings: [Binding]

    enum CodingKeys: String, CodingKey {
        case primaryStore = "primary_store"
        case searchRoots = "search_roots"
        case bindings
    }

    static func decode(_ stdout: String) -> StudioModelLocationSnapshot? {
        guard let data = stdout.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(StudioModelLocationSnapshot.self, from: data)
    }
}

/// A pending destructive edit, held until the operator confirms it.
struct StudioModelLocationConfirmation: Identifiable {
    enum Kind {
        case removeRoot(String)
        case unbind(modelID: String, path: String)
    }

    let kind: Kind

    var id: String {
        switch kind {
        case .removeRoot(let path): "remove:\(path)"
        case .unbind(let modelID, let path): "unbind:\(modelID):\(path)"
        }
    }

    var title: String {
        switch kind {
        case .removeRoot: "Stop searching this root?"
        case .unbind: "Remove this binding?"
        }
    }

    var message: String {
        switch kind {
        case .removeRoot(let path):
            return "mere.run will stop searching \(path). No files are deleted."
        case .unbind(let modelID, _):
            return "mere.run will stop resolving \(modelID) from its bound directory. No files are deleted."
        }
    }
}

/// The first-class store editor: register read-only roots and explicit per-model bindings
/// so a model kept on an external volume is usable without leaving the app.
struct StudioModelLocationsView: View {
    @EnvironmentObject private var controller: MereRunController

    let onLocationsChanged: () -> Void

    @State private var snapshot: StudioModelLocationSnapshot?
    @State private var statusMessage = "Loading model locations"
    @State private var busy = false
    @State private var bindModelID = ""
    @State private var bindPath = ""
    @State private var bindAcceptModelLicense = false
    @State private var confirmation: StudioModelLocationConfirmation?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().overlay(MereRunTheme.border.opacity(0.6))
            ScrollView {
                VStack(alignment: .leading, spacing: MereRunTheme.Spacing.lg) {
                    primaryStoreCard
                    searchRootsCard
                    bindingsCard
                }
                .padding(MereRunTheme.Spacing.xl)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(MereRunTheme.background)
        .foregroundStyle(MereRunTheme.textPrimary)
        .task { await refresh() }
        .alert(item: $confirmation) { pending in
            Alert(
                title: Text(pending.title),
                message: Text(pending.message),
                primaryButton: .destructive(Text("Remove")) {
                    Task { await apply(pending) }
                },
                secondaryButton: .cancel()
            )
        }
    }

    private var header: some View {
        HStack(spacing: MereRunTheme.Spacing.md) {
            Text(statusMessage)
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
                .lineLimit(1)
            Spacer()
            if busy { ProgressView().controlSize(.small) }
            Button {
                Task { await refresh() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.mereSecondary)
            .disabled(busy)
        }
        .padding(.horizontal, MereRunTheme.Spacing.xl)
        .padding(.vertical, MereRunTheme.Spacing.sm)
    }

    private var primaryStoreCard: some View {
        card {
            Text("Writable store")
                .font(MereRunTheme.sectionFont)
            Text("Downloads and manifests are written here. It cannot be removed.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)
            if let store = snapshot?.primaryStore {
                HStack {
                    availabilityDot(store.available)
                    Text(store.path)
                        .font(MereRunTheme.monoFont)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button("Reveal") { reveal(store.path) }
                        .buttonStyle(.mereSecondary)
                        .accessibilityLabel("Reveal the writable model store in Finder")
                }
            }
        }
    }

    private var searchRootsCard: some View {
        card {
            HStack {
                Text("Read-only search roots")
                    .font(MereRunTheme.sectionFont)
                Spacer()
                Button {
                    addSearchRoot()
                } label: {
                    Label("Add root", systemImage: "externaldrive.badge.plus")
                }
                .buttonStyle(.mereSecondary)
                .disabled(busy)
            }
            Text(
                "Each root holds directories named for canonical model ids. mere.run reads from them "
                    + "and never writes to them, so an external volume stays untouched."
            )
            .font(MereRunTheme.captionFont)
            .foregroundStyle(MereRunTheme.textMuted)
            .fixedSize(horizontal: false, vertical: true)

            let roots = snapshot?.searchRoots ?? []
            if roots.isEmpty {
                Text("No search roots registered.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            } else {
                ForEach(roots) { root in
                    HStack {
                        availabilityDot(root.available)
                        Text(root.path)
                            .font(MereRunTheme.monoFont)
                            .textSelection(.enabled)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        if !root.available {
                            Text("Offline")
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.yellow)
                        }
                        Button("Reveal") { reveal(root.path) }
                            .buttonStyle(.mereSecondary)
                            .disabled(!root.available)
                            .accessibilityLabel("Reveal search root \(root.path) in Finder")
                        Button("Remove") {
                            confirmation = StudioModelLocationConfirmation(kind: .removeRoot(root.path))
                        }
                        .buttonStyle(.mereSecondary)
                        .disabled(busy)
                        .accessibilityLabel("Remove search root \(root.path)")
                    }
                }
            }
        }
    }

    private var bindingsCard: some View {
        card {
            Text("Explicit bindings")
                .font(MereRunTheme.sectionFont)
            Text("Point one canonical model id at a directory that is not named for it.")
                .font(MereRunTheme.captionFont)
                .foregroundStyle(MereRunTheme.textMuted)

            HStack(spacing: 8) {
                TextField("Canonical model id", text: $bindModelID)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                TextField("Model directory", text: $bindPath)
                    .mereField(cornerRadius: MereRunTheme.Radius.sm)
                Button("Choose…") {
                    if let url = StudioSpecialistFiles.chooseDirectory(title: "Model directory") {
                        bindPath = url.path
                    }
                }
                .buttonStyle(.bordered)
                Button("Bind") {
                    Task { await bind() }
                }
                .buttonStyle(.merePrimary)
                .disabled(busy || bindModelID.isBlank || bindPath.isBlank)
            }

            Toggle(
                "I reviewed and accept this model's listed third-party terms",
                isOn: $bindAcceptModelLicense
            )
            .font(MereRunTheme.captionFont)
            .disabled(busy)

            let bindings = snapshot?.bindings ?? []
            if bindings.isEmpty {
                Text("No explicit bindings.")
                    .font(MereRunTheme.captionFont)
                    .foregroundStyle(MereRunTheme.textMuted)
            } else {
                ForEach(bindings) { binding in
                    HStack {
                        availabilityDot(binding.available)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(binding.modelID)
                                .font(MereRunTheme.bodyFont)
                            Text(binding.path)
                                .font(MereRunTheme.monoFont)
                                .foregroundStyle(MereRunTheme.textMuted)
                                .textSelection(.enabled)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        Spacer()
                        if !binding.usageTermsAcknowledged {
                            Label("Terms not accepted", systemImage: "exclamationmark.triangle.fill")
                                .font(MereRunTheme.captionFont)
                                .foregroundStyle(MereRunTheme.yellow)
                        }
                        Button("Reveal") { reveal(binding.path) }
                            .buttonStyle(.mereSecondary)
                            .disabled(!binding.available)
                            .accessibilityLabel("Reveal \(binding.modelID) directory in Finder")
                        Button("Unbind") {
                            confirmation = StudioModelLocationConfirmation(
                                kind: .unbind(modelID: binding.modelID, path: binding.path)
                            )
                        }
                        .buttonStyle(.mereSecondary)
                        .disabled(busy)
                        .accessibilityLabel("Unbind \(binding.modelID)")
                    }
                }
            }
        }
    }

    private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: MereRunTheme.Spacing.sm, content: content)
            .padding(MereRunTheme.Spacing.md)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                    .fill(MereRunTheme.surface.opacity(0.72))
                    .overlay {
                        RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                            .strokeBorder(MereRunTheme.border.opacity(0.55), lineWidth: 1)
                    }
            }
    }

    /// Availability is otherwise carried by colour alone, so the dot needs a spoken label.
    private func availabilityDot(_ available: Bool) -> some View {
        Circle()
            .fill(available ? MereRunTheme.green : MereRunTheme.textMuted)
            .frame(width: 7, height: 7)
            .accessibilityElement()
            .accessibilityLabel(available ? "Available" : "Offline")
    }

    private func reveal(_ path: String) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    private func refresh() async {
        busy = true
        defer { busy = false }
        let result = await controller.utilityCommandResult(
            args: ["model", "location", "list", "--json"],
            masksSecrets: false
        )
        guard result.exitCode == 0, let decoded = StudioModelLocationSnapshot.decode(result.stdout) else {
            statusMessage = "Could not read model locations."
            return
        }
        snapshot = decoded
        let rootCount = decoded.searchRoots.count
        let bindingCount = decoded.bindings.count
        statusMessage = "\(rootCount) search root\(rootCount == 1 ? "" : "s")"
            + " · \(bindingCount) binding\(bindingCount == 1 ? "" : "s")"
    }

    private func addSearchRoot() {
        guard let url = StudioSpecialistFiles.chooseDirectory(title: "Read-only model search root") else {
            return
        }
        Task { await runLocationCommand(["model", "location", "add", url.path], success: "Added \(url.path)") }
    }

    private func bind() async {
        let modelID = bindModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let path = bindPath.trimmingCharacters(in: .whitespacesAndNewlines)
        var args = ["model", "location", "bind", modelID, path]
        if bindAcceptModelLicense { args.append("--accept-model-license") }
        let succeeded = await runLocationCommand(
            args,
            success: "Bound \(modelID)"
        )
        if succeeded {
            bindModelID = ""
            bindPath = ""
            bindAcceptModelLicense = false
        }
    }

    private func apply(_ pending: StudioModelLocationConfirmation) async {
        switch pending.kind {
        case .removeRoot(let path):
            await runLocationCommand(
                ["model", "location", "remove", path],
                success: "Removed \(path)"
            )
        case .unbind(let modelID, let path):
            await runLocationCommand(
                ["model", "location", "unbind", modelID, path],
                success: "Unbound \(modelID)"
            )
        }
    }

    @discardableResult
    private func runLocationCommand(_ args: [String], success: String) async -> Bool {
        busy = true
        let result = await controller.utilityCommandResult(args: args, masksSecrets: false)
        busy = false
        if result.exitCode == 0 {
            statusMessage = success
            await refresh()
            onLocationsChanged()
            return true
        } else {
            statusMessage = result.outputText.isEmpty
                ? "The location command failed."
                : result.outputText
            return false
        }
    }
}
