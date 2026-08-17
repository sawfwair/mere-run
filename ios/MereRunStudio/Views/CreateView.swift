import SwiftUI
import MereRunRelayKit

/// Prompt-first creation. The form is generated from the same node catalog
/// the CLI and Graph Studio use; the model list comes from what the paired
/// fleet actually has installed. Asset inputs (photos) are a follow-up.
struct CreateView: View {
    private static let kinds = ["image.generate", "video.generate"]
    private static let modelPrefixes = ["image.generate": "image-", "video.generate": "video-"]

    @EnvironmentObject private var relay: RelayStore
    @State private var kind = "image.generate"
    @State private var prompt = ""
    @State private var model = ""
    @State private var integerFields: [String: String] = [:]
    @State private var submitting = false
    @State private var errorMessage: String?
    @State private var submittedJobID: String?

    private var entry: WorkflowNodeCatalogEntry? {
        WorkflowNodeRegistry.entry(for: kind)
    }

    private var installedModels: [String] {
        let prefix = Self.modelPrefixes[kind] ?? ""
        return (relay.workerProbe?.installedModelIDs ?? []).filter { $0.hasPrefix(prefix) }
    }

    private var optionalIntegerInputs: [WorkflowNodeField] {
        entry?.inputs.filter { $0.type == .integer && $0.required != true } ?? []
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Create", selection: $kind) {
                        ForEach(Self.kinds, id: \.self) { kind in
                            Text(WorkflowNodeRegistry.entry(for: kind)?.title ?? kind).tag(kind)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Section("Prompt") {
                    TextField("What should your fleet make?", text: $prompt, axis: .vertical)
                        .lineLimit(3...6)
                }

                Section("Model") {
                    if installedModels.isEmpty {
                        Text("No installed \(Self.modelPrefixes[kind] ?? "")models reported by the fleet.")
                            .font(.footnote)
                            .foregroundStyle(MereTheme.textMuted)
                    } else {
                        Picker("Model", selection: $model) {
                            ForEach(installedModels, id: \.self) { id in
                                Text(id).tag(id)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                }

                if !optionalIntegerInputs.isEmpty {
                    Section("Options") {
                        ForEach(optionalIntegerInputs, id: \.name) { field in
                            LabeledContent(field.name.replacingOccurrences(of: "_", with: " ")) {
                                TextField(
                                    "auto",
                                    text: Binding(
                                        get: { integerFields[field.name] ?? "" },
                                        set: { integerFields[field.name] = $0 }
                                    )
                                )
                                .keyboardType(.numberPad)
                                .multilineTextAlignment(.trailing)
                                .frame(maxWidth: 120)
                            }
                        }
                    }
                }

                if let errorMessage {
                    Section {
                        Text(errorMessage)
                            .font(.footnote)
                            .foregroundStyle(MereTheme.failure)
                    }
                }

                Section {
                    Button {
                        Task { await submit() }
                    } label: {
                        if submitting {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Text("Run on your fleet").frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(prompt.isEmpty || model.isEmpty || submitting)
                }
            }
            .navigationTitle("Create")
            .navigationDestination(item: $submittedJobID) { jobID in
                RunDetailView(jobID: jobID)
            }
            .task {
                await relay.refreshWorkerProbe()
                if model.isEmpty { model = installedModels.first ?? "" }
            }
            .onChange(of: kind) {
                model = installedModels.first ?? ""
            }
        }
    }

    private func submit() async {
        submitting = true
        defer { submitting = false }
        var arguments: [String: WorkflowValue] = [
            "prompt": .string(prompt),
            "model": .string(model)
        ]
        for (name, raw) in integerFields {
            if let value = Int64(raw.trimmingCharacters(in: .whitespaces)) {
                arguments[name] = .integer(value)
            }
        }
        do {
            let job = try await relay.submit(kind: kind, arguments: arguments)
            errorMessage = nil
            submittedJobID = job.jobID
        } catch let error as RelayClientError {
            errorMessage = error.message
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
