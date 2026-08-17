import SwiftUI
import MereRunRelayKit

/// The paired fleet: summary, per-node status and inventory, recent activity.
struct FleetView: View {
    @EnvironmentObject private var relay: RelayStore
    @State private var snapshot: RelayFleetSnapshot?
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            List {
                if let summary = snapshot?.summary {
                    Section {
                        LabeledContent("Available", value: "\(summary.availableNodes)/\(summary.totalNodes) nodes")
                        LabeledContent("Queued", value: "\(summary.queueDepth)")
                        LabeledContent("Routable models", value: "\(summary.routableModels)")
                    }
                }
                Section("Online") {
                    ForEach(onlineNodes, id: \.deviceID) { node in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(node.deviceName)
                                    .foregroundStyle(MereTheme.textPrimary)
                                Spacer()
                                MereStatusLabel(text: node.status, color: statusColor(node.status))
                            }
                            Text(inventoryLine(node))
                                .font(.footnote)
                                .foregroundStyle(MereTheme.textMuted)
                                .lineLimit(2)
                        }
                    }
                }
                if !offlineNodes.isEmpty {
                    Section("Offline") {
                        ForEach(offlineNodes, id: \.deviceID) { node in
                            HStack {
                                Text(node.deviceName)
                                    .font(.footnote)
                                    .foregroundStyle(MereTheme.textMuted)
                                Spacer()
                                Text(node.status)
                                    .font(.footnote)
                                    .foregroundStyle(MereTheme.textMuted)
                            }
                        }
                    }
                }
                if let activity = snapshot?.activity, !activity.isEmpty {
                    Section("Recent activity") {
                        ForEach(activity.prefix(10), id: \.id) { entry in
                            HStack {
                                Text(entry.label)
                                    .font(.footnote)
                                    .foregroundStyle(MereTheme.textSecondary)
                                    .lineLimit(1)
                                Spacer()
                                MereStatusLabel(text: entry.status, color: statusColor(entry.status))
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
            }
            .scrollContentBackground(.hidden)
            .background(MereTheme.background.ignoresSafeArea())
            .navigationTitle("Fleet")
            .refreshable { await load() }
            .task { await load() }
        }
    }

    private var onlineNodes: [RelayFleetNode] {
        (snapshot?.nodes ?? []).filter { $0.status.lowercased() != "offline" }
            .sorted { $0.deviceName < $1.deviceName }
    }

    private var offlineNodes: [RelayFleetNode] {
        (snapshot?.nodes ?? []).filter { $0.status.lowercased() == "offline" }
            .sorted { $0.deviceName < $1.deviceName }
    }

    private func load() async {
        guard let client = relay.client else { return }
        do {
            snapshot = try await client.fleet()
            errorMessage = nil
        } catch let error as RelayClientError {
            errorMessage = error.message
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func inventoryLine(_ node: RelayFleetNode) -> String {
        let models = node.capabilities.graphWorker?.installedModelIDs
            ?? node.runtime?.installedModels
            ?? node.capabilities.models
        return models.isEmpty ? "No installed models reported" : models.joined(separator: ", ")
    }
}

func statusColor(_ status: String) -> Color {
    switch status.lowercased() {
    case "online", "available", "finished", "succeeded", "idle":
        MereTheme.success
    case "busy", "running", "assigned", "queued", "preflighting", "planned":
        MereTheme.caution
    case "failed", "error", "offline", "revoked", "cancelled":
        MereTheme.failure
    default:
        MereTheme.textMuted
    }
}
