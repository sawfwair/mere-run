import Foundation
import StudioKit
import SwiftUI

struct StudioLayaDecisionView: View {
    @EnvironmentObject private var controller: MereRunController
    @EnvironmentObject private var library: StudioLibraryStore
    @StudioStoredValue("Laya.requestID") private var requestID: UUID? = nil
    @StudioStoredValue("Laya.input") private var input = ""
    @StudioStoredValue("Laya.model") private var model = "text-decide-laya"
    @StudioStoredValue("Laya.preflight") private var preflight = false
    @StudioStoredValue("Laya.output") private var output = ""

    private var draft: CommandDraft {
        var value = CommandDraft()
        value.inputPath = input
        value.model = model
        value.outputPath = output
        value.preflight = preflight
        value.force = true
        return value
    }

    var body: some View {
        StudioAnalysisLayout {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Decisions").font(MereRunTheme.sectionFont)
                    Text("Ask choice, score, and true-or-false questions about the same text. Each answer includes a probability distribution.")
                        .foregroundStyle(.secondary)
                    Picker("Model", selection: $model) {
                        Text("Laya · English").tag("text-decide-laya")
                        Text("Laya · Multilingual").tag("text-decide-laya-multilingual")
                        Text("Laya · Typed decisions").tag("text-decide-laya-typed-decisions")
                    }
                    StudioPathField(label: "JSON request", placeholder: "/path/to/request.json", path: $input, picksDirectory: false)
                    Text("The request contains state text and an ordered questions array. Install the selected model from Models before running.")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("Inspect token budgets", isOn: $preflight)
                    Button(preflight ? "Inspect request" : "Evaluate questions") {
                        output = StudioSpecialistFiles.timestampedDirectory(component: "decisions")
                            .appending(path: preflight ? "preflight.json" : "decisions.json").path
                        requestID = StudioSpecialistRunner.submit(
                            templateID: .textDecide, mode: .chat, draft: draft, controller: controller, library: library)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(18)
            }
        } result: {
            StudioSpecialistResultView(requestID: requestID)
                .padding(18)
        }
        .background(MereRunTheme.background)
        .studioTaskCommand(.textDecide, draft: draft)
    }
}
