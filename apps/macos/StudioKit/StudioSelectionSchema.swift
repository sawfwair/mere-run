import MereRunContract

extension StudioTaskSchema {
    /// Choices that change the operation or runtime, before its model is selected.
    package static func operationFields(
        for task: StudioTask, draft: StudioTaskDraft, source: StudioScopeSource
    ) -> [StudioContractField<StudioTaskDraft>] {
        let routing = source.capability(for: draft.templateID)?.routing
        let modelFlags = Set((routing?.modelFlags ?? []) + (routing?.families.compactMap(\.modelFlag) ?? []))
        return fields(for: task, draft: draft, source: source).filter {
            $0.overrideID == .variant || ($0.option.kind == .choice
                && routing?.routingFlags.contains($0.flag) == true && !modelFlags.contains($0.flag))
        }.map { field in
            guard field.overrideID != .variant,
                  let declared = source.capability(for: draft.templateID)?.options.first(where: { $0.flag == field.flag }) else { return field }
            // Operation choices must include other runtimes, even when the current model
            // narrows this option to one value. The model is selected after the operation.
            return StudioContractField(option: declared, bindings: field.bindings, overrideID: field.overrideID)
        }
    }

    /// The active runtime's managed-model argument. Folder-based and model-free commands
    /// have no managed-model menu; selector-routed commands can use a different flag per runtime.
    package static func modelPickerFlag(for draft: StudioTaskDraft, source: StudioScopeSource) -> String? {
        guard let scope = source.scope(for: draft) else { return nil }
        let flags: [String]
        if let routing = scope.capability.routing, routing.routesBySelectors {
            flags = scope.family?.modelFlag.map { [$0] } ?? routing.modelFlags
        } else {
            flags = ["--model"]
        }
        return flags.first { flag in
            scope.options.contains { $0.flag == flag && $0.kind == .string }
        }
    }
}

extension StudioDraft {
    /// An operation edit keeps compatible models and local paths. If the old managed model
    /// cannot run the new operation, return to that operation's default instead of displaying
    /// a model the CLI will ignore. An explicitly edited model is left for normal validation.
    package mutating func reconcileModelAfterOperationChange(
        from previous: StudioDraft, mode: StudioMode, source: StudioScopeSource
    ) {
        guard model == previous.model, !model.isBlank, !StudioTaskSchema.isLocalPath(model),
              let before = source.scope(mode: mode, draft: previous), let after = source.scope(mode: mode, draft: self),
              let routing = after.capability.routing else { return }
        let flags = routing.routingFlags.subtracting(routing.modelFlags)
        let changed = before.capability.id != after.capability.id
            || flags.contains { before.invocation.values[$0] != after.invocation.values[$0] }
        guard changed, let allowed = StudioModelScope(mode: mode, draft: self, source: source).runnableModels,
              !allowed.contains(model) else { return }
        model = ""
        modelSource = nil
    }
}
