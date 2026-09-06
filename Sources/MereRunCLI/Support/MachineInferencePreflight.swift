import Foundation
import MereRunRelayKit

enum MachineInferencePreflight {
    static func diagnostics(
        arguments: [String],
        host: MachineInferenceHostSnapshot = MachineInferenceCoordinator.currentHostSnapshot()
    ) -> [PreflightDiagnostic] {
        guard let request = CLIInferenceAdmissionClassifier.request(arguments: arguments) else { return [] }
        var diagnostics: [PreflightDiagnostic] = []
        let minimumDisk = MachineInferenceCoordinator.minimumDiskBytes(physicalMemoryBytes: host.physicalMemoryBytes)
        if let available = host.availableDiskBytes, available < minimumDisk {
            diagnostics.append(.init(
                id: "machine_insufficient_disk", severity: .blocker, title: "Insufficient disk headroom",
                message: MachineInferenceAdmissionError.insufficientDisk(
                    available: available, required: minimumDisk
                ).localizedDescription
            ))
        }
        let minimumMemory = request.resourceClass.minimumAvailableBytes
        if let available = host.availableMemoryBytes, available < minimumMemory {
            diagnostics.append(.init(
                id: "machine_insufficient_memory", severity: .blocker, title: "Insufficient memory headroom",
                message: MachineInferenceAdmissionError.insufficientMemory(
                    available: available, required: minimumMemory
                ).localizedDescription
            ))
        }
        if host.memoryPressure == .critical {
            diagnostics.append(.init(
                id: "machine_critical_memory_pressure", severity: .blocker, title: "Critical memory pressure",
                message: MachineInferenceAdmissionError.criticalMemoryPressure.localizedDescription
            ))
        }
        return diagnostics
    }
}
