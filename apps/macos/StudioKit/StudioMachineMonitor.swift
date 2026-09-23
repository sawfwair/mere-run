import Darwin
import Foundation

/// This Mac's load, sampled every two seconds for the life of the app: CPU busy share, memory in
/// use, and thermal state, with a short history for the menu bar's sparklines. It reads host
/// statistics, so it reports whether or not a server is running.
@MainActor
package final class StudioMachineMonitor: ObservableObject {
    /// One reading of the machine.
    package struct Sample: Equatable {
        /// Share of CPU time spent busy since the previous reading, 0...1.
        package var cpu: Double
        /// Memory in use the way Activity Monitor counts it: app memory, wired, and compressed.
        package var memoryUsedBytes: UInt64
        package var memoryTotalBytes: UInt64
        package var thermalState: ProcessInfo.ThermalState

        package init(cpu: Double, memoryUsedBytes: UInt64, memoryTotalBytes: UInt64, thermalState: ProcessInfo.ThermalState) {
            self.cpu = cpu
            self.memoryUsedBytes = memoryUsedBytes
            self.memoryTotalBytes = memoryTotalBytes
            self.thermalState = thermalState
        }
    }

    /// Two minutes of readings at the default interval.
    package static let historyLength = 60

    @Published package private(set) var latest: Sample?
    @Published package private(set) var cpuHistory: [Double] = []

    private let sampler: () -> Sample?
    private var pollingTask: Task<Void, Never>?

    /// `sampler` reads the host; tests and snapshot boards pass a scripted one.
    package init(sampler: (() -> Sample?)? = nil) {
        self.sampler = sampler ?? HostSampler().sample
    }

    package func start(interval: Duration = .seconds(2)) {
        guard pollingTask == nil else { return }
        pollingTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                self?.sampleNow()
                try? await Task.sleep(for: interval)
            }
        }
    }

    package func stop() {
        pollingTask?.cancel()
        pollingTask = nil
    }

    /// Takes one reading and appends it to the history.
    package func sampleNow() {
        guard let sample = sampler() else { return }
        latest = sample
        cpuHistory.append(sample.cpu)
        if cpuHistory.count > Self.historyLength {
            cpuHistory.removeFirst(cpuHistory.count - Self.historyLength)
        }
    }
}

/// Reads CPU ticks and VM statistics from the Mach host port. CPU share is the busy fraction of
/// the ticks since the previous reading.
private final class HostSampler {
    private let host = mach_host_self()
    private var previousTicks: (busy: UInt64, total: UInt64)?

    /// Reads the tick counters once, so the first sample has a baseline instead of reading 0%.
    init() {
        _ = cpuShare()
    }

    func sample() -> StudioMachineMonitor.Sample? {
        guard let memory = memoryUsed() else { return nil }
        return StudioMachineMonitor.Sample(
            cpu: cpuShare(),
            memoryUsedBytes: memory,
            memoryTotalBytes: ProcessInfo.processInfo.physicalMemory,
            thermalState: ProcessInfo.processInfo.thermalState
        )
    }

    private func cpuShare() -> Double {
        var info = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(host, HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        let user = UInt64(info.cpu_ticks.0), system = UInt64(info.cpu_ticks.1)
        let idle = UInt64(info.cpu_ticks.2), nice = UInt64(info.cpu_ticks.3)
        let busy = user + system + nice
        let total = busy + idle
        defer { previousTicks = (busy, total) }
        guard let previous = previousTicks, total > previous.total else { return 0 }
        return Double(busy - previous.busy) / Double(total - previous.total)
    }

    private func memoryUsed() -> UInt64? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(host, HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pageSize = UInt64(getpagesize())
        let appMemory = UInt64(stats.internal_page_count) - UInt64(min(stats.purgeable_count, stats.internal_page_count))
        return (appMemory + UInt64(stats.wire_count) + UInt64(stats.compressor_page_count)) * pageSize
    }
}
