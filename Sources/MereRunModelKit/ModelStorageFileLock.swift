import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Cross-process exclusion for operations that mutate the Hub payload store.
public final class ModelStorageFileLock: @unchecked Sendable {
    private var descriptor: Int32

    private init(descriptor: Int32) {
        self.descriptor = descriptor
    }

    deinit {
        unlock()
    }

    public static func acquire(
        hubDirectory: URL,
        fileManager: FileManager = .default
    ) throws -> ModelStorageFileLock {
        try fileManager.createDirectory(at: hubDirectory, withIntermediateDirectories: true)
        let lockURL = hubDirectory.appendingPathComponent(".mererun-storage.lock", isDirectory: false)
        let descriptor = lockURL.path.withCString {
            open($0, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        }
        guard descriptor >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        guard flock(descriptor, LOCK_EX) == 0 else {
            let code = errno
            close(descriptor)
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        return ModelStorageFileLock(descriptor: descriptor)
    }

    public func unlock() {
        guard descriptor >= 0 else { return }
        _ = flock(descriptor, LOCK_UN)
        close(descriptor)
        descriptor = -1
    }
}
