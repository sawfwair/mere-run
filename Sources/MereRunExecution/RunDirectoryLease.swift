import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Nonblocking process lease. The descriptor is never inherited across exec.
public final class RunDirectoryLease: @unchecked Sendable {
    private let mutex = NSLock()
    private var descriptor: Int32

    private init(_ descriptor: Int32) { self.descriptor = descriptor }
    deinit { release() }

    public static func createDirectory(_ directory: URL) throws {
        guard directory.path.withCString({ mkdir($0, S_IRWXU) }) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    public static func acquire(in directory: URL, filename: String) throws -> RunDirectoryLease? {
        let path = directory.appendingPathComponent(filename).path
        let descriptor = path.withCString { open($0, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR) }
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let code = errno
            close(descriptor)
            if code == EWOULDBLOCK || code == EAGAIN { return nil }
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        return RunDirectoryLease(descriptor)
    }

    /// A reader deciding whether an abandoned record can be recovered. A directory
    /// that cannot hold a lock file leaves liveness unknown, so the caller reports
    /// the record as written instead of failing the read.
    public static func acquireForRecovery(in directory: URL, filename: String) throws -> RunDirectoryLease? {
        do {
            return try acquire(in: directory, filename: filename)
        } catch let error as POSIXError where error.code == .EACCES || error.code == .EPERM || error.code == .EROFS {
            return nil
        }
    }

    public func release() {
        mutex.withLock {
            guard descriptor >= 0 else { return }
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
            descriptor = -1
        }
    }
}
