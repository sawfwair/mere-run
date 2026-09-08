import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Nonblocking process lease. The descriptor is never inherited across exec.
final class ImageRunLease: @unchecked Sendable {
    private let mutex = NSLock()
    private var descriptor: Int32

    private init(_ descriptor: Int32) { self.descriptor = descriptor }
    deinit { release() }

    static func createDirectory(_ directory: URL) throws {
        guard directory.path.withCString({ mkdir($0, S_IRWXU) }) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }

    static func acquire(in directory: URL) throws -> ImageRunLease? {
        let path = directory.appendingPathComponent(".image-run.lock").path
        let descriptor = path.withCString { open($0, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR) }
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let code = errno
            close(descriptor)
            if code == EWOULDBLOCK || code == EAGAIN { return nil }
            throw POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO)
        }
        return ImageRunLease(descriptor)
    }

    func release() {
        mutex.withLock {
            guard descriptor >= 0 else { return }
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
            descriptor = -1
        }
    }
}
