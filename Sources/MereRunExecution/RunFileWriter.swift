import Crypto
import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Publishes a complete private file without opening the previous version.
/// Operation owners retain their run lease around this write.
enum RunFileWriter {
    static func write(_ data: Data, to destination: URL) throws -> RunArtifact {
        let directory = destination.deletingLastPathComponent()
        let temporary = directory.appendingPathComponent(".run-write-\(UUID().uuidString).tmp")
        let descriptor = temporary.path.withCString {
            open($0, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        }
        guard descriptor >= 0 else { throw posixError() }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer {
            try? handle.close()
            try? FileManager.default.removeItem(at: temporary)
        }
#if canImport(Darwin)
        // Set protection before writing sensitive bytes. Keeping this descriptor
        // open lets the write finish if the device locks in the meantime.
        let volume = try temporary.resourceValues(forKeys: [.volumeSupportsFileProtectionKey])
        if volume.allValues[.volumeSupportsFileProtectionKey] as? Bool == true {
            try FileManager.default.setAttributes(
                [.protectionKey: FileProtectionType.completeUnlessOpen], ofItemAtPath: temporary.path
            )
        }
#endif
        try handle.write(contentsOf: data)
        try handle.synchronize()
        let result = temporary.path.withCString { source in
            destination.path.withCString { target in rename(source, target) }
        }
        guard result == 0 else { throw posixError() }
        let directoryDescriptor = directory.path.withCString { open($0, O_RDONLY | O_DIRECTORY | O_CLOEXEC) }
        guard directoryDescriptor >= 0 else { throw posixError() }
        defer { close(directoryDescriptor) }
        guard fsync(directoryDescriptor) == 0 else { throw posixError() }
        return RunArtifact(
            url: destination, sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            byteCount: UInt64(data.count)
        )
    }

    private static func posixError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }
}
