#if canImport(Cmlx)
import Cmlx
#endif

/// Version of the vendored MLX core this binary was compiled against.
///
/// Used to validate the AOT Metal kernel library (`default.metallib`) at
/// startup: the metallib is built separately from `swift build` (see
/// `scripts/build_mlx_metallib.sh`) and carries a version stamp; loading a
/// library built from a different mlx core silently corrupts inference.
public enum MLXRuntimeVersion {
    /// The mlx core version string (e.g. "0.31.1"), or nil when the MLX C
    /// runtime is not linked (Linux prebuilt MLX builds).
    public static var coreVersion: String? {
        #if canImport(Cmlx)
        var str = mlx_string_new()
        defer { mlx_string_free(str) }
        guard mlx_version(&str) == 0, let data = mlx_string_data(str) else {
            return nil
        }
        return String(cString: data)
        #else
        return nil
        #endif
    }
}
