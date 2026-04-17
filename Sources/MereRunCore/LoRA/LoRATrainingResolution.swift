import Foundation

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

public struct LoRAResolvedResolution: Sendable, Hashable {
    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }
}

public enum LoRATrainingResolution {
    public static func resolve(
        width: Int,
        height: Int,
        maxResolution: Int?,
        multiple: Int,
        errorContext: String
    ) throws -> LoRAResolvedResolution {
        guard width > 0, height > 0 else {
            throw NSError(
                domain: "LoRATrainingResolution",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Image dimensions must be > 0: \(errorContext)"]
            )
        }
        guard multiple > 0 else {
            throw NSError(
                domain: "LoRATrainingResolution",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Resolution multiple must be > 0 (got \(multiple))."]
            )
        }
        if let maxResolution, maxResolution <= 0 {
            throw NSError(
                domain: "LoRATrainingResolution",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "maxResolution must be > 0 (got \(maxResolution))."]
            )
        }

        var resolvedWidth = width
        var resolvedHeight = height
        if let maxResolution {
            let maxDim = max(resolvedWidth, resolvedHeight)
            if maxDim > maxResolution {
                let scale = Double(maxResolution) / Double(maxDim)
                resolvedWidth = Int(Double(resolvedWidth) * scale)
                resolvedHeight = Int(Double(resolvedHeight) * scale)
            }
        }

        resolvedWidth = multiple * (resolvedWidth / multiple)
        resolvedHeight = multiple * (resolvedHeight / multiple)
        guard resolvedWidth > 0, resolvedHeight > 0 else {
            throw NSError(
                domain: "LoRATrainingResolution",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Image too small for training (needs >=\(multiple)px): \(errorContext) (\(width)x\(height))"]
            )
        }

        return LoRAResolvedResolution(width: resolvedWidth, height: resolvedHeight)
    }

    public static func resolveFromImage(
        at imageURL: URL,
        maxResolution: Int?,
        multiple: Int
    ) throws -> LoRAResolvedResolution {
        let size = try imageSize(at: imageURL)
        return try resolve(
            width: size.width,
            height: size.height,
            maxResolution: maxResolution,
            multiple: multiple,
            errorContext: imageURL.path
        )
    }

    public static func imageSize(at imageURL: URL) throws -> (width: Int, height: Int) {
        #if canImport(CoreGraphics)
        guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let height = properties[kCGImagePropertyPixelHeight] as? NSNumber else {
            throw NSError(
                domain: "LoRATrainingResolution",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Failed to read image dimensions: \(imageURL.path)"]
            )
        }
        return (width: width.intValue, height: height.intValue)
        #else
        throw NSError(
            domain: "LoRATrainingResolution",
            code: 6,
            userInfo: [NSLocalizedDescriptionKey: "Image size inference requires CoreGraphics support."]
        )
        #endif
    }

    public static func allocateSteps(totalSteps: Int, bucketSizes: [Int]) -> [Int] {
        guard totalSteps > 0, !bucketSizes.isEmpty else {
            return Array(repeating: 0, count: bucketSizes.count)
        }

        var steps = Array(repeating: 0, count: bucketSizes.count)
        let totalSamples = max(bucketSizes.reduce(0, +), 1)

        if totalSteps >= bucketSizes.count {
            for index in bucketSizes.indices {
                steps[index] = 1
            }
            let remaining = totalSteps - bucketSizes.count
            if remaining == 0 { return steps }

            var rawExtras: [(index: Int, base: Int, fraction: Double)] = []
            rawExtras.reserveCapacity(bucketSizes.count)
            for index in bucketSizes.indices {
                let weighted = Double(remaining) * Double(bucketSizes[index]) / Double(totalSamples)
                let base = Int(weighted.rounded(.down))
                rawExtras.append((index: index, base: base, fraction: weighted - Double(base)))
                steps[index] += base
            }

            let used = rawExtras.reduce(0) { $0 + $1.base }
            let leftovers = remaining - used
            if leftovers > 0 {
                let order = rawExtras.sorted {
                    if $0.fraction == $1.fraction {
                        return bucketSizes[$0.index] > bucketSizes[$1.index]
                    }
                    return $0.fraction > $1.fraction
                }
                for idx in 0..<leftovers {
                    steps[order[idx % order.count].index] += 1
                }
            }
            return steps
        }

        let order = bucketSizes.indices.sorted {
            if bucketSizes[$0] == bucketSizes[$1] {
                return $0 < $1
            }
            return bucketSizes[$0] > bucketSizes[$1]
        }
        for index in order.prefix(totalSteps) {
            steps[index] = 1
        }
        return steps
    }
}
