// swift-tools-version: 6.0
import Foundation
import PackageDescription

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()

func packagePath(_ relativePath: String) -> URL {
  packageRoot.appendingPathComponent(relativePath, isDirectory: false)
}

func packagePathExists(_ relativePath: String) -> Bool {
  FileManager.default.fileExists(atPath: packagePath(relativePath).path)
}

func packageDirectoryContainsSwiftSources(_ relativePath: String) -> Bool {
  let url = packagePath(relativePath)
  var isDirectory: ObjCBool = false
  let fileManager = FileManager.default
  guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
    return false
  }
  guard let enumerator = fileManager.enumerator(
    at: url,
    includingPropertiesForKeys: [.isRegularFileKey],
    options: [.skipsHiddenFiles]
  ) else {
    return false
  }
  for case let sourceURL as URL in enumerator where sourceURL.pathExtension == "swift" {
    return true
  }
  return false
}

#if os(Linux)
let hostIsLinux = true
#else
let hostIsLinux = false
#endif

#if arch(x86_64)
let hostArch = "x86_64"
#elseif arch(arm64)
let hostArch = "arm64"
#else
let hostArch = "unknown"
#endif

let packagePlatformOverride = ProcessInfo.processInfo.environment["MERERUN_PACKAGE_PLATFORM"]?.lowercased()
let isLinuxPackage = if packagePlatformOverride == "linux" {
  true
} else if packagePlatformOverride == "darwin" || packagePlatformOverride == "macos" {
  false
} else {
  hostIsLinux
}

let packagePlatforms: [SupportedPlatform]? = isLinuxPackage ? nil : [.macOS(.v15), .iOS(.v18)]
let commonSwiftSettings: [SwiftSetting] = [
  .interoperabilityMode(.Cxx)
]
let hasMediaIOTarget = packageDirectoryContainsSwiftSources("Sources/MediaIO")

var products: [Product] = [
  .library(name: "MereRunCore", targets: ["MereRunCore"]),
  .library(name: "AudioCore", targets: ["AudioCore"]),
  .library(name: "AudioCodecs", targets: ["AudioCodecs"]),
  .library(name: "AudioSTT", targets: ["AudioSTT"]),
  .library(name: "AudioTTS", targets: ["AudioTTS"]),
  .executable(name: "mere.run", targets: ["MereRunCLI"])
]
if !isLinuxPackage {
  products.append(.executable(name: "mere.run.app", targets: ["MereRunApp"]))
}

var targets: [Target] = []
var mereRunCoreDependencies: [Target.Dependency] = [
  .product(name: "MLX", package: "mlx-swift"),
  .product(name: "MLXFast", package: "mlx-swift"),
  .product(name: "MLXNN", package: "mlx-swift"),
  .product(name: "MLXOptimizers", package: "mlx-swift"),
  .product(name: "MLXRandom", package: "mlx-swift"),
  .product(name: "Crypto", package: "swift-crypto"),
  .product(name: "Transformers", package: "swift-transformers")
]

if hasMediaIOTarget {
  products.append(.library(name: "MediaIO", targets: ["MediaIO"]))
  targets.append(
    .target(
      name: "MediaIO",
      dependencies: [],
      path: "Sources/MediaIO",
      exclude: [
        "README.md"
      ],
      swiftSettings: commonSwiftSettings
    )
  )
  if packageDirectoryContainsSwiftSources("Sources/MediaIOSmoke") {
    targets.append(
      .executableTarget(
        name: "MediaIOSmoke",
        dependencies: [
          "MediaIO"
        ],
        path: "Sources/MediaIOSmoke",
        swiftSettings: commonSwiftSettings
      )
    )
  }
  mereRunCoreDependencies.append("MediaIO")
}

if isLinuxPackage {
  if packagePathExists("Sources/llama/module.modulemap") {
    targets.append(
      .systemLibrary(
        name: "llama",
        path: "Sources/llama",
        pkgConfig: "llama"
      )
    )
    mereRunCoreDependencies.append("llama")
  }
} else {
  targets.append(
    .binaryTarget(
      name: "llama",
      path: "vendor/llama.xcframework"
    )
  )
  mereRunCoreDependencies.append("llama")
}

let linuxNativeLinkerSettings: [LinkerSetting] = isLinuxPackage
  ? [.unsafeFlags(["-Xlinker", "-L\(packagePath(".build/native/linux-\(hostArch)/llama/lib").path)"])]
  : []

var mereRunCoreLinkerSettings: [LinkerSetting] = linuxNativeLinkerSettings
if !isLinuxPackage {
  mereRunCoreLinkerSettings.append(.linkedFramework("Metal"))
}

var audioCodecsDependencies: [Target.Dependency] = [
  .product(name: "MLX", package: "mlx-swift")
]
if hasMediaIOTarget {
  audioCodecsDependencies.append("MediaIO")
}

var mereRunCoreTestDependencies: [Target.Dependency] = [
  "MereRunCore",
  "AudioCore",
  "AudioCodecs",
  "AudioSTT",
  "AudioTTS",
  .product(name: "Crypto", package: "swift-crypto")
]
if hasMediaIOTarget {
  mereRunCoreTestDependencies.append("MediaIO")
}

targets.append(contentsOf: [
  .target(
    name: "MereRunCore",
    dependencies: mereRunCoreDependencies,
    path: "Sources/MereRunCore",
    exclude: [
      "README.md",
      "ACEStep/README.md",
      "ACEStep/Model/README.md",
      "ACEStep/VAE/README.md",
      "CodeGen/README.md",
      "DeepseekV4Flash/README.md",
      "FalconPerception/README.md",
      "Flux2Klein/README.md",
      "Flux2Klein/Model/Transformer/README.md",
      "Gemma4/README.md",
      "HiDreamO1/README.md",
      "LightOnOCR/README.md",
      "LTX/README.md",
      "LoRA/README.md",
      "PrivacyFilter/README.md",
      "Psi/README.md",
      "Q35/README.md",
      "QwenImageEdit/README.md",
      "QwenImageEdit/Model/Transformer/README.md",
      "QwenImageEdit/Model/VAE/README.md",
      "SAM3/README.md",
      "Support/README.md",
      "VLM/README.md",
      "ZImageI2L/README.md",
      "ZImageI2L/Model/README.md",
      "ZImageTurbo/README.md",
      "ZImageTurbo/Model/TextEncoder/README.md",
      "ZImageTurbo/Model/TextEncoder/Vision/README.md",
      "ZImageTurbo/Model/Transformer/README.md",
      "ZImageTurbo/Model/VAE/README.md",
      "ZImageTurbo/Util/README.md"
    ],
    swiftSettings: commonSwiftSettings,
    linkerSettings: mereRunCoreLinkerSettings
  ),
  .target(
    name: "AudioCore",
    dependencies: [],
    path: "Sources/AudioCore",
    exclude: [
      "README.md"
    ],
    swiftSettings: commonSwiftSettings
  ),
  .target(
    name: "AudioCodecs",
    dependencies: audioCodecsDependencies,
    path: "Sources/AudioCodecs",
    exclude: [
      "README.md"
    ],
    swiftSettings: commonSwiftSettings
  ),
  .target(
    name: "AudioSTT",
    dependencies: [
      "MereRunCore",
      "AudioCore",
      "AudioCodecs",
      .product(name: "MLX", package: "mlx-swift"),
      .product(name: "MLXFast", package: "mlx-swift"),
      .product(name: "MLXNN", package: "mlx-swift"),
      .product(name: "MLXRandom", package: "mlx-swift"),
      .product(name: "Transformers", package: "swift-transformers")
    ],
    path: "Sources/AudioSTT",
    exclude: [
      "Parakeet/README.md",
      "Qwen3ASR/Model/README.md",
      "Qwen3ASR/README.md"
    ],
    swiftSettings: commonSwiftSettings
  ),
  .target(
    name: "AudioTTS",
    dependencies: [
      "MereRunCore",
      "AudioCore",
      "AudioCodecs",
      .product(name: "MLX", package: "mlx-swift"),
      .product(name: "MLXFast", package: "mlx-swift"),
      .product(name: "MLXNN", package: "mlx-swift"),
      .product(name: "MLXRandom", package: "mlx-swift"),
      .product(name: "Transformers", package: "swift-transformers")
    ],
    path: "Sources/AudioTTS",
    exclude: [
      "Qwen3TTS/README.md"
    ],
    swiftSettings: commonSwiftSettings
  ),
  .executableTarget(
    name: "MereRunCLI",
    dependencies: [
      "MereRunCore",
      "AudioCore",
      "AudioCodecs",
      "AudioSTT",
      "AudioTTS",
      .product(name: "ArgumentParser", package: "swift-argument-parser"),
      .product(name: "Hummingbird", package: "hummingbird")
    ],
    path: "Sources/MereRunCLI",
    exclude: [
      "Commands/README.md",
      "Support/README.md"
    ],
    resources: [
      .process("Guides")
    ],
    swiftSettings: commonSwiftSettings,
    linkerSettings: linuxNativeLinkerSettings
  ),
  .testTarget(
    name: "MereRunCoreTests",
    dependencies: mereRunCoreTestDependencies,
    path: "Tests/MereRunCoreTests",
    swiftSettings: commonSwiftSettings,
    linkerSettings: linuxNativeLinkerSettings
  ),
  .testTarget(
    name: "MereRunCLITests",
    dependencies: ["MereRunCLI"],
    path: "Tests/MereRunCLITests",
    swiftSettings: commonSwiftSettings,
    linkerSettings: linuxNativeLinkerSettings
  )
])

if !isLinuxPackage {
  targets.append(
    .executableTarget(
      name: "MereRunApp",
      dependencies: [],
      path: "Sources/MereRunApp",
      exclude: [
        "README.md"
      ]
    )
  )
  targets.append(
    .testTarget(
      name: "MereRunAppTests",
      dependencies: ["MereRunApp"],
      path: "Tests/MereRunAppTests"
    )
  )
}

let package = Package(
  name: "MereRun",
  platforms: packagePlatforms,
  products: products,
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    .package(
      url: "https://github.com/huggingface/swift-transformers",
      from: "1.3.0"
    ),
    .package(url: "https://github.com/apple/swift-crypto.git", from: "4.0.0"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
    .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0")
  ],
  targets: targets
)
