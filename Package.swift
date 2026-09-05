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
let useLinuxPrebuiltMLX = isLinuxPackage
  && ProcessInfo.processInfo.environment["MERERUN_MLX_SWIFT_LINKAGE"]?.lowercased() == "cuda-prebuilt"
let prebuiltMLXBuildRoot = ProcessInfo.processInfo.environment["MERERUN_MLX_SWIFT_BUILD_DIR"]
  ?? packagePath(".build/native/linux-\(hostArch)/build/mlx-swift-cuda-smoke").path
let prebuiltMLXSourceRoot = ProcessInfo.processInfo.environment["MERERUN_MLX_SWIFT_SOURCE_DIR"]
  ?? packagePath(".build/native/src/mlx-swift").path
let prebuiltMLXSwiftSettings: [SwiftSetting] = useLinuxPrebuiltMLX
  ? [
      .unsafeFlags([
        "-I", prebuiltMLXBuildRoot,
        "-I", "\(prebuiltMLXBuildRoot)/swift",
        "-I", "\(prebuiltMLXSourceRoot)/Source/Cmlx/include",
        "-I", "\(prebuiltMLXBuildRoot)/_deps/mlx-c-src",
        "-I", "\(prebuiltMLXBuildRoot)/_deps/swift-numerics-src/Sources/_NumericsShims/include"
      ])
    ]
  : []
let linuxPackageSwiftSettings: [SwiftSetting] = isLinuxPackage
  ? [
      .unsafeFlags([
        "-strict-concurrency=targeted"
      ])
    ]
  : []
let commonSwiftSettings: [SwiftSetting] = [
  .interoperabilityMode(.Cxx)
] + prebuiltMLXSwiftSettings + linuxPackageSwiftSettings
let hasMediaIOTarget = packageDirectoryContainsSwiftSources("Sources/MediaIO")
let hasMagentaRT2Binary = !isLinuxPackage && packagePathExists("vendor/magentart.xcframework")

func mlxDependency(_ name: String) -> [Target.Dependency] {
  useLinuxPrebuiltMLX ? [] : [.product(name: name, package: "mlx-swift")]
}

func shellSplit(_ value: String) -> [String] {
  value.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" }).map(String.init)
}

let prebuiltMLXExtraLinkFlags = ProcessInfo.processInfo.environment["MERERUN_MLX_SWIFT_LINK_FLAGS"].map(shellSplit) ?? []
let prebuiltMLXLinkerSettings: [LinkerSetting] = useLinuxPrebuiltMLX
  ? [
      .unsafeFlags([
        "-L", prebuiltMLXBuildRoot,
        "-L", "\(prebuiltMLXBuildRoot)/_deps/mlx-c-build",
        "-L", "\(prebuiltMLXBuildRoot)/_deps/mlx-build",
        "-L", "\(prebuiltMLXBuildRoot)/_deps/mlx-build/mlx/io",
        "-L", "\(prebuiltMLXBuildRoot)/lib",
        "-lMLXOptimizers",
        "-lMLXNN",
        "-lMLXRandom",
        "-lMLXFast",
        "-lMLXFFT",
        "-lMLXLinalg",
        "-lMLX",
        "-lmlxc",
        "-lmlx",
        "-lgguflib",
        "-lNumerics",
        "-lComplexModule",
        "-lRealModule",
        "-lswiftSwiftOnoneSupport",
        "-lpthread",
        "-ldl",
        "-lstdc++",
        "-lm",
        "-lblas",
        "-llapack",
        "-lopenblas"
      ] + prebuiltMLXExtraLinkFlags)
    ]
  : []

var products: [Product] = [
  .library(name: "MereRunContract", targets: ["MereRunContract"]),
  .library(name: "MereRunEvaluation", targets: ["MereRunEvaluation"]),
  .library(name: "MereRunRelayKit", targets: ["MereRunRelayKit"]),
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
var mereRunCoreDependencies: [Target.Dependency] = mlxDependency("MLX")
mereRunCoreDependencies.append(contentsOf: mlxDependency("MLXFast"))
mereRunCoreDependencies.append(contentsOf: mlxDependency("MLXFFT"))
mereRunCoreDependencies.append(contentsOf: mlxDependency("MLXNN"))
mereRunCoreDependencies.append(contentsOf: mlxDependency("MLXOptimizers"))
mereRunCoreDependencies.append(contentsOf: mlxDependency("MLXRandom"))
mereRunCoreDependencies.append("AudioCodecs")
mereRunCoreDependencies.append(.product(name: "Crypto", package: "swift-crypto"))
mereRunCoreDependencies.append(.product(name: "Transformers", package: "swift-transformers"))
if !isLinuxPackage {
  mereRunCoreDependencies.append(
    .product(
      name: "CONNXRuntime",
      package: "swift-onnxruntime",
      condition: .when(platforms: [.macOS])
    )
  )
  mereRunCoreDependencies.append(
    .target(name: "COnnxRuntimeCoreML", condition: .when(platforms: [.macOS]))
  )
}

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
    mereRunCoreDependencies.append(.target(name: "llama"))
  }
} else {
  targets.append(
    .binaryTarget(
      name: "llama",
      path: "vendor/llama.xcframework"
    )
  )
  // The iOS Studio runtime uses the native MLX chat and image paths. Keeping
  // llama.cpp macOS-only also avoids embedding unused platform slices in the
  // phone app.
  mereRunCoreDependencies.append(.target(name: "llama", condition: .when(platforms: [.macOS])))
}
if hasMagentaRT2Binary {
  targets.append(
    .binaryTarget(
      name: "magentart",
      path: "vendor/magentart.xcframework"
    )
  )
  // The Magenta RT2 binary ships macOS slices only; iOS builds of the core
  // exclude realtime music.
  mereRunCoreDependencies.append(.target(name: "magentart", condition: .when(platforms: [.macOS])))
}
if !isLinuxPackage {
  targets.append(
    .target(
      name: "COnnxRuntimeCoreML",
      dependencies: [
        .product(
          name: "CONNXRuntime",
          package: "swift-onnxruntime",
          condition: .when(platforms: [.macOS])
        )
      ],
      path: "Sources/COnnxRuntimeCoreML",
      publicHeadersPath: "include"
    )
  )
}

let linuxNativeLlamaLibraryPath = packagePath(".build/native/linux-\(hostArch)/llama/lib").path
let linuxNativeLinkerSettings: [LinkerSetting] = isLinuxPackage
  ? [.unsafeFlags([
      "-Xlinker", "-L\(linuxNativeLlamaLibraryPath)",
      "-Xlinker", "-rpath", "-Xlinker", linuxNativeLlamaLibraryPath,
      "-Xlinker", "-rpath", "-Xlinker", "$ORIGIN/lib"
    ])] + prebuiltMLXLinkerSettings
  : []

var mereRunCoreLinkerSettings: [LinkerSetting] = linuxNativeLinkerSettings
if !isLinuxPackage {
  mereRunCoreLinkerSettings.append(.linkedFramework("Metal"))
  mereRunCoreLinkerSettings.append(.linkedFramework("Vision"))
  mereRunCoreLinkerSettings.append(.linkedFramework("ImageIO"))
  mereRunCoreLinkerSettings.append(.linkedFramework("CoreVideo"))
  // IOKit does not exist on iOS.
  mereRunCoreLinkerSettings.append(.linkedFramework("IOKit", .when(platforms: [.macOS])))
}

var audioCodecsDependencies: [Target.Dependency] = mlxDependency("MLX")
if hasMediaIOTarget {
  audioCodecsDependencies.append("MediaIO")
}

var mereRunCoreTestDependencies: [Target.Dependency] = [
  "MereRunContract",
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

var audioRuntimeDependencies: [Target.Dependency] = [
  "MereRunCore",
  "AudioCore",
  "AudioCodecs",
  .product(name: "Transformers", package: "swift-transformers")
]
audioRuntimeDependencies.append(contentsOf: mlxDependency("MLX"))
audioRuntimeDependencies.append(contentsOf: mlxDependency("MLXFast"))
audioRuntimeDependencies.append(contentsOf: mlxDependency("MLXNN"))
audioRuntimeDependencies.append(contentsOf: mlxDependency("MLXRandom"))

var mereRunCLIDependencies: [Target.Dependency] = [
  "MereRunContract",
  "MereRunEvaluation",
  "MereRunRelayKit",
  "MereRunCore",
  "AudioCore",
  "AudioCodecs",
  "AudioSTT",
  "AudioTTS",
  .product(name: "ArgumentParser", package: "swift-argument-parser"),
  .product(name: "Crypto", package: "swift-crypto"),
  .product(name: "Hummingbird", package: "hummingbird")
]
if hasMediaIOTarget {
  mereRunCLIDependencies.append("MediaIO")
}

targets.append(contentsOf: [
  .target(
    name: "MereRunContract",
    dependencies: [],
    path: "Sources/MereRunContract",
    exclude: [
      "README.md"
    ]
  ),
  .target(
    name: "MereRunEvaluation",
    dependencies: [
      .product(name: "Crypto", package: "swift-crypto")
    ],
    path: "Sources/MereRunEvaluation",
    exclude: [
      "README.md"
    ]
  ),
  .target(
    name: "MereRunRelayKit",
    dependencies: [
      .product(name: "Crypto", package: "swift-crypto")
    ],
    path: "Sources/MereRunRelayKit",
    exclude: [
      "README.md"
    ]
  ),
  .target(
    name: "MereRunCore",
    dependencies: mereRunCoreDependencies,
    path: "Sources/MereRunCore",
    exclude: [
      "README.md",
      "ACEStep/README.md",
      "ACEStep/Model/README.md",
      "ACEStep/VAE/README.md",
      "APBWE/README.md",
      "Asset3D/README.md",
      "CodeGen/README.md",
      "Cosmos3/README.md",
      "Decode/README.md",
      "DeepseekV4Flash/README.md",
      "DiffusionGemma/README.md",
      "DepthAnything3/README.md",
      "FaceAnalysis/README.md",
      "FalconPerception/README.md",
      "Flux1/README.md",
      "Flux2Klein/README.md",
      "Flux2Klein/Model/Transformer/README.md",
      "Gemma4/README.md",
      "Geometry/README.md",
      "HiDreamO1/README.md",
      "Ideogram4/README.md",
      "Inkling/README.md",
      "InstantMesh/README.md",
      "Krea2/README.md",
      "Laguna/README.md",
      "MuScriptor/README.md",
      "MuseGlimmer/README.md",
      "NemotronH/README.md",
      "NemotronOmni/README.md",
      "LFM2/README.md",
      "LightOnOCR/README.md",
      "LTX/README.md",
      "LoRA/README.md",
      "MagentaRT2/README.md",
      "MLX/README.md",
      "MMAudio/README.md",
      "MultimodalEmbeddings/README.md",
      "MiniMaxH3/README.md",
      "MiniMaxMusic3/README.md",
      "PrivacyFilter/README.md",
      "RoFormer/README.md",
      "UniverSR/README.md",
      "Pose/README.md",
      "OpticalFlow/README.md",
      "OlmoEarth/README.md",
      "Psi/README.md",
      "Q35/README.md",
      "Quantization/README.md",
      "QwenImageEdit/README.md",
      "QwenImageEdit/Model/Transformer/README.md",
      "QwenImageEdit/Model/VAE/README.md",
      "SAM3/README.md",
      "SenseNovaU15/README.md",
      "SCAIL2/README.md",
      "Support/README.md",
      "TESSERA/README.md",
      "Trellis2/README.md",
      "TerraMindFire/README.md",
      "TerraMindFlood/README.md",
      "TripoSR/README.md",
      "VLM/README.md",
      "VideoDepth/README.md",
      "VideoDepth/VDA/README.md",
      "VisionGeometry/MoGe2/README.md",
      "Wan2/README.md",
      "Woosh/README.md",
      "ZImageI2L/README.md",
      "ZImageI2L/Model/README.md",
      "ZImageTurbo/README.md",
      "ZImageTurbo/Model/TextEncoder/README.md",
      "ZImageTurbo/Model/TextEncoder/LLMGeneration/README.md",
      "ZImageTurbo/Model/TextEncoder/Vision/README.md",
      "ZImageTurbo/Model/Transformer/README.md",
      "ZImageTurbo/Model/VAE/README.md",
      "ZImageTurbo/Util/README.md"
    ],
    resources: [
      .process("Resources")
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
    dependencies: audioRuntimeDependencies,
    path: "Sources/AudioSTT",
    exclude: [
      "Parakeet/README.md",
      "Qwen3ASR/Model/README.md",
      "Qwen3ASR/README.md",
      "Sortformer/README.md"
    ],
    swiftSettings: commonSwiftSettings,
    linkerSettings: isLinuxPackage ? [] : [
      .linkedFramework("CoreML", .when(platforms: [.macOS, .iOS]))
    ]
  ),
  .target(
    name: "AudioTTS",
    dependencies: audioRuntimeDependencies,
    path: "Sources/AudioTTS",
    exclude: [
      "Qwen3TTS/README.md"
    ],
    swiftSettings: commonSwiftSettings
  ),
  .executableTarget(
    name: "MereRunCLI",
    dependencies: mereRunCLIDependencies,
    path: "Sources/MereRunCLI",
    exclude: [
      "Commands/README.md",
      "Support/README.md"
    ],
    resources: [
      .process("Guides"),
      .process("BenchmarkSuites")
    ],
    swiftSettings: commonSwiftSettings,
    linkerSettings: linuxNativeLinkerSettings
  ),
  .testTarget(
    name: "MereRunContractTests",
    dependencies: ["MereRunContract"],
    path: "Tests/MereRunContractTests"
  ),
  .testTarget(
    name: "MereRunEvaluationTests",
    dependencies: ["MereRunEvaluation"],
    path: "Tests/MereRunEvaluationTests",
    resources: [
      .copy("Fixtures")
    ]
  ),
  .testTarget(
    name: "MereRunRelayKitTests",
    dependencies: ["MereRunRelayKit"],
    path: "Tests/MereRunRelayKitTests"
  ),
  .testTarget(
    name: "MereRunCoreTests",
    dependencies: mereRunCoreTestDependencies,
    path: "Tests/MereRunCoreTests",
    resources: [
      .copy("Fixtures")
    ],
    swiftSettings: commonSwiftSettings,
    linkerSettings: linuxNativeLinkerSettings
  ),
  .testTarget(
    name: "MereRunCLITests",
    dependencies: ["MereRunCLI", "MereRunEvaluation", "MediaIO"],
    path: "Tests/MereRunCLITests",
    resources: [
      .copy("Fixtures")
    ],
    swiftSettings: commonSwiftSettings,
    linkerSettings: linuxNativeLinkerSettings
  ),
  .testTarget(
    name: "AudioTTSTests",
    dependencies: ["AudioTTS"],
    path: "Tests/AudioTTSTests",
    swiftSettings: commonSwiftSettings,
    linkerSettings: linuxNativeLinkerSettings
  )
])

if !isLinuxPackage {
  // The Studio is three targets: StudioKit is the SwiftUI-free model layer (CLI environment,
  // process runner, jobs, catalog, library, readiness), StudioUI is every view, and MereRunApp is
  // the executable shell that owns the scenes, the menus, Settings, and Sparkle. Cross-module
  // declarations use `package` so the split keeps the visibility the single target had without
  // widening any public API.
  targets.append(
    .target(
      name: "StudioKit",
      dependencies: ["MereRunContract"],
      path: "apps/macos/StudioKit"
    )
  )
  targets.append(
    .target(
      name: "StudioUI",
      dependencies: ["StudioKit", "MereRunContract"],
      path: "apps/macos/StudioUI",
      resources: [
        // Caveat (OFL 1.1) for the sidebar wordmark; registered at launch by MereRunTheme.Brand.
        .copy("Resources/Fonts")
      ],
      linkerSettings: [
        .linkedFramework("AVKit")
      ]
    )
  )
  // Test doubles shared by StudioKitTests and StudioUITests. Nothing that ships depends on it, so
  // the recording process runner stays out of the app binary.
  targets.append(
    .target(
      name: "StudioTestSupport",
      dependencies: ["StudioKit"],
      path: "apps/macos/StudioTestSupport"
    )
  )
  targets.append(
    .executableTarget(
      name: "MereRunApp",
      dependencies: [
        "StudioKit",
        "StudioUI",
        "MereRunContract",
        .product(name: "Sparkle", package: "Sparkle")
      ],
      path: "apps/macos/MereRunStudio",
      linkerSettings: [
        .unsafeFlags([
          "-Xlinker", "-rpath",
          "-Xlinker", "@loader_path/../Frameworks"
        ])
      ]
    )
  )
  targets.append(
    .testTarget(
      name: "StudioKitTests",
      dependencies: ["StudioKit", "StudioTestSupport", "MereRunContract"],
      path: "apps/macos/StudioKitTests",
      // Read from source with #filePath by CommandArgumentGoldenTests, not from the bundle.
      exclude: ["Fixtures"]
    )
  )
  targets.append(
    .testTarget(
      name: "StudioUITests",
      dependencies: ["StudioUI", "StudioKit", "StudioTestSupport", "MereRunContract"],
      path: "apps/macos/StudioUITests"
    )
  )
  targets.append(
    .testTarget(
      name: "MereRunAppTests",
      dependencies: ["MereRunApp", "MereRunContract"],
      path: "apps/macos/MereRunStudioTests"
    )
  )
}

var packageDependencies: [Package.Dependency] = (useLinuxPrebuiltMLX ? [] : [
  .package(
    url: "https://github.com/sawfwair/mlx-swift",
    revision: "7558b9cff75746e3ce25802aecbdc498b240af7f"
  )
]) + [
  .package(
    url: "https://github.com/huggingface/swift-transformers",
    from: "1.3.0"
  ),
  .package(url: "https://github.com/apple/swift-crypto.git", "4.0.0"..<"4.4.0"),
  .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
  .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0")
]
if !isLinuxPackage {
  packageDependencies.append(
    .package(url: "https://github.com/readdle/swift-onnxruntime.git", exact: "1.20.1")
  )
  packageDependencies.append(
    .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.9.5")
  )
}

let package = Package(
  name: "MereRun",
  platforms: packagePlatforms,
  products: products,
  dependencies: packageDependencies,
  targets: targets
)
