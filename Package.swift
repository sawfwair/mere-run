// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "MereRun",
  platforms: [.macOS(.v15), .iOS(.v18)],
  products: [
    .library(name: "MereRunCore", targets: ["MereRunCore"]),
    .library(name: "AudioCore", targets: ["AudioCore"]),
    .library(name: "AudioCodecs", targets: ["AudioCodecs"]),
    .library(name: "AudioSTT", targets: ["AudioSTT"]),
    .library(name: "AudioTTS", targets: ["AudioTTS"]),
    .executable(name: "mere.run", targets: ["MereRunCLI"]),
    .executable(name: "mere.run.app", targets: ["MereRunApp"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    .package(
      url: "https://github.com/huggingface/swift-transformers",
      from: "1.3.0"
    ),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
    .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0")
  ],
  targets: [
    .binaryTarget(
      name: "llama",
      path: "vendor/llama.xcframework"
    ),
    .target(
      name: "MereRunCore",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXOptimizers", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
        "llama"
      ],
      path: "Sources/MereRunCore",
      exclude: [
        "LTX/README.md",
        "LoRA/README.md"
      ],
      swiftSettings: [
        .interoperabilityMode(.Cxx)
      ],
      linkerSettings: [
        .linkedFramework("Metal")
      ]
    ),
    .target(
      name: "AudioCore",
      dependencies: [],
      path: "Sources/AudioCore",
      swiftSettings: [
        .interoperabilityMode(.Cxx)
      ]
    ),
    .target(
      name: "AudioCodecs",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift")
      ],
      path: "Sources/AudioCodecs",
      swiftSettings: [
        .interoperabilityMode(.Cxx)
      ]
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
        "Qwen3ASR/README.md"
      ],
      swiftSettings: [
        .interoperabilityMode(.Cxx)
      ]
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
      swiftSettings: [
        .interoperabilityMode(.Cxx)
      ]
    ),
    .executableTarget(
      name: "MereRunApp",
      dependencies: [],
      path: "Sources/MereRunApp"
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
        "Commands/README.md"
      ],
      swiftSettings: [
        .interoperabilityMode(.Cxx)
      ]
    ),
    .testTarget(
      name: "MereRunCoreTests",
      dependencies: [
        "MereRunCore",
        "AudioCore",
        "AudioCodecs",
        "AudioSTT",
        "AudioTTS"
      ],
      path: "Tests/MereRunCoreTests",
      swiftSettings: [
        .interoperabilityMode(.Cxx)
      ]
    ),
    .testTarget(
      name: "MereRunCLITests",
      dependencies: ["MereRunCLI"],
      path: "Tests/MereRunCLITests",
      swiftSettings: [
        .interoperabilityMode(.Cxx)
      ]
    ),
    .testTarget(
      name: "MereRunAppTests",
      dependencies: ["MereRunApp"],
      path: "Tests/MereRunAppTests"
    ),
  ]
)
