#!/usr/bin/env bash
set -euo pipefail

# Run the app's SwiftUI-independent lifecycle sources and their actual XCTest files on the host.
# This isolated test package needs neither a simulator nor checkpoint/model dependencies.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_package="$(mktemp -d "${TMPDIR:-/tmp}/mere-run-ios-states.XXXXXX")"
trap 'rm -rf "$test_package"' EXIT
mkdir "$test_package/Tests"
cp "$repo_root/apps/ios/MereRunStudio/App/RuntimeResidencyState.swift" \
   "$repo_root/apps/ios/MereRunStudio/App/ChatTurnStore.swift" \
   "$repo_root/apps/ios/MereRunStudio/App/RelaySnapshotStore.swift" \
   "$repo_root/apps/ios/MereRunStudio/App/PendingModelDownloadStore.swift" \
   "$repo_root/apps/ios/MereRunStudio/App/ModelDownloadOperation.swift" \
   "$repo_root/apps/ios/MereRunStudio/App/RuntimeWarmup.swift" \
   "$repo_root/apps/ios/MereRunStudio/App/RunPollingPolicy.swift" \
   "$repo_root/apps/ios/MereRunStudio/App/RunSnapshotPoller.swift" \
   "$repo_root/apps/ios/MereRunStudioTests/RuntimeResidencyStateTests.swift" \
   "$repo_root/apps/ios/MereRunStudioTests/ChatTurnStoreTests.swift" \
   "$repo_root/apps/ios/MereRunStudioTests/RelaySnapshotStoreTests.swift" \
   "$repo_root/apps/ios/MereRunStudioTests/ModelDownloadOperationTests.swift" \
   "$repo_root/apps/ios/MereRunStudioTests/PendingModelDownloadStoreTests.swift" \
   "$repo_root/apps/ios/MereRunStudioTests/RuntimeWarmupTests.swift" \
   "$repo_root/apps/ios/MereRunStudioTests/RunPollingPolicyTests.swift" \
   "$repo_root/apps/ios/MereRunStudioTests/RunSnapshotPollerTests.swift" \
   "$test_package/Tests/"
cat > "$test_package/Package.swift" <<'SWIFT'
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "IOSAppStateTests",
    platforms: [.macOS(.v15)],
    targets: [.testTarget(name: "IOSAppStateTests", path: "Tests")]
)
SWIFT
swift test --package-path "$test_package" "$@"
