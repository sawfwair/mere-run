import XCTest
@testable import MereRunCore

final class ChatPrefixCacheTests: XCTestCase {
    func testChangedImageRejectsItsKVButRetainsAnEarlierCompatiblePrefix() {
        var cache = ChatNativePrefixCache<[String]>()
        cache.store(identity: "scope", tokens: [1, 2], value: [], ttl: 300)
        cache.store(identity: "scope", tokens: [1, 2, 9, 9, 3], value: ["old-pixels"], ttl: 300)
        let tokens = [1, 2, 9, 9, 3, 4]
        XCTAssertEqual(cache.lookup(identity: "scope", tokens: tokens, isCompatible: { images, count in
            images == (count > 2 ? ["old-pixels"] : [])
        })?.tokens.count, 5)
        XCTAssertEqual(cache.lookup(identity: "scope", tokens: tokens, isCompatible: { images, count in
            images == (count > 2 ? ["new-pixels"] : [])
        })?.tokens.count, 2)
        XCTAssertNil(cache.lookup(identity: "other-scope", tokens: tokens))
    }
    func testMovingAMarkerBackDoesNotReuseABeyondBoundaryPrefix() {
        var cache = ChatNativePrefixCache<String>()
        XCTAssertTrue(cache.store(identity: "scope", tokens: [1, 2, 3, 4], value: "long", ttl: 300))
        XCTAssertNil(cache.lookup(identity: "scope", tokens: [1, 2, 3, 4, 5], ttls: [2: 300]))
        XCTAssertTrue(cache.store(identity: "scope", tokens: [1, 2], value: "short", ttl: 300))
        XCTAssertEqual(cache.lookup(identity: "scope", tokens: [1, 2, 3, 4, 5], ttls: [2: 300])?.value, "short")
        XCTAssertEqual(cache.lookup(identity: "scope", tokens: [1, 2, 3, 4, 5])?.value, "long")
    }

    func testRealReadsRefreshTTLAndCanExtendAnExistingPrefixToOneHour() {
        let now = Date(timeIntervalSince1970: 1000)
        var cache = ChatNativePrefixCache<String>()
        XCTAssertTrue(cache.store(identity: "scope", tokens: [1, 2], value: "native", ttl: 300, now: now))
        XCTAssertNotNil(cache.lookup(identity: "scope", tokens: [1, 2, 3], now: now.addingTimeInterval(299)))
        XCTAssertNotNil(cache.lookup(identity: "scope", tokens: [1, 2, 4], now: now.addingTimeInterval(301)))
        XCTAssertNil(cache.lookup(identity: "scope", tokens: [1, 2], now: now.addingTimeInterval(602)))
        XCTAssertTrue(cache.store(identity: "scope", tokens: [1, 2], value: "native", ttl: 300, now: now))
        XCTAssertNotNil(cache.lookup(identity: "scope", tokens: [1, 2], ttls: [2: 3600], now: now.addingTimeInterval(2)))
        XCTAssertEqual(cache.entries.count, 1)
        XCTAssertNotNil(cache.lookup(identity: "scope", tokens: [1, 2], now: now.addingTimeInterval(1000)))
        XCTAssertNil(cache.lookup(identity: "scope", tokens: [1, 2], now: now.addingTimeInterval(4601)))
    }
    func testMarkerInsideAMessageReusesOnlyTheUnchangedPrefix() {
        let prepared = [1, 2, 3, 4, 5, 9]
        let partialMessage = [1, 2, 3, 9]
        let boundary = MuseGlimmerGenerator.cacheBoundaryTokenCount(prepared: prepared, renderedBoundary: partialMessage)
        var cache = ChatNativePrefixCache<String>()
        XCTAssertTrue(cache.store(identity: "scope", tokens: Array(prepared.prefix(boundary)), value: "native", ttl: 300))
        XCTAssertEqual(cache.lookup(identity: "scope", tokens: [1, 2, 3, 8, 9])?.tokens, [1, 2, 3])
        XCTAssertNil(cache.lookup(identity: "scope", tokens: [1, 2, 8, 4, 5, 9]))
    }
    func testNativeCacheCanRetainALargeClientToolPrefix() {
        var cache = ChatNativePrefixCache<String>(maximumTokens: MuseGlimmerGenerator.maximumCachedPrefixTokens)
        let tokens = Array(0..<(MuseGlimmerGenerator.maximumCachedPrefixTokens / 2 + 1))
        XCTAssertTrue(cache.store(identity: "client", tokens: tokens, value: "native", ttl: 300))
        XCTAssertEqual(cache.lookup(identity: "client", tokens: tokens + [42])?.tokens.count, tokens.count)
        XCTAssertTrue(cache.store(identity: "client", tokens: tokens + [42], value: "updated", ttl: 300))
        XCTAssertEqual(cache.entries.count, 1, "The aggregate token limit bounds retained GPU state")
        XCTAssertFalse(cache.store(identity: "client", tokens: Array(0...MuseGlimmerGenerator.maximumCachedPrefixTokens), value: "too large", ttl: 300))
    }
    func testReuseTTLIdentityAndEviction() {
        let now = Date(timeIntervalSince1970: 1000)
        var cache = ChatNativePrefixCache<String>(maximumEntries: 2, maximumTokens: 6)
        XCTAssertTrue(cache.store(identity: "scope/model/template", tokens: [1, 2], value: "native-a", ttl: 300, now: now))
        XCTAssertEqual(cache.lookup(identity: "scope/model/template", tokens: [1, 2, 3], now: now)?.value, "native-a")
        XCTAssertNil(cache.lookup(identity: "other/model/template", tokens: [1, 2, 3], now: now))
        XCTAssertNil(cache.lookup(identity: "scope/model/template", tokens: [1, 9, 3], now: now), "Context edits invalidate reuse at the changed token")
        XCTAssertNil(cache.lookup(identity: "scope/model/template", tokens: [1, 2], now: now.addingTimeInterval(300)))
        XCTAssertTrue(cache.store(identity: "a", tokens: [1, 2], value: "a", ttl: 3600, now: now))
        XCTAssertTrue(cache.store(identity: "b", tokens: [3, 4], value: "b", ttl: 3600, now: now))
        _ = cache.lookup(identity: "a", tokens: [1, 2], now: now)
        XCTAssertTrue(cache.store(identity: "c", tokens: [5, 6], value: "c", ttl: 3600, now: now))
        XCTAssertNil(cache.lookup(identity: "b", tokens: [3, 4], now: now))
        XCTAssertNotNil(cache.lookup(identity: "a", tokens: [1, 2], now: now))
        XCTAssertFalse(cache.store(identity: "too-large", tokens: Array(0...8), value: "no", ttl: 300, now: now))
        cache.removeAll(); XCTAssertTrue(cache.entries.isEmpty)
    }
}
