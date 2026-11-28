import Foundation
import MereRunCore
import MLX

public enum MiniMaxH3ModelOptimizer {
    public static func artifactURLs(resources: MiniMaxH3Resources) -> [URL] {
        [resources.adaLNCachePackIndexURL] + MiniMaxH3AdaLNCachePack.productionSchedules.map {
            resources.rootURL.appending(path: $0.filename)
        }
    }

    @discardableResult
    public static func optimize(
        resources: MiniMaxH3Resources,
        replacing: Bool = false,
        progressHandler: (@Sendable (Int, Int) -> Void)? = nil
    ) throws -> URL {
        let missing = resources.validate()
        guard missing.isEmpty else { throw MiniMaxH3GeneratorError.missingModelFiles(missing) }
        let configuration = try resources.loadConfiguration()
        let sourceIdentity = try resources.adaLNCacheSourceIdentity()
        let indexURL = resources.adaLNCachePackIndexURL
        if FileManager.default.fileExists(atPath: indexURL.path) {
            for schedule in MiniMaxH3AdaLNCachePack.productionSchedules {
                let schedules = try schedule.schedules()
                let selection = try MiniMaxH3AdaLNCachePack.load(
                    from: resources.rootURL,
                    configuration: .init(configuration),
                    videoSchedule: schedules.video,
                    audioSchedule: schedules.audio,
                    sourceIdentity: sourceIdentity
                )
                guard selection.exact else {
                    throw MiniMaxH3AdaLNCacheError.incompatible(
                        "exact production schedule is missing: \(schedule.filename)"
                    )
                }
            }
            if !replacing {
                return indexURL
            }
            if try resources.requiresAdaLNCache() {
                return indexURL
            }
        }
        if try resources.requiresAdaLNCache() {
            throw MiniMaxH3AdaLNCacheError.incompatible(
                "a pruned transformer cannot synthesize a missing exact cache pack"
            )
        }
        let transformer = try MiniMaxH3ModelLoader.loadTransformer(
            resources: resources,
            configuration: configuration
        )
        var entries: [MiniMaxH3AdaLNCachePack.Entry] = []
        for schedule in MiniMaxH3AdaLNCachePack.productionSchedules {
            let schedules = try schedule.schedules()
            let cache = transformer.precomputeAdaLN(
                videoSchedule: schedules.video,
                audioSchedule: schedules.audio,
                sourceIdentity: sourceIdentity,
                progressHandler: progressHandler
            )
            let url = resources.rootURL.appending(path: schedule.filename)
            try cache.save(to: url, replacing: replacing)
            entries.append(try MiniMaxH3AdaLNCachePack.entry(schedule: schedule, url: url))
        }
        return try MiniMaxH3AdaLNCachePack.write(
            entries: entries,
            sourceIdentity: sourceIdentity,
            to: resources.rootURL
        )
    }
}
