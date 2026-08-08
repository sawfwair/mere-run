import ArgumentParser

struct Geo: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "geo",
        abstract: "Run native geospatial inference models on local Earth-observation data.",
        subcommands: [GeoFlood.self, GeoFire.self, GeoTESSERA.self, GeoOlmoEarth.self]
    )
}
