#!/usr/bin/env swift
import AppKit
import Foundation

// Deterministic, synthetic diagnostics with explicit ground truth. These probes
// measure narrow behaviors; they are not a standardized benchmark.
let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
func image(_ name: String, draw: () -> Void) throws {
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 512, pixelsHigh: 512,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: 512, height: 512).fill()
    draw()
    NSGraphicsContext.restoreGraphicsState()
    try bitmap.representation(using: .png, properties: [:])!.write(to: root.appendingPathComponent(name))
}
func text(_ value: String, x: CGFloat, y: CGFloat, size: CGFloat = 40) {
    (value as NSString).draw(at: NSPoint(x: x, y: y), withAttributes: [
        .font: NSFont.monospacedSystemFont(ofSize: size, weight: .bold), .foregroundColor: NSColor.black
    ])
}
try image("ocr.png") { text("RIVER", x: 145, y: 280, size: 64); text("4827", x: 168, y: 180, size: 64) }
try image("count.png") {
    NSColor.systemRed.setFill()
    for x in [CGFloat(45), 205, 365] { NSBezierPath(ovalIn: NSRect(x: x, y: 320, width: 100, height: 100)).fill() }
    NSColor.systemBlue.setFill()
    NSRect(x: 200, y: 100, width: 110, height: 110).fill()
}
try image("spatial.png") {
    NSColor.systemBlue.setFill(); NSRect(x: 40, y: 200, width: 130, height: 130).fill()
    NSColor.systemRed.setFill(); NSBezierPath(ovalIn: NSRect(x: 330, y: 200, width: 130, height: 130)).fill()
}
try image("chart.png") {
    text("Units", x: 30, y: 460, size: 28)
    NSColor.black.setFill(); NSRect(x: 55, y: 80, width: 410, height: 3).fill()
    for (x, height, label, value) in [(100, 120, "A", "3"), (235, 280, "B", "7"), (370, 200, "C", "5")] {
        NSColor.systemBlue.setFill(); NSRect(x: x, y: 83, width: 70, height: height).fill()
        text(label, x: CGFloat(x + 20), y: 35, size: 30)
        text(value, x: CGFloat(x + 20), y: CGFloat(90 + height), size: 30)
    }
}
struct Probe: Encodable {
    let id: String
    let prompt: String
    var image: String? = nil
    var expected: String? = nil
    var maxTokens = 64
    var thinking = false
}
let probes: [Probe] = [
    .init(id: "smoke", prompt: "Reply with exactly READY and no other text.", expected: "READY", maxTokens: 16),
    .init(id: "arithmetic", prompt: "What is 17 * 23? Reply with only the integer.", expected: "391"),
    .init(id: "logic", prompt: "All glims are vors. No vors are tazes. Can any glim be a taze? Reply with only yes or no in lowercase.", expected: "no"),
    .init(id: "extraction", prompt: "Extract the order ID from: 'Customer Nora; order KX-7319; paid 42 dollars.' Reply with only the order ID.", expected: "KX-7319"),
    .init(id: "sorting", prompt: "Sort these integers in ascending order: 19, -3, 0, 7, -11. Reply with comma-separated integers and no spaces.", expected: "-11,-3,0,7,19"),
    .init(id: "instruction", prompt: "Return exactly the third word of this list: oak cedar maple birch pine. No punctuation or explanation.", expected: "maple"),
    .init(id: "translation", prompt: "Translate the French word 'chat' into English. Reply with only the lowercase English word.", expected: "cat"),
    .init(id: "reasoning", prompt: "A train travels 90 km in 1.5 hours, then 60 km in 1 hour. What is its average speed for the whole trip in km/h? Give the final answer as just a number.", expected: "60", maxTokens: 256, thinking: true),
    .init(id: "ocr", prompt: "Read both lines of text. Reply with the first line, one space, then the second line. No other text.", image: "ocr.png", expected: "RIVER 4827"),
    .init(id: "count", prompt: "How many red circles are in the image? Reply with only the integer.", image: "count.png", expected: "3"),
    .init(id: "spatial", prompt: "What color is the shape on the right? Reply with only the lowercase color name.", image: "spatial.png", expected: "red"),
    .init(id: "chart", prompt: "Which category has the highest value in the chart? Reply with only its single capital letter.", image: "chart.png", expected: "B"),
    .init(id: "decode-speed", prompt: "Write a numbered list of the integers from 1 to 100, one per line.", maxTokens: 128),
    .init(id: "decode-speed-repeat", prompt: "Write a numbered list of the integers from 1 to 100, one per line.", maxTokens: 128)
]
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
struct Manifest: Encodable { let cases: [Probe] }
try encoder.encode(Manifest(cases: probes)).write(to: root.appendingPathComponent("manifest.json"))
