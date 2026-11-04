#!/usr/bin/env swift

import AVFoundation
import Foundation
import Vision

struct Sample {
    let time: Double
    let distance: Double
}

struct Result: Codable {
    let analysis: String
    let video: String
    let decodedFrames: Int
    let detectedFrames: Int
    let detectionCoverage: Double
    let durationSec: Double
    let distanceP10: Double?
    let distanceP50: Double?
    let distanceP90: Double?
    let estimatedOpenCloseCycles: Int?
    let estimateStatus: String
    let method: String
}

func quantile(_ values: [Double], _ probability: Double) -> Double? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    let position = Double(sorted.count - 1) * probability
    let lower = Int(floor(position))
    let upper = Int(ceil(position))
    if lower == upper { return sorted[lower] }
    let weight = position - Double(lower)
    return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

func movingAverage(_ values: [Double], radius: Int) -> [Double] {
    values.indices.map { index in
        let lower = max(0, index - radius)
        let upper = min(values.count - 1, index + radius)
        let window = values[lower...upper]
        return window.reduce(0, +) / Double(window.count)
    }
}

func countCycles(_ distances: [Double]) -> Int? {
    guard distances.count >= 20,
          let low = quantile(distances, 0.30),
          let high = quantile(distances, 0.70),
          high - low > 0.02 else { return nil }
    let smoothed = movingAverage(distances, radius: 2)
    enum State { case unknown, open, closed }
    var state = State.unknown
    var cycles = 0
    for distance in smoothed {
        if distance >= high {
            state = .open
        } else if distance <= low, state == .open {
            cycles += 1
            state = .closed
        } else if distance <= low, state == .unknown {
            state = .closed
        }
    }
    return cycles
}

guard CommandLine.arguments.count == 2 else {
    fputs("usage: analyze_finger_tapping_cycles.swift VIDEO\n", stderr)
    exit(2)
}

let videoURL = URL(fileURLWithPath: CommandLine.arguments[1])
let asset = AVURLAsset(url: videoURL)
guard let track = asset.tracks(withMediaType: .video).first else {
    fputs("video track not found\n", stderr)
    exit(2)
}
let reader = try AVAssetReader(asset: asset)
let output = AVAssetReaderTrackOutput(
    track: track,
    outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
    ]
)
output.alwaysCopiesSampleData = false
reader.add(output)
guard reader.startReading() else {
    throw reader.error ?? NSError(domain: "HawkeyeVision", code: 1)
}

let request = VNDetectHumanHandPoseRequest()
request.maximumHandCount = 1
var decodedFrames = 0
var samples: [Sample] = []
var durationSec = 0.0

while let sampleBuffer = output.copyNextSampleBuffer() {
    decodedFrames += 1
    let time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
    durationSec = max(durationSec, time)
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
    do {
        try handler.perform([request])
        guard let observation = request.results?.first else { continue }
        let points = try observation.recognizedPoints(.all)
        guard let thumb = points[.thumbTip],
              let index = points[.indexTip],
              thumb.confidence >= 0.30,
              index.confidence >= 0.30 else { continue }
        let dx = Double(thumb.location.x - index.location.x)
        let dy = Double(thumb.location.y - index.location.y)
        samples.append(Sample(time: time, distance: sqrt(dx * dx + dy * dy)))
    } catch {
        continue
    }
}

let distances = samples.map(\.distance)
let coverage = decodedFrames > 0 ? Double(samples.count) / Double(decodedFrames) : 0
let estimatedCycles = coverage >= 0.70 ? countCycles(distances) : nil
let status: String
if coverage < 0.70 {
    status = "insufficient_hand_detection"
} else if estimatedCycles == nil {
    status = "insufficient_motion_separation"
} else {
    status = "automated_qc_estimate_requires_human_confirmation"
}
let result = Result(
    analysis: "thumb-index open-close cycle estimate",
    video: videoURL.lastPathComponent,
    decodedFrames: decodedFrames,
    detectedFrames: samples.count,
    detectionCoverage: (coverage * 1000).rounded() / 1000,
    durationSec: (durationSec * 1000).rounded() / 1000,
    distanceP10: quantile(distances, 0.10).map { ($0 * 1000).rounded() / 1000 },
    distanceP50: quantile(distances, 0.50).map { ($0 * 1000).rounded() / 1000 },
    distanceP90: quantile(distances, 0.90).map { ($0 * 1000).rounded() / 1000 },
    estimatedOpenCloseCycles: estimatedCycles,
    estimateStatus: status,
    method: "macOS Vision hand pose; P30/P70 hysteresis on smoothed normalized thumb-index tip distance"
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
FileHandle.standardOutput.write(try encoder.encode(result))
FileHandle.standardOutput.write(Data("\n".utf8))
