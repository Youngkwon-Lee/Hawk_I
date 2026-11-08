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

// Research-only hand-trace interchange with hawkeye-labeling-tool's
// analyze_finger_tapping_trace.py. This is 2D image-plane data, NOT 3D.
struct NormalizedTraceSample: Codable {
    let time_s: Double
    let gap_norm: Double
    let confidence: Double
}

struct TraceProvenance: Codable {
    let coordinate_space: String
    let normalization_reference: String
    let calibrated_physical_distance: Bool
    let side_verification: String
    let continuous_hand_identity_verified: Bool
}

struct NormalizedTrace: Codable {
    let task: String
    let side: String
    let unit: String
    let decoded_frames: Int
    let samples: [NormalizedTraceSample]
    let provenance: TraceProvenance
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

let arguments = Array(CommandLine.arguments.dropFirst())
let traceExportRequested = arguments.count == 5 &&
    arguments[1] == "--trace-json" && arguments[3] == "--side" &&
    (arguments[4] == "left" || arguments[4] == "right")
guard arguments.count == 1 || traceExportRequested else {
    fputs("usage: analyze_finger_tapping_cycles.swift VIDEO [--trace-json OUTPUT.json --side left|right]\n", stderr)
    exit(2)
}

let videoURL = URL(fileURLWithPath: arguments[0])
let traceURL = traceExportRequested ? URL(fileURLWithPath: arguments[2]) : nil
if let traceURL = traceURL {
    guard traceURL.pathExtension.lowercased() == "json",
          traceURL.standardizedFileURL != videoURL.standardizedFileURL,
          !FileManager.default.fileExists(atPath: traceURL.path) else {
        fputs("trace export must be a new local .json file, different from input\n", stderr)
        exit(2)
    }
}
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
var normalizedTraceSamples: [NormalizedTraceSample] = []
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

        // Unlike the legacy cycle summary, this optional export requires a
        // palm-span proxy and real presentation timestamps on every sample.
        // Pixel dimensions compensate for non-square normalized image axes;
        // the resulting ratio is NOT a calibrated 3D/physical distance.
        if traceURL != nil,
           let indexMCP = points[.indexMCP],
           let littleMCP = points[.littleMCP],
           indexMCP.confidence >= 0.30,
           littleMCP.confidence >= 0.30,
           time.isFinite, time >= 0 {
            let width = Double(CVPixelBufferGetWidth(pixelBuffer))
            let height = Double(CVPixelBufferGetHeight(pixelBuffer))
            let spanX = Double(indexMCP.location.x - littleMCP.location.x) * width
            let spanY = Double(indexMCP.location.y - littleMCP.location.y) * height
            let span = hypot(spanX, spanY)
            let gap = hypot(dx * width, dy * height)
            if span > 1e-6 {
                let ratio = gap / span
                if ratio.isFinite, ratio >= 0, ratio <= 3 {
                    let confidence = Double(min(thumb.confidence, index.confidence,
                                                indexMCP.confidence, littleMCP.confidence))
                    normalizedTraceSamples.append(NormalizedTraceSample(
                        time_s: time, gap_norm: ratio, confidence: confidence))
                }
            }
        }
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
if let traceURL = traceURL {
    let trace = NormalizedTrace(
        task: "finger_tapping", side: arguments[4],
        unit: "thumb_index_gap_over_palm_width",
        decoded_frames: decodedFrames, samples: normalizedTraceSamples,
        provenance: TraceProvenance(
            coordinate_space: "2d_image_plane",
            normalization_reference: "index_mcp_to_little_mcp_image_pixel_span",
            calibrated_physical_distance: false,
            side_verification: "operator_declared_unverified",
            continuous_hand_identity_verified: false))
    try encoder.encode(trace).write(to: traceURL, options: .withoutOverwriting)
}
FileHandle.standardOutput.write(try encoder.encode(result))
FileHandle.standardOutput.write(Data("\n".utf8))
