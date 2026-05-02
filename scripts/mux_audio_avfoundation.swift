import AVFoundation
import Foundation

if CommandLine.arguments.count < 4 {
    fputs("usage: swift mux_audio_avfoundation.swift <video.mp4> <audio.mp3|m4a|wav> <output.mp4>\n", stderr)
    exit(2)
}

let videoURL = URL(fileURLWithPath: CommandLine.arguments[1])
let audioURL = URL(fileURLWithPath: CommandLine.arguments[2])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[3])

try? FileManager.default.removeItem(at: outputURL)

let videoAsset = AVURLAsset(url: videoURL)
let audioAsset = AVURLAsset(url: audioURL)
let composition = AVMutableComposition()

guard let sourceVideoTrack = try await videoAsset.loadTracks(withMediaType: .video).first else {
    fputs("missing video track\n", stderr)
    exit(1)
}

guard let compositionVideoTrack = composition.addMutableTrack(
    withMediaType: .video,
    preferredTrackID: kCMPersistentTrackID_Invalid
) else {
    fputs("cannot create composition video track\n", stderr)
    exit(1)
}

let videoDuration = try await videoAsset.load(.duration)
try compositionVideoTrack.insertTimeRange(
    CMTimeRange(start: .zero, duration: videoDuration),
    of: sourceVideoTrack,
    at: .zero
)
compositionVideoTrack.preferredTransform = try await sourceVideoTrack.load(.preferredTransform)

if let sourceAudioTrack = try await audioAsset.loadTracks(withMediaType: .audio).first,
   let compositionAudioTrack = composition.addMutableTrack(
    withMediaType: .audio,
    preferredTrackID: kCMPersistentTrackID_Invalid
   ) {
    let audioDuration = try await audioAsset.load(.duration)
    let insertedDuration = min(videoDuration, audioDuration)
    try compositionAudioTrack.insertTimeRange(
        CMTimeRange(start: .zero, duration: insertedDuration),
        of: sourceAudioTrack,
        at: .zero
    )
} else {
    fputs("missing audio track\n", stderr)
    exit(1)
}

guard let exporter = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
    fputs("cannot create exporter\n", stderr)
    exit(1)
}
exporter.outputURL = outputURL
exporter.outputFileType = .mp4
exporter.shouldOptimizeForNetworkUse = true

await exporter.export()

switch exporter.status {
case .completed:
    print("wrote \(outputURL.path)")
default:
    fputs("export failed: \(String(describing: exporter.error))\n", stderr)
    exit(1)
}
