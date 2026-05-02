import AppKit
import AVFoundation
import CoreVideo
import Foundation

if CommandLine.arguments.count < 5 {
    fputs("usage: swift encode_frames_avfoundation.swift <frames-dir> <output.mp4> <fps> <frame-count>\n", stderr)
    exit(2)
}

let framesDir = URL(fileURLWithPath: CommandLine.arguments[1])
let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let fps = Int32(CommandLine.arguments[3]) ?? 12
let frameCount = Int(CommandLine.arguments[4]) ?? 0
let width = 1080
let height = 1920

try? FileManager.default.removeItem(at: outputURL)

let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
let settings: [String: Any] = [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: width,
    AVVideoHeightKey: height,
    AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: 8_000_000,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
    ]
]

let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
input.expectsMediaDataInRealTime = false

let attrs: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
    kCVPixelBufferWidthKey as String: width,
    kCVPixelBufferHeightKey as String: height
]

let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)

guard writer.canAdd(input) else {
    fputs("cannot add video input\n", stderr)
    exit(1)
}
writer.add(input)

guard writer.startWriting() else {
    fputs("cannot start writer: \(String(describing: writer.error))\n", stderr)
    exit(1)
}
writer.startSession(atSourceTime: .zero)

func pixelBuffer(from imageURL: URL) -> CVPixelBuffer? {
    guard let nsImage = NSImage(contentsOf: imageURL) else { return nil }
    var rect = CGRect(origin: .zero, size: nsImage.size)
    guard let cgImage = nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil) else { return nil }

    var pxBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, nil, &pxBuffer)
    guard status == kCVReturnSuccess, let buffer = pxBuffer else { return nil }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    guard let context = CGContext(
        data: CVPixelBufferGetBaseAddress(buffer),
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
    ) else { return nil }

    context.interpolationQuality = .high
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    return buffer
}

for i in 0..<frameCount {
    while !input.isReadyForMoreMediaData {
        Thread.sleep(forTimeInterval: 0.002)
    }
    let frameURL = framesDir.appendingPathComponent(String(format: "frame_%04d.png", i))
    guard let buffer = pixelBuffer(from: frameURL) else {
        fputs("failed to read \(frameURL.path)\n", stderr)
        exit(1)
    }
    let time = CMTime(value: CMTimeValue(i), timescale: fps)
    if !adaptor.append(buffer, withPresentationTime: time) {
        fputs("append failed at frame \(i): \(String(describing: writer.error))\n", stderr)
        exit(1)
    }
    if i % 60 == 0 {
        print("encoded \(i)/\(frameCount)")
    }
}

input.markAsFinished()
writer.finishWriting {
    if writer.status == .completed {
        print("wrote \(outputURL.path)")
        exit(0)
    } else {
        fputs("writer failed: \(String(describing: writer.error))\n", stderr)
        exit(1)
    }
}

RunLoop.main.run()
