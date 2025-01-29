//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import AVFoundation
import Accelerate
import CoreAudio
import CoreML

// Core Audio Device
#if os(macOS)
  public typealias DeviceID = AudioDeviceID
#else
  public typealias DeviceID = String
#endif

public struct AudioDevice: Identifiable, Hashable {
  public let id: DeviceID
  public let name: String
}

public protocol AudioProcessing {
  /// Loads audio data from a specified file path.
  /// - Parameters:
  ///   - audioFilePath: The file path of the audio file.
  ///   - startTime: Optional start time in seconds to read from
  ///   - endTime: Optional end time in seconds to read until
  /// - Returns: `AVAudioPCMBuffer` containing the audio data.
  static func loadAudio(
    fromPath audioFilePath: String, startTime: Double?, endTime: Double?,
    maxReadFrameSize: AVAudioFrameCount?
  ) throws -> AVAudioPCMBuffer

  /// Loads and converts audio data from a specified file paths.
  /// - Parameter audioPaths: The file paths of the audio files.
  /// - Returns: Array of `.success` if the file was loaded and converted correctly, otherwise `.failure`
  static func loadAudio(at audioPaths: [String]) async -> [Result<[Float], Swift.Error>]

  ///  Pad or trim the audio data to the desired length.
  /// - Parameters:
  ///   - audioArray:An array of audio frames to be padded or trimmed.
  ///   - startIndex: The index of the audio frame to start at.
  ///   - frameLength: The desired length of the audio data in frames.
  ///   - saveSegment: Save the segment to disk for debugging.
  /// - Returns: An optional `MLMultiArray` containing the audio data.
  static func padOrTrimAudio(
    fromArray audioArray: [Float],
    startAt startIndex: Int,
    toLength frameLength: Int,
    saveSegment: Bool
  ) -> MLMultiArray?

  /// Stores the audio samples to be transcribed
  var audioSamples: ContiguousArray<Float> { get }

  /// Empties the audio samples array, keeping the last `keep` samples
  func purgeAudioSamples(keepingLast keep: Int)

  /// Purges silence from the audio samples array, keeping 10 seconds before the silence and 5 last seconds of audio
  func purgeSilnceFromSamples(silenceLength: Int)

  /// A measure of current buffer's energy in dB normalized from 0 - 1 based on the quietest buffer's energy in a specified window
  var relativeEnergy: [Float] { get }

  /// How many past buffers of audio to use to calculate relative energy
  /// The lowest average energy value in the buffer within this amount of previous buffers will used as the silence baseline
  var relativeEnergyWindow: Int { get set }

  /// Starts recording audio from the specified input device, resetting the previous state
  func startRecordingLive(inputDeviceID: DeviceID?, callback: (([Float]) -> Void)?) throws

  /// Pause recording
  func pauseRecording()

  /// Stops recording and cleans up resources
  func stopRecording()

  /// Resume recording audio from the specified input device, appending to continuous `audioArray` after pause
  func resumeRecordingLive(inputDeviceID: DeviceID?, callback: (([Float]) -> Void)?) throws

}

/// Overrideable default methods for AudioProcessing
extension AudioProcessing {
  /// Loads and converts audio data from a specified file paths.
  /// - Parameter audioPaths: The file paths of the audio files.
  /// - Returns: `AVAudioPCMBuffer` containing the audio data.
  @available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
  public static func loadAudioAsync(fromPath audioFilePath: String) async throws -> AVAudioPCMBuffer
  {
    return try await Task {
      try AudioProcessor.loadAudio(fromPath: audioFilePath)
    }.value
  }

  public func startRecordingLive(inputDeviceID: DeviceID? = nil, callback: (([Float]) -> Void)?)
    throws
  {
    try startRecordingLive(inputDeviceID: inputDeviceID, callback: callback)
  }

  public func resumeRecordingLive(inputDeviceID: DeviceID? = nil, callback: (([Float]) -> Void)?)
    throws
  {
    try resumeRecordingLive(inputDeviceID: inputDeviceID, callback: callback)
  }

  public static func padOrTrimAudio(
    fromArray audioArray: [Float], startAt startIndex: Int = 0, toLength frameLength: Int = 480_000,
    saveSegment: Bool = false
  ) -> MLMultiArray? {
    guard startIndex >= 0, startIndex < audioArray.count else {
      Logging.error("startIndex is outside the buffer size")
      return nil
    }

    let endFrameIndex = min(audioArray.count, startIndex + frameLength)
    let actualFrameLength = endFrameIndex - startIndex

    let audioSamplesArray = try! MLMultiArray(
      shape: [NSNumber(value: frameLength)], dataType: .float32)

    if actualFrameLength > 0 {
      audioArray.withUnsafeBufferPointer { sourcePointer in
        audioSamplesArray.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: 0)
          .initialize(
            from: sourcePointer.baseAddress!.advanced(by: startIndex), count: actualFrameLength)
      }
    }

    // If the buffer is smaller than the desired frameLength, pad the rest with zeros using vDSP
    if actualFrameLength < frameLength {
      vDSP_vclr(
        audioSamplesArray.dataPointer.assumingMemoryBound(to: Float.self).advanced(
          by: actualFrameLength), 1, vDSP_Length(frameLength - actualFrameLength))
    }

    // Save the file for debugging
    if saveSegment {
      let desiredFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
      let newBuffer = AVAudioPCMBuffer(
        pcmFormat: desiredFormat, frameCapacity: AVAudioFrameCount(frameLength))!
      for i in 0..<frameLength {
        newBuffer.floatChannelData?[0][i] = audioSamplesArray[i].floatValue
      }
      newBuffer.frameLength = AVAudioFrameCount(frameLength)

      // Save to file
      let filename = "segment_\(Double(startIndex) / 16000.0).wav"
      var fileURL = getDownloadsDirectory().appendingPathComponent("WhisperKitSegments")
      fileURL = fileURL.appendingPathComponent(filename)
      do {
        try saveBuffer(newBuffer, to: fileURL)
        Logging.debug("Saved audio segment to \(fileURL)")
      } catch {
        Logging.debug("Could not save file: \(error)")
      }
    }

    return audioSamplesArray
  }

  public static func createWaveformImage(fromAudio audioBuffer: [Float], withSize size: CGSize)
    -> CGImage
  {
    let width = Int(size.width)
    let height = Int(size.height)
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    let bitsPerComponent = 8
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo =
      CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue

    var data = audioBuffer
    let dataProvider = CGDataProvider(
      data: NSData(bytes: &data, length: data.count * MemoryLayout<Float>.size))
    let cgImage = CGImage(
      width: width,
      height: height,
      bitsPerComponent: bitsPerComponent,
      bitsPerPixel: bytesPerPixel * bitsPerComponent,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
      provider: dataProvider!,
      decode: nil,
      shouldInterpolate: false,
      intent: CGColorRenderingIntent.defaultIntent
    )!
    return cgImage
  }
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class AudioProcessor: NSObject, AudioProcessing {
  private var lastInputDevice: DeviceID?
  private var currentTap: ProcessTapProtocol?
  private let processingQueue = DispatchQueue(
    label: "com.whisperkit.audioProcessor.processingQueue", qos: .userInitiated)
  private var accumulationBuffer: AVAudioPCMBuffer?

  public var audioEngine: AVAudioEngine?

  // Add serial queue for thread-safe access
  private let audioQueue = DispatchQueue(label: "com.whisperkit.audioprocessor.audioSampleQueue")
  private var _audioSamples: ContiguousArray<Float> = []
  public var audioSamples: ContiguousArray<Float> {
    get {
      audioQueue.sync { _audioSamples }
    }
    set {
      audioQueue.sync { _audioSamples = newValue }
    }
  }

  public var audioEnergy: [(rel: Float, avg: Float, max: Float, min: Float)] = []
  public var relativeEnergyWindow: Int = 20
  public var relativeEnergy: [Float] {
    return self.audioEnergy.map { $0.rel }
  }

  public var audioBufferCallback: (([Float]) -> Void)?
  public var minBufferLength = Int(Double(WhisperKit.sampleRate) * 0.1)  // 0.1 second of audio at 16,000 Hz

  // MARK: - Loading and conversion

  public static func loadAudio(
    fromPath audioFilePath: String,
    startTime: Double? = 0,
    endTime: Double? = nil,
    maxReadFrameSize: AVAudioFrameCount? = nil
  ) throws -> AVAudioPCMBuffer {
    guard FileManager.default.fileExists(atPath: audioFilePath) else {
      throw WhisperError.loadAudioFailed("Resource path does not exist \(audioFilePath)")
    }

    let audioFileURL = URL(fileURLWithPath: audioFilePath)
    let audioFile = try AVAudioFile(
      forReading: audioFileURL, commonFormat: .pcmFormatFloat32, interleaved: false)
    return try loadAudio(
      fromFile: audioFile, startTime: startTime, endTime: endTime,
      maxReadFrameSize: maxReadFrameSize)
  }

  public static func loadAudio(
    fromFile audioFile: AVAudioFile,
    startTime: Double? = 0,
    endTime: Double? = nil,
    maxReadFrameSize: AVAudioFrameCount? = nil
  ) throws -> AVAudioPCMBuffer {
    let sampleRate = audioFile.fileFormat.sampleRate
    let channelCount = audioFile.fileFormat.channelCount
    let frameLength = AVAudioFrameCount(audioFile.length)

    // Calculate the frame range based on the start and end seconds
    let startFrame = AVAudioFramePosition((startTime ?? 0) * sampleRate)
    let endFrame: AVAudioFramePosition
    if let end = endTime {
      endFrame = min(AVAudioFramePosition(end * sampleRate), AVAudioFramePosition(audioFile.length))
    } else {
      endFrame = AVAudioFramePosition(audioFile.length)
    }

    let frameCount = AVAudioFrameCount(endFrame - startFrame)

    // Seek to the start frame
    audioFile.framePosition = startFrame

    var outputBuffer: AVAudioPCMBuffer?

    // If the audio file already meets the desired format, read directly into the output buffer
    if sampleRate == 16000 && channelCount == 1 {
      guard
        let buffer = AVAudioPCMBuffer(
          pcmFormat: audioFile.processingFormat, frameCapacity: frameCount)
      else {
        throw WhisperError.loadAudioFailed("Unable to create audio buffer")
      }
      do {
        try audioFile.read(into: buffer, frameCount: frameCount)
      } catch {
        throw WhisperError.loadAudioFailed("Failed to read audio file: \(error)")
      }
      outputBuffer = buffer
    } else {
      // Audio needs resampling to 16khz
      let maxReadFrameSize = maxReadFrameSize ?? Constants.defaultAudioReadFrameSize
      outputBuffer = resampleAudio(
        fromFile: audioFile, toSampleRate: 16000, channelCount: 1, frameCount: frameCount,
        maxReadFrameSize: maxReadFrameSize)
    }

    if let outputBuffer = outputBuffer {
      Logging.debug(
        "Audio source details - Sample Rate: \(sampleRate) Hz, Channel Count: \(channelCount), Frame Length: \(frameLength), Duration: \(Double(frameLength) / sampleRate)s"
      )
      Logging.debug(
        "Audio buffer details - Sample Rate: \(outputBuffer.format.sampleRate) Hz, Channel Count: \(outputBuffer.format.channelCount), Frame Length: \(outputBuffer.frameLength), Duration: \(Double(outputBuffer.frameLength) / outputBuffer.format.sampleRate)s"
      )

      logCurrentMemoryUsage("After loadAudio function")

      return outputBuffer
    } else {
      throw WhisperError.loadAudioFailed("Failed to process audio buffer")
    }
  }

  public static func loadAudioAsFloatArray(
    fromPath audioFilePath: String,
    startTime: Double? = 0,
    endTime: Double? = nil
  ) throws -> [Float] {
    guard FileManager.default.fileExists(atPath: audioFilePath) else {
      throw WhisperError.loadAudioFailed("Resource path does not exist \(audioFilePath)")
    }

    let audioFileURL = URL(fileURLWithPath: audioFilePath)
    let audioFile = try AVAudioFile(
      forReading: audioFileURL, commonFormat: .pcmFormatFloat32, interleaved: false)
    let inputSampleRate = audioFile.fileFormat.sampleRate
    let inputFrameCount = AVAudioFrameCount(audioFile.length)
    let inputDuration = Double(inputFrameCount) / inputSampleRate

    let start = startTime ?? 0
    let end = min(endTime ?? inputDuration, inputDuration)

    // Load 10m of audio at a time to reduce peak memory while converting
    // Particularly impactful for large audio files
    let chunkDuration: Double = 60 * 10
    var currentTime = start
    var result: [Float] = []

    while currentTime < end {
      let chunkEnd = min(currentTime + chunkDuration, end)

      try autoreleasepool {
        let buffer = try loadAudio(
          fromFile: audioFile,
          startTime: currentTime,
          endTime: chunkEnd
        )

        let floatArray = Self.convertBufferToArray(buffer: buffer)
        result.append(contentsOf: floatArray)
      }

      currentTime = chunkEnd
    }

    return result
  }

  public static func loadAudio(at audioPaths: [String]) async -> [Result<[Float], Swift.Error>] {
    await withTaskGroup(of: [(index: Int, result: Result<[Float], Swift.Error>)].self) {
      taskGroup -> [Result<[Float], Swift.Error>] in
      for (index, audioPath) in audioPaths.enumerated() {
        taskGroup.addTask {
          do {
            let audio = try AudioProcessor.loadAudioAsFloatArray(fromPath: audioPath)
            return [(index: index, result: .success(audio))]
          } catch {
            return [(index: index, result: .failure(error))]
          }
        }
      }
      var batchResult = [(index: Int, result: Result<[Float], Swift.Error>)]()
      for await result in taskGroup {
        batchResult.append(contentsOf: result)
      }
      batchResult.sort(by: { $0.index < $1.index })
      return batchResult.map { $0.result }
    }
  }

  /// Resamples audio from a file to a specified sample rate and channel count.
  /// - Parameters:
  ///   - audioFile: The input audio file.
  ///   - sampleRate: The desired output sample rate.
  ///   - channelCount: The desired output channel count.
  ///   - frameCount: The desired frames to read from the input audio file. (default: all).
  ///   - maxReadFrameSize: Maximum number of frames to read at once (default: 10 million).
  /// - Returns: Resampled audio as an AVAudioPCMBuffer, or nil if resampling fails.
  public static func resampleAudio(
    fromFile audioFile: AVAudioFile,
    toSampleRate sampleRate: Double,
    channelCount: AVAudioChannelCount,
    frameCount: AVAudioFrameCount? = nil,
    maxReadFrameSize: AVAudioFrameCount = Constants.defaultAudioReadFrameSize
  ) -> AVAudioPCMBuffer? {
    let inputSampleRate = audioFile.fileFormat.sampleRate
    let inputStartFrame = audioFile.framePosition
    let inputFrameCount = frameCount ?? AVAudioFrameCount(audioFile.length)
    let inputDuration = Double(inputFrameCount) / inputSampleRate
    let endFramePosition = min(
      inputStartFrame + AVAudioFramePosition(inputFrameCount), audioFile.length + 1)

    guard
      let outputFormat = AVAudioFormat(
        standardFormatWithSampleRate: sampleRate, channels: channelCount)
    else {
      Logging.error("Failed to create output audio format")
      return nil
    }

    Logging.debug("Resampling \(String(format: "%.2f", inputDuration)) seconds of audio")

    // Create the output buffer with full capacity
    guard
      let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: AVAudioFrameCount(inputDuration * outputFormat.sampleRate))
    else {
      Logging.error("Failed to create output buffer")
      return nil
    }

    let inputBuffer = AVAudioPCMBuffer(
      pcmFormat: audioFile.processingFormat, frameCapacity: maxReadFrameSize)!
    var nextPosition = inputStartFrame
    while nextPosition < endFramePosition {
      let framePosition = audioFile.framePosition
      let remainingFrames = AVAudioFrameCount(endFramePosition - framePosition)
      let framesToRead = min(remainingFrames, maxReadFrameSize)
      nextPosition = framePosition + Int64(framesToRead)

      let currentPositionInSeconds = Double(framePosition) / inputSampleRate
      let nextPositionInSeconds = Double(nextPosition) / inputSampleRate
      Logging.debug(
        "Resampling \(String(format: "%.2f", currentPositionInSeconds))s - \(String(format: "%.2f", nextPositionInSeconds))s"
      )

      do {
        try audioFile.read(into: inputBuffer, frameCount: framesToRead)
        guard
          let resampledChunk = resampleAudio(
            fromBuffer: inputBuffer,
            toSampleRate: outputFormat.sampleRate,
            channelCount: outputFormat.channelCount)
        else {
          Logging.error("Failed to resample audio chunk")
          return nil
        }

        // Append the resampled chunk to the output buffer
        guard outputBuffer.appendContents(of: resampledChunk) else {
          Logging.error("Failed to append audio chunk")
          return nil
        }
      } catch {
        Logging.error("Error reading audio file: \(error)")
        return nil
      }
    }

    return outputBuffer
  }

  /// Resamples an audio buffer to a specified sample rate and channel count.
  /// - Parameters:
  ///   - inputBuffer: The input audio buffer.
  ///   - sampleRate: The desired output sample rate.
  ///   - channelCount: The desired output channel count.
  /// - Returns: Resampled audio as an AVAudioPCMBuffer, or nil if resampling fails.
  public static func resampleAudio(
    fromBuffer inputBuffer: AVAudioPCMBuffer, toSampleRate sampleRate: Double,
    channelCount: AVAudioChannelCount
  ) -> AVAudioPCMBuffer? {
    guard
      let outputFormat = AVAudioFormat(
        standardFormatWithSampleRate: sampleRate, channels: channelCount)
    else {
      Logging.error("Failed to create output audio format")
      return nil
    }

    guard let converter = AVAudioConverter(from: inputBuffer.format, to: outputFormat) else {
      Logging.error("Failed to create audio converter")
      return nil
    }

    do {
      return try Self.resampleBuffer(inputBuffer, with: converter)
    } catch {
      Logging.error("Failed to resample buffer: \(error)")
      return nil
    }
  }

  /// Resamples an audio buffer using the provided converter.
  /// - Parameters:
  ///   - buffer: The input audio buffer.
  ///   - converter: The audio converter to use for resampling.
  /// - Returns: Resampled audio as an AVAudioPCMBuffer.
  /// - Throws: WhisperError if resampling fails.
  public static func resampleBuffer(_ buffer: AVAudioPCMBuffer, with converter: AVAudioConverter)
    throws -> AVAudioPCMBuffer
  {
    var capacity =
      converter.outputFormat.sampleRate * Double(buffer.frameLength)
      / converter.inputFormat.sampleRate

    // Check if the capacity is a whole number
    if capacity.truncatingRemainder(dividingBy: 1) != 0 {
      // Round to the nearest whole number, which is non-zero
      let roundedCapacity = max(1, capacity.rounded(.toNearestOrEven))
      Logging.debug(
        "Rounding buffer frame capacity from \(capacity) to \(roundedCapacity) to better fit new sample rate"
      )
      capacity = roundedCapacity
    }

    guard
      let convertedBuffer = AVAudioPCMBuffer(
        pcmFormat: converter.outputFormat,
        frameCapacity: AVAudioFrameCount(capacity)
      )
    else {
      throw WhisperError.audioProcessingFailed("Failed to create converted buffer")
    }

    let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
      if buffer.frameLength == 0 {
        outStatus.pointee = .endOfStream
        return nil
      } else {
        outStatus.pointee = .haveData
        return buffer
      }
    }

    var error: NSError?
    let status = converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)

    if status == .error, let conversionError = error {
      throw WhisperError.audioProcessingFailed("Error converting audio: \(conversionError)")
    }

    return convertedBuffer
  }

  // MARK: - Utility

  /// Detect voice activity in the given buffer of relative energy values.
  /// - Parameters:
  ///   - relativeEnergy: relative energy values
  ///   - nextBufferInSeconds: duration of the next buffer in seconds
  ///   - energyValuesToConsider: number of energy values to consider
  ///   - silenceThreshold: silence threshold
  /// - Returns: true if voice is detected, false otherwise
  public static func isVoiceDetected(
    in relativeEnergy: [Float],
    nextBufferInSeconds: Float,
    silenceThreshold: Float
  ) -> Bool {
    // Calculate the number of energy values to consider based on the duration of the next buffer
    // Each energy value corresponds to 1 buffer length (100ms of audio), hence we divide by 0.1
    let energyValuesToConsider = max(0, Int(nextBufferInSeconds / 0.1))

    // Extract the relevant portion of energy values from the currentRelativeEnergy array
    let nextBufferEnergies = relativeEnergy.suffix(energyValuesToConsider)

    // Determine the number of energy values to check for voice presence
    // Considering up to the last 1 second of audio, which translates to 10 energy values
    let numberOfValuesToCheck = max(10, nextBufferEnergies.count - 10)

    // Check if any of the energy values in the considered range exceed the silence threshold
    // This indicates the presence of voice in the buffer
    return nextBufferEnergies.prefix(numberOfValuesToCheck).contains { $0 > silenceThreshold }
  }

  /// Calculate non-silent chunks of an audio.
  /// - Parameter signal: audio signal
  /// - Returns: an array of tuples indicating the start and end indices of non-silent chunks
  public static func calculateNonSilentChunks(
    in signal: [Float]
  ) -> [(startIndex: Int, endIndex: Int)] {
    EnergyVAD().calculateActiveChunks(in: signal)
  }

  /// Calculate voice activity in chunks of an audio based on energy threshold.
  /// - Parameters:
  ///   - signal: Audio signal
  ///   - chunkCount: Number of chunks
  ///   - frameLengthSamples: Frame length in samples
  ///   - frameOverlapSamples: frame overlap in samples, this is helpful to catch large energy values at the very end of a frame
  ///   - energyThreshold: Energy threshold for silence detection, default is 0.05. Chunks with energy below this threshold are considered silent.
  /// - Returns: An array of booleans indicating whether each chunk is non-silent
  public static func calculateVoiceActivityInChunks(
    of signal: [Float],
    chunkCount: Int,
    frameLengthSamples: Int,
    frameOverlapSamples: Int = 0,
    energyThreshold: Float = 0.022
  ) -> [Bool] {
    var chunkEnergies = [Float]()
    for chunkIndex in 0..<chunkCount {
      let startIndex = chunkIndex * frameLengthSamples
      let endIndex = min(startIndex + frameLengthSamples + frameOverlapSamples, signal.count)
      let chunk = Array(signal[startIndex..<endIndex])
      let avgEnergy = calculateAverageEnergy(of: chunk)
      chunkEnergies.append(avgEnergy)
    }

    let vadResult = chunkEnergies.map { $0 > energyThreshold }

    return vadResult
  }

  /// Calculate average energy of a signal chunk.
  /// - Parameter signal: Chunk of audio signal.
  /// - Returns: Average (RMS) energy of the signal chunk.
  public static func calculateAverageEnergy(of signal: [Float]) -> Float {
    var rmsEnergy: Float = 0.0
    vDSP_rmsqv(signal, 1, &rmsEnergy, vDSP_Length(signal.count))
    return rmsEnergy
  }

  /// Calculate energy of a signal chunk.
  /// - Parameter signal: Chunk of audio signal.
  /// - Returns: Tuple containing average (RMS energy), maximum, and minimum values.
  public static func calculateEnergy(of signal: [Float]) -> (avg: Float, max: Float, min: Float) {
    var rmsEnergy: Float = 0.0
    var minEnergy: Float = 0.0
    var maxEnergy: Float = 0.0

    // Calculate the root mean square of the signal
    vDSP_rmsqv(signal, 1, &rmsEnergy, vDSP_Length(signal.count))

    // Calculate the maximum sample value of the signal
    vDSP_maxmgv(signal, 1, &maxEnergy, vDSP_Length(signal.count))

    // Calculate the minimum sample value of the signal
    vDSP_minmgv(signal, 1, &minEnergy, vDSP_Length(signal.count))

    return (rmsEnergy, maxEnergy, minEnergy)
  }

  public static func calculateRelativeEnergy(of signal: [Float], relativeTo reference: Float?)
    -> Float
  {
    let signalEnergy = calculateAverageEnergy(of: signal)

    // Make sure reference is greater than 0
    // Default 1e-3 measured empirically in a silent room
    let referenceEnergy = max(1e-8, reference ?? 1e-3)

    // Convert to dB
    let dbEnergy = 20 * log10(signalEnergy)
    let refEnergy = 20 * log10(referenceEnergy)

    // Normalize based on reference
    // NOTE: since signalEnergy elements are floats from 0 to 1, max (full volume) is always 0dB
    let normalizedEnergy = rescale(value: dbEnergy, min: refEnergy, max: 0)

    // Clamp from 0 to 1
    return max(0, min(normalizedEnergy, 1))
  }

  public static func convertBufferToArray(buffer: AVAudioPCMBuffer, chunkSize: Int = 1024)
    -> [Float]
  {
    guard let channelData = buffer.floatChannelData else {
      return []
    }

    let frameLength = Int(buffer.frameLength)
    let startPointer = channelData[0]

    var result: [Float] = []
    result.reserveCapacity(frameLength)  // Reserve the capacity to avoid multiple allocations

    var currentFrame = 0
    while currentFrame < frameLength {
      let remainingFrames = frameLength - currentFrame
      let currentChunkSize = min(chunkSize, remainingFrames)

      var chunk = [Float](repeating: 0, count: currentChunkSize)

      chunk.withUnsafeMutableBufferPointer { bufferPointer in
        vDSP_mmov(
          startPointer.advanced(by: currentFrame),
          bufferPointer.baseAddress!,
          vDSP_Length(currentChunkSize),
          1,
          vDSP_Length(currentChunkSize),
          1
        )
      }

      result.append(contentsOf: chunk)
      currentFrame += currentChunkSize

      memset(
        startPointer.advanced(by: currentFrame - currentChunkSize), 0,
        currentChunkSize * MemoryLayout<Float>.size)
    }

    return result
  }

  public static func requestRecordPermission() async -> Bool {
    if #available(macOS 14, iOS 17, *) {
      return await AVAudioApplication.requestRecordPermission()
    } else {
      #if os(watchOS)
        // watchOS does not support AVCaptureDevice
        return true
      #else
        let microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        switch microphoneStatus {
        case .notDetermined:
          return await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { granted in
              continuation.resume(returning: granted)
            }
          }
        case .restricted, .denied:
          Logging.error("Microphone access denied")
          return false
        case .authorized:
          return true
        @unknown default:
          Logging.error("Unknown authorization status")
          return false
        }
      #endif
    }
  }

  #if os(macOS)
    public static func getAudioDevices() -> [AudioDevice] {
      var devices = [AudioDevice]()

      var propertySize: UInt32 = 0
      var status: OSStatus = noErr

      // Get the number of devices
      var propertyAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
      )
      status = AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0,
        nil,
        &propertySize
      )
      if status != noErr {
        Logging.error("Error: Unable to get the number of audio devices.")
        return devices
      }

      // Get the device IDs
      let deviceCount = Int(propertySize) / MemoryLayout<AudioDeviceID>.size
      var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
      status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &propertyAddress,
        0,
        nil,
        &propertySize,
        &deviceIDs
      )
      if status != noErr {
        Logging.error("Error: Unable to get the audio device IDs.")
        return devices
      }

      // Get device info for each device
      for deviceID in deviceIDs {
        var deviceName = ""
        var inputChannels = 0

        // Get device name
        var propertySize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var name: Unmanaged<CFString>?
        propertyAddress.mSelector = kAudioDevicePropertyDeviceNameCFString

        status = AudioObjectGetPropertyData(
          deviceID,
          &propertyAddress,
          0,
          nil,
          &propertySize,
          &name
        )
        if status == noErr, let deviceNameCF = name?.takeRetainedValue() as String? {
          deviceName = deviceNameCF
        }

        // Get input channels
        propertyAddress.mSelector = kAudioDevicePropertyStreamConfiguration
        propertyAddress.mScope = kAudioDevicePropertyScopeInput
        status = AudioObjectGetPropertyDataSize(deviceID, &propertyAddress, 0, nil, &propertySize)
        if status == noErr {
          let bufferListPointer = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
          defer { bufferListPointer.deallocate() }
          status = AudioObjectGetPropertyData(
            deviceID, &propertyAddress, 0, nil, &propertySize, bufferListPointer)
          if status == noErr {
            let bufferList = UnsafeMutableAudioBufferListPointer(bufferListPointer)
            for buffer in bufferList {
              inputChannels += Int(buffer.mNumberChannels)
            }
          }
        }

        if inputChannels > 0 {
          devices.append(AudioDevice(id: deviceID, name: deviceName))
        }
      }

      return devices
    }
  #endif

  deinit {
    stopRecording()
  }
}

// MARK: - Streaming

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
extension AudioProcessor {
  /// We have a new buffer, process and store it.
  /// NOTE: Assumes audio is 16khz mono
  public func processBuffer(_ buffer: [Float]) {
    audioQueue.sync {
      _audioSamples.append(contentsOf: buffer)
    }

    // Find the lowest average energy of the last 20 buffers ~2 seconds
    let minAvgEnergy = self.audioEnergy.suffix(20).reduce(Float.infinity) { min($0, $1.avg) }
    let relativeEnergy = Self.calculateRelativeEnergy(of: buffer, relativeTo: minAvgEnergy)

    // Update energy for buffers with valid data
    let signalEnergy = Self.calculateEnergy(of: buffer)
    let newEnergy = (relativeEnergy, signalEnergy.avg, signalEnergy.max, signalEnergy.min)
    self.audioEnergy.append(newEnergy)

    // Call the callback with the new buffer
    audioBufferCallback?(buffer)

    // Print the current size of the audio buffer
    if self.audioSamples.count % (minBufferLength * Int(relativeEnergyWindow)) == 0 {
      Logging.debug(
        "Current audio size: \(self.audioSamples.count) samples, most recent buffer: \(buffer.count) samples, most recent energy: \(newEnergy)"
      )
    }
  }

  #if os(macOS)
    func assignAudioInput(inputNode: AVAudioInputNode, inputDeviceID: AudioDeviceID) {
      guard let audioUnit = inputNode.audioUnit else {
        Logging.error("Failed to access the audio unit of the input node.")
        return
      }

      var inputDeviceID = inputDeviceID

      let error = AudioUnitSetProperty(
        audioUnit,
        kAudioOutputUnitProperty_CurrentDevice,
        kAudioUnitScope_Global,
        0,
        &inputDeviceID,
        UInt32(MemoryLayout<AudioDeviceID>.size)
      )

      if error != noErr {
        Logging.error("Error setting Audio Unit property: \(error)")
      } else {
        Logging.info("Successfully set input device.")
      }
    }
  #endif

  /// Attempts to setup the shared audio session if available on the device's OS
  public func setupAudioSessionForDevice() throws {
    #if !os(macOS)  // AVAudioSession is not available on macOS

      #if !os(watchOS)  // watchOS does not support .defaultToSpeaker
        let options: AVAudioSession.CategoryOptions = [.defaultToSpeaker, .allowBluetooth]
      #else
        let options: AVAudioSession.CategoryOptions = .mixWithOthers
      #endif

      let audioSession = AVAudioSession.sharedInstance()
      do {
        try audioSession.setCategory(.playAndRecord, options: options)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
      } catch let error as NSError {
        throw WhisperError.audioProcessingFailed("Failed to set up audio session: \(error)")
      }
    #endif
  }

  public func setupEngine(inputDeviceID: DeviceID? = nil) throws -> AVAudioEngine {
    let audioEngine = AVAudioEngine()
    let inputNode = audioEngine.inputNode

    #if os(macOS)
      if let inputDeviceID = inputDeviceID {
        assignAudioInput(inputNode: inputNode, inputDeviceID: inputDeviceID)
      }
    #endif

    let hardwareSampleRate = audioEngine.inputNode.inputFormat(forBus: 0).sampleRate
    let inputFormat = inputNode.outputFormat(forBus: 0)

    guard
      let nodeFormat = AVAudioFormat(
        commonFormat: inputFormat.commonFormat, sampleRate: hardwareSampleRate,
        channels: inputFormat.channelCount, interleaved: inputFormat.isInterleaved)
    else {
      throw WhisperError.audioProcessingFailed("Failed to create node format")
    }

    // Desired format (16,000 Hz, 1 channel)
    guard
      let desiredFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(WhisperKit.sampleRate),
        channels: AVAudioChannelCount(1), interleaved: false)
    else {
      throw WhisperError.audioProcessingFailed("Failed to create desired format")
    }

    guard let converter = AVAudioConverter(from: nodeFormat, to: desiredFormat) else {
      throw WhisperError.audioProcessingFailed("Failed to create audio converter")
    }

    let eq = AVAudioUnitEQ(numberOfBands: 2)

    // 🎛️ Configure the EQ for noise reduction
    configureEQ(eq)

    // 🔌 Connect the nodes: input → EQ → main engine
    audioEngine.attach(eq)
    audioEngine.connect(inputNode, to: eq, format: nodeFormat)
    audioEngine.connect(eq, to: audioEngine.mainMixerNode, format: nodeFormat)

    let bufferSize = AVAudioFrameCount(minBufferLength)  // 100ms - 400ms supported
    eq.installTap(onBus: 0, bufferSize: bufferSize, format: nodeFormat) {
      [weak self] (buffer: AVAudioPCMBuffer, _: AVAudioTime) in
      guard let self = self else { return }
      var buffer = buffer
      if !buffer.format.sampleRate.isEqual(to: Double(WhisperKit.sampleRate)) {
        do {
          buffer = try Self.resampleBuffer(buffer, with: converter)
        } catch {
          Logging.error("Failed to resample buffer: \(error)")
          return
        }
      }

      let newBufferArray = Self.convertBufferToArray(buffer: buffer)
      self.processBuffer(newBufferArray)
    }

    audioEngine.prepare()
    try audioEngine.start()

    return audioEngine
  }

  private func configureEQ(_ eq: AVAudioUnitEQ) {
    eq.globalGain = 0.0  // No overall gain change

    // 🎚 Band 1: High-pass filter (Removes low-frequency hum)
    let lowCutFilter = eq.bands[0]
    lowCutFilter.filterType = .highPass
    lowCutFilter.frequency = 100.0  // Removes noise below 100Hz
    lowCutFilter.bandwidth = 1.0
    lowCutFilter.gain = 0.0
    lowCutFilter.bypass = false

    // 🎚 Band 2: Mid-frequency noise reduction (typing, clicks)
    let midNoiseReduction = eq.bands[1]
    midNoiseReduction.filterType = .parametric
    midNoiseReduction.frequency = 2000.0  // Reduce noise in 2kHz range
    midNoiseReduction.bandwidth = 1.0
    midNoiseReduction.gain = -10.0  // Reduce noise intensity
    midNoiseReduction.bypass = false
  }

  public func purgeAudioSamples(keepingLast keep: Int) {
    audioQueue.sync(flags: .barrier) {
      if _audioSamples.count > keep {
        _audioSamples = ContiguousArray(_audioSamples.suffix(keep))
      }
    }
  }

  public func purgeSilnceFromSamples(silenceLength: Int) {
    audioQueue.sync(flags: .barrier) {
      let totalSamples = _audioSamples.count
      let middleSecond = WhisperKit.sampleRate * 10  // 1 second of samples
      let endSeconds = WhisperKit.sampleRate * 5

      // If we don't have enough samples for meaningful processing, return early
      if totalSamples <= (middleSecond + silenceLength) {
        return
      }

      // Calculate the range to keep
      let endIndex = totalSamples - silenceLength
      let startIndex = max(0, endIndex - middleSecond)

      // Use removeSubrange for efficient removal without creating new array
      if startIndex > 0 {
        _audioSamples.removeSubrange(0..<startIndex)
      }
      if _audioSamples.count > middleSecond {
        _audioSamples.removeSubrange(middleSecond...(_audioSamples.count - 1 - endSeconds))
      }
    }
  }

  public func startRecordingLive(
    inputDeviceID: DeviceID? = nil, callback: (([Float]) -> Void)? = nil
  ) throws {
    audioQueue.sync {
      _audioSamples = []
      _audioSamples.reserveCapacity(30 * WhisperKit.sampleRate)
    }
    audioEnergy = []

    try? setupAudioSessionForDevice()

    audioEngine = try setupEngine(inputDeviceID: inputDeviceID)

    // Set the callback
    audioBufferCallback = callback

    lastInputDevice = inputDeviceID
  }

  public func resumeRecordingLive(
    inputDeviceID: DeviceID? = nil, callback: (([Float]) -> Void)? = nil
  ) throws {
    try? setupAudioSessionForDevice()

    if inputDeviceID == lastInputDevice {
      try audioEngine?.start()
    } else {
      audioEngine = try setupEngine(inputDeviceID: inputDeviceID)
    }

    // Set the callback only if the provided callback is not nil
    if let callback = callback {
      audioBufferCallback = callback
    }
  }

  public func pauseRecording() {
    audioEngine?.pause()
  }

  public func stopRecording() {
    // Remove the tap on any attached node
    audioEngine?.attachedNodes.forEach { node in
      node.removeTap(onBus: 0)
    }

    // Stop the audio engine
    audioEngine?.stop()
    audioEngine = nil
  }
}

public protocol ProcessTapProtocol {
  func activate()
  func invalidate(reason: ProcessTapInvalidationReason)

  var activated: Bool { get }

  var tapStreamDescription: AudioStreamBasicDescription? { get }

  func run(
    on queue: DispatchQueue, ioBlock: @escaping AudioDeviceIOBlock,
    invalidationHandler: ((ProcessTapProtocol, ProcessTapInvalidationReason) -> Void)?
  ) throws
}

public enum ProcessTapInvalidationReason {
  case processTerminated
  case audioDeviceChaned
}

extension AudioProcessor {
  @MainActor
  public func startRecordingLiveOutput(
    tap: ProcessTapProtocol,
    callback: (([Float]) -> Void)? = nil,
    invalidationHandler: ((ProcessTapProtocol, ProcessTapInvalidationReason) -> Void)? =
      nil
  ) throws {
    audioSamples = []
    audioEnergy = []

    try? setupAudioSessionForDevice()

    try setupEngineForOutput(
      tap: tap,
      invalidationHandler: invalidationHandler
    )

    // Set the callback
    audioBufferCallback = callback
  }

  @MainActor
  func setupEngineForOutput(
    tap: ProcessTapProtocol,
    invalidationHandler: ((ProcessTapProtocol, ProcessTapInvalidationReason) -> Void)?
  ) throws {

    // Clean up existing tap if any
    currentTap?.invalidate(reason: .processTerminated)

    self.currentTap = tap

    if currentTap?.activated == false { currentTap?.activate() }

    guard var streamDescription = currentTap?.tapStreamDescription else {
      throw WhisperError.audioProcessingFailed("Tap stream description not available.")
    }

    guard let nodeFormat = AVAudioFormat(streamDescription: &streamDescription) else {
      throw WhisperError.audioProcessingFailed("Failed to create AVAudioFormat.")
    }

    // Desired format (16,000 Hz, 1 channel)
    guard
      let desiredFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: Double(WhisperKit.sampleRate),
        channels: AVAudioChannelCount(1), interleaved: false)
    else {
      throw WhisperError.audioProcessingFailed("Failed to create desired format")
    }

    guard let converter = AVAudioConverter(from: nodeFormat, to: desiredFormat) else {
      throw WhisperError.audioProcessingFailed("Failed to create audio converter")
    }

    self.accumulationBuffer =
      AVAudioPCMBuffer(pcmFormat: nodeFormat, frameCapacity: AVAudioFrameCount(minBufferLength))!
    self.accumulationBuffer?.frameLength = 0

    try tap.run(
      on: processingQueue,
      ioBlock: { [weak self] _, inInputData, _, _, _ in
        guard let self = self else { return }
        self.processInputData(inInputData, nodeFormat: nodeFormat, converter: converter)
      }, invalidationHandler: invalidationHandler
    )
  }

  private func processInputData(
    _ inInputData: UnsafePointer<AudioBufferList>,
    nodeFormat: AVAudioFormat,
    converter: AVAudioConverter
  ) {
    // Convert input data to AVAudioPCMBuffer
    let audioBuffer = inInputData.pointee.mBuffers
    guard let dataPointer = audioBuffer.mData else {
      Logging.error("Invalid audio buffer data")
      return
    }

    // Determine frame count correctly
    let incomingFrameCount = audioBuffer.mDataByteSize / UInt32(MemoryLayout<Float>.size)

    // Create AVAudioPCMBuffer
    guard
      var pcmBuffer = AVAudioPCMBuffer(
        pcmFormat: nodeFormat,
        frameCapacity: AVAudioFrameCount(incomingFrameCount)
      )
    else {
      Logging.error("Failed to create PCM buffer")
      return
    }

    pcmBuffer.frameLength = AVAudioFrameCount(incomingFrameCount)

    guard let accBuffer = accumulationBuffer else {
      Logging.error("Accumulation buffer is nil")
      return
    }

    let accFramesUsed = accBuffer.frameLength
    let accFramesAvailable = accBuffer.frameCapacity - accFramesUsed
    let framesToCopy = min(accFramesAvailable, incomingFrameCount)

    // Copy data safely
    if let dstPointer = accBuffer.floatChannelData?[0] {
      let sourcePointer = dataPointer.assumingMemoryBound(to: Float.self)
      // Use memcpy for safe memory copying
      memcpy(
        dstPointer.advanced(by: Int(accFramesUsed)),
        sourcePointer,
        Int(framesToCopy) * MemoryLayout<Float>.size
      )
    }

    accBuffer.frameLength += framesToCopy

    if accBuffer.frameLength == AVAudioFrameCount(minBufferLength) {
      if !accBuffer.format.sampleRate.isEqual(to: Double(WhisperKit.sampleRate)) {
        do {
          pcmBuffer = try Self.resampleBuffer(accBuffer, with: converter)
        } catch {
          Logging.error("Failed to resample buffer: \(error)")
          return
        }
      } else {
        pcmBuffer = accBuffer
      }

      let newBufferArray = Self.convertBufferToArray(buffer: pcmBuffer)
      self.processBuffer(newBufferArray)

      self.accumulationBuffer = AVAudioPCMBuffer(
        pcmFormat: nodeFormat,
        frameCapacity: AVAudioFrameCount(minBufferLength)
      )!
      self.accumulationBuffer?.frameLength = 0
    }
  }
}
