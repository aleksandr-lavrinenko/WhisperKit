//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Accelerate
import AVFoundation
import CoreGraphics
import CoreML
import Foundation

public protocol FeatureExtractorOutputType {}
extension MLMultiArray: FeatureExtractorOutputType {}

public protocol FeatureExtracting {
    associatedtype OutputType: FeatureExtractorOutputType

    var melCount: Int? { get }
    var windowSamples: Int? { get }
    func logMelSpectrogram(fromAudio inputAudio: MLMultiArray) async throws -> OutputType?
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
open class FeatureExtractor: FeatureExtracting, WhisperMLModel {
    public var model: MLModel?

    public init() {}

    public var melCount: Int? {
        guard let inputDescription = model?.modelDescription.outputDescriptionsByName["melspectrogram_features"] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[1]
    }

    public var windowSamples: Int? {
        guard let inputDescription = model?.modelDescription.inputDescriptionsByName["audio"] else { return nil }
        guard inputDescription.type == .multiArray else { return nil }
        guard let shapeConstraint = inputDescription.multiArrayConstraint else { return nil }
        let shape = shapeConstraint.shape.map { $0.intValue }
        return shape[0]  // The audio input is a 1D array
    }

    public func logMelSpectrogram(fromAudio inputAudio: MLMultiArray) async throws -> MLMultiArray? {
        guard let model else {
            throw WhisperError.modelsUnavailable()
        }
        try Task.checkCancellation()
        
        if checkForNaNOrInf(array: inputAudio) {
            print("🚨 Warning: NaN or Inf found in inputAudio!")
        }
        
        clipSmallValues(array: inputAudio)
        
        if checkForNaNOrInf(array: inputAudio) {
            print("🚨 Warning: NaN or Inf found in inputAudio!")
        }
        
        print("Input Audio Shape: \(inputAudio.shape)") // Expecting [480000]
        print("Input Audio Count: \(inputAudio.count)") // Should match 480000
        print("Input Audio Data Type: \(inputAudio.dataType)") // Should be .float32

        let interval = Logging.beginSignpost("ExtractAudioFeatures", signposter: Logging.FeatureExtractor.signposter)
        defer { Logging.endSignpost("ExtractAudioFeatures", interval: interval, signposter: Logging.FeatureExtractor.signposter) }

        let options = MLPredictionOptions()
        options.usesCPUOnly = true
        let modelInputs = MelSpectrogramInput(audio: inputAudio)
        let outputFeatures = try await model.asyncPrediction(from: modelInputs, options: options)
        let output = MelSpectrogramOutput(features: outputFeatures)
        return output.melspectrogramFeatures
    }

    
    func checkForNaNOrInf(array: MLMultiArray) -> Bool {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<array.count {
            if ptr[i].isNaN || ptr[i].isInfinite {
                print("Found NaN or Inf at index \(i): \(ptr[i])")
                return true
            }
        }
        return false
    }
    
    func clipSmallValues(array: MLMultiArray, minValue: Float = 2e-4) {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<array.count {
            if ptr[i] < minValue || abs(ptr[i]) == 0.0 {
                ptr[i] = minValue
            }
        }
    }
}
