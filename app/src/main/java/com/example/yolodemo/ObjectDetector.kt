package com.example.yolodemo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ObjectDetector(private val context: Context) {
    private lateinit var interpreter: Interpreter
    private val inputSize = 640
    private val labels = mutableListOf<String>()
    private val confThreshold = 0.45f
    private val iouThreshold = 0.5f

    data class Detection(val label: String, val confidence: Float, val boundingBox: RectF)

    fun initialize() {
        try {
            context.assets.open("labels.txt").bufferedReader().forEachLine {
                labels.add(it.trim())
            }
            Log.d("YOLO", "Labels loaded: ${labels.size} -> $labels")

            val modelBuffer = loadModelFile("best_integer_quant.tflite")
            val options = Interpreter.Options().apply {
                numThreads = 4
            }
            interpreter = Interpreter(modelBuffer, options)

            val inputShape = interpreter.getInputTensor(0).shape()
            val inputType = interpreter.getInputTensor(0).dataType()
            val outputShape = interpreter.getOutputTensor(0).shape()
            val outputType = interpreter.getOutputTensor(0).dataType()
            val scale = interpreter.getOutputTensor(0).quantizationParams().scale
            val zeroPoint = interpreter.getOutputTensor(0).quantizationParams().zeroPoint

            Log.d("YOLO", "Input shape: ${inputShape.toList()}, type: $inputType")
            Log.d("YOLO", "Output shape: ${outputShape.toList()}, type: $outputType")
            Log.d("YOLO", "Output quantization: scale=$scale, zeroPoint=$zeroPoint")
            Log.d("YOLO", "Model loaded successfully")

        } catch (e: Exception) {
            Log.e("YOLO", "Error initializing model: ${e.message}", e)
        }
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        return try {
            val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            val inputType = interpreter.getInputTensor(0).dataType()

            if (inputType == DataType.FLOAT32) {
                // Float32 model: normalize to [0, 1]
                val inputArray = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(3) } } }
                for (y in 0 until inputSize) {
                    for (x in 0 until inputSize) {
                        val pixel = resized.getPixel(x, y)
                        inputArray[0][y][x][0] = ((pixel shr 16) and 0xFF) / 255.0f
                        inputArray[0][y][x][1] = ((pixel shr 8) and 0xFF) / 255.0f
                        inputArray[0][y][x][2] = (pixel and 0xFF) / 255.0f
                    }
                }
                val outputShape = interpreter.getOutputTensor(0).shape()
                val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
                interpreter.run(inputArray, outputBuffer.buffer.rewind())
                parseOutput(outputBuffer.floatArray, bitmap.width, bitmap.height)

            } else {
                // INT8/UINT8 model: send raw bytes [0-255]
                val inputArray = Array(1) { Array(inputSize) { Array(inputSize) { ByteArray(3) } } }
                for (y in 0 until inputSize) {
                    for (x in 0 until inputSize) {
                        val pixel = resized.getPixel(x, y)
                        inputArray[0][y][x][0] = ((pixel shr 16) and 0xFF).toByte()
                        inputArray[0][y][x][1] = ((pixel shr 8) and 0xFF).toByte()
                        inputArray[0][y][x][2] = (pixel and 0xFF).toByte()
                    }
                }
                val outputShape = interpreter.getOutputTensor(0).shape()
                val outputType = interpreter.getOutputTensor(0).dataType()
                val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType)
                interpreter.run(inputArray, outputBuffer.buffer.rewind())

                // Dequantize output
                val scale = interpreter.getOutputTensor(0).quantizationParams().scale
                val zeroPoint = interpreter.getOutputTensor(0).quantizationParams().zeroPoint
                val floatOutput = if (outputType == DataType.UINT8 || outputType == DataType.INT8) {
                    outputBuffer.buffer.rewind()
                    FloatArray(outputBuffer.flatSize) { _ ->
                        val raw = outputBuffer.buffer.get().toInt() and 0xFF
                        (raw - zeroPoint) * scale
                    }
                } else {
                    outputBuffer.floatArray
                }

                parseOutput(floatOutput, bitmap.width, bitmap.height)
            }

        } catch (e: Exception) {
            Log.e("YOLO", "Detection error: ${e.message}", e)
            emptyList()
        }
    }

    private fun parseOutput(output: FloatArray, imgW: Int, imgH: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numClasses = labels.size
        val numBoxes = 8400

        Log.d("YOLO", "Output array size: ${output.size}")
        Log.d("YOLO", "Max value in output: ${output.max()}")
        Log.d("YOLO", "imgW: $imgW, imgH: $imgH")

        for (i in 0 until numBoxes) {
            var maxConf = 0f
            var classIdx = -1

            for (c in 0 until numClasses) {
                val conf = output[(4 + c) * numBoxes + i]
                if (conf > maxConf) {
                    maxConf = conf
                    classIdx = c
                }
            }

            if (maxConf >= confThreshold && classIdx >= 0) {
                // Coordinates are already normalized (0-1), just scale to image size
                val cx = output[0 * numBoxes + i] * imgW
                val cy = output[1 * numBoxes + i] * imgH
                val w  = output[2 * numBoxes + i] * imgW
                val h  = output[3 * numBoxes + i] * imgH

                Log.d("YOLO", "Detection: label=${labels[classIdx]}, conf=$maxConf, cx=$cx, cy=$cy, w=$w, h=$h")

                detections.add(
                    Detection(
                        label = labels[classIdx],
                        confidence = maxConf,
                        boundingBox = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                    )
                )
            }
        }

        Log.d("YOLO", "Raw detections before NMS: ${detections.size}")
        return applyNMS(detections)
    }

    private fun applyNMS(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val result = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(it.boundingBox, best.boundingBox) > iouThreshold }
        }
        Log.d("YOLO", "Detections after NMS: ${result.size}")
        return result
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)
        val interArea = maxOf(0f, interRight - interLeft) * maxOf(0f, interBottom - interTop)
        val unionArea = (a.width() * a.height()) + (b.width() * b.height()) - interArea
        return if (unionArea == 0f) 0f else interArea / unionArea
    }

    fun close() = interpreter.close()
}