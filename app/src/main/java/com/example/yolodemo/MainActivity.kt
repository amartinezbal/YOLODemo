package com.example.yolodemo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.camera.view.TransformExperimental
import androidx.camera.view.transform.CoordinateTransform
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.doOnLayout
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@androidx.annotation.OptIn(TransformExperimental::class)
class MainActivity : AppCompatActivity() {
    private lateinit var detector: ObjectDetector
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        findViewById<PreviewView>(R.id.viewFinder).implementationMode =
            PreviewView.ImplementationMode.COMPATIBLE

        try {
            detector = ObjectDetector(this).apply { initialize() }
        } catch (e: Exception) {
            Log.e("YOLO", "Failed to initialize detector: ${e.message}", e)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            findViewById<PreviewView>(R.id.viewFinder).doOnLayout { startCamera() }
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                10
            )
        }
    }

    private fun startCamera() {
        Log.d("YOLO", "startCamera() called")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val previewView = findViewById<PreviewView>(R.id.viewFinder)
            val viewPort = previewView.viewPort ?: return@addListener
            val transformFactory = ImageProxyTransformFactory().apply {
                isUsingCropRect = true
                isUsingRotationDegrees = true
            }

            val preview = Preview.Builder().build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val analyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        try {
                            val bitmap = imageProxy.toBitmap()
                            val crop = imageProxy.cropRect
                            val croppedBitmap = Bitmap.createBitmap(
                                bitmap, crop.left, crop.top, crop.width(), crop.height()
                            )
                            val sourceTransform = transformFactory.getOutputTransform(imageProxy)

                            // Fix rotation
                            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                            val matrix = android.graphics.Matrix()
                            matrix.postRotate(rotationDegrees.toFloat())
                            val rotatedBitmap = Bitmap.createBitmap(
                                croppedBitmap, 0, 0, croppedBitmap.width, croppedBitmap.height, matrix, true
                            )

                            val results = detector.detect(rotatedBitmap)
                            Log.d("YOLO", "Rotation: $rotationDegrees | Detections: ${results.size} | Size: ${rotatedBitmap.width}x${rotatedBitmap.height}")

                            runOnUiThread {
                                val targetTransform = previewView.outputTransform
                                val mappedResults = if (targetTransform == null) {
                                    emptyList()
                                } else {
                                    val transform = CoordinateTransform(sourceTransform, targetTransform)
                                    results.map { detection ->
                                        val box = RectF(detection.boundingBox)
                                        transform.mapRect(box)
                                        detection.copy(boundingBox = box)
                                    }
                                }
                                findViewById<OverlayView>(R.id.overlay)
                                    .setDetections(mappedResults)
                            }

                            rotatedBitmap.recycle()
                            if (croppedBitmap !== rotatedBitmap) croppedBitmap.recycle()
                            if (bitmap !== croppedBitmap && bitmap !== rotatedBitmap) bitmap.recycle()

                        } catch (e: Exception) {
                            Log.e("YOLO", "Detection error: ${e.message}", e)
                        } finally {
                            imageProxy.close()
                        }
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    UseCaseGroup.Builder()
                        .setViewPort(viewPort)
                        .addUseCase(preview)
                        .addUseCase(analyzer)
                        .build()
                )
                Log.d("YOLO", "Camera bound successfully")
            } catch (e: Exception) {
                Log.e("YOLO", "Camera binding failed: ${e.message}", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 10 && grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            findViewById<PreviewView>(R.id.viewFinder).doOnLayout { startCamera() }
        } else {
            Log.e("YOLO", "Camera permission denied")
            finish()
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
        cameraExecutor.shutdown()
    }
}