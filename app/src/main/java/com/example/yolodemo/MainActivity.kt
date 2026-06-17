package com.example.yolodemo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var detector: ObjectDetector
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            detector = ObjectDetector(this).apply { initialize() }
        } catch (e: Exception) {
            Log.e("YOLO", "Failed to initialize detector: ${e.message}", e)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCamera()
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

            val preview = Preview.Builder().build()
                .also {
                    it.setSurfaceProvider(
                        findViewById<PreviewView>(R.id.viewFinder).surfaceProvider
                    )
                }

            val analyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        try {
                            val bitmap = imageProxy.toBitmap()

                            // Fix rotation
                            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                            val matrix = android.graphics.Matrix()
                            matrix.postRotate(rotationDegrees.toFloat())
                            val rotatedBitmap = Bitmap.createBitmap(
                                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                            )

                            val results = detector.detect(rotatedBitmap)
                            Log.d("YOLO", "Rotation: $rotationDegrees | Detections: ${results.size} | Size: ${rotatedBitmap.width}x${rotatedBitmap.height}")

                            runOnUiThread {
                                findViewById<OverlayView>(R.id.overlay)
                                    .setDetections(results, rotatedBitmap.width, rotatedBitmap.height)
                            }

                            bitmap.recycle()
                            rotatedBitmap.recycle()

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
                    preview,
                    analyzer
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
            startCamera()
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