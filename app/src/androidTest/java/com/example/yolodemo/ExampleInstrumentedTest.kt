package com.example.yolodemo

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.util.Size
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.camera.view.TransformExperimental
import androidx.camera.view.transform.CoordinateTransform
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.camera.view.transform.OutputTransform
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*
import java.lang.reflect.Proxy
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
@androidx.annotation.OptIn(TransformExperimental::class)
class ExampleInstrumentedTest {
    @Test
    fun letterboxPreservesAspectRatioAndRestoresBoxes() {
        for (size in listOf(Size(486, 1080), Size(1080, 486), Size(640, 640), Size(333, 777))) {
            val transform = ObjectDetector.LetterboxTransform(size.width, size.height, 640)
            val source = Bitmap.createBitmap(size.width, size.height, Bitmap.Config.ARGB_8888)
            source.eraseColor(Color.RED)
            val padded = transform.prepare(source)
            try {
                assertEquals(640, padded.width)
                assertEquals(640, padded.height)
                assertFalse(source.isRecycled)
                assertTrue(kotlin.math.abs(
                    transform.resizedWidth - size.width * minOf(640f / size.width, 640f / size.height)
                ) <= 0.5f)
                assertTrue(kotlin.math.abs(
                    transform.resizedHeight - size.height * minOf(640f / size.width, 640f / size.height)
                ) <= 0.5f)
                assertEquals(Color.RED, padded.getPixel(320, 320))
                if (transform.paddingLeft > 0 || transform.paddingTop > 0) {
                    assertEquals(Color.rgb(114, 114, 114), padded.getPixel(0, 0))
                }
                for (fraction in listOf(RectF(0f, 0f, 1f, 1f), RectF(0.1f, 0.2f, 0.6f, 0.8f))) {
                    val restored = transform.mapBox(
                        (transform.paddingLeft + fraction.centerX() * transform.resizedWidth) / 640f,
                        (transform.paddingTop + fraction.centerY() * transform.resizedHeight) / 640f,
                        fraction.width() * transform.resizedWidth / 640f,
                        fraction.height() * transform.resizedHeight / 640f
                    )
                    assertNotNull(restored)
                    assertRectEquals(RectF(
                        fraction.left * size.width, fraction.top * size.height,
                        fraction.right * size.width, fraction.bottom * size.height
                    ), restored!!)
                }
            } finally {
                padded.recycle()
                source.recycle()
            }
        }
    }

    @Test
    fun letterboxRejectsPaddingAndClipsPartialBoxes() {
        val portrait = ObjectDetector.LetterboxTransform(320, 640, 640)
        assertNull(portrait.mapBox(0.1f, 0.5f, 0.1f, 0.2f))
        assertNull(portrait.mapBox(Float.NaN, 0.5f, 0.2f, 0.2f))
        assertNull(portrait.mapBox(0.5f, 0.5f, -0.2f, 0.2f))
        assertRectEquals(RectF(0f, 160f, 160f, 480f),
            portrait.mapBox(0.25f, 0.5f, 0.5f, 0.5f)!!)
        val landscape = ObjectDetector.LetterboxTransform(640, 320, 640)
        assertNull(landscape.mapBox(0.5f, 0.1f, 0.2f, 0.1f))
        assertRectEquals(RectF(160f, 0f, 480f, 160f),
            landscape.mapBox(0.5f, 0.25f, 0.5f, 0.5f)!!)
    }

    @Test
    fun pretrainedModelRunsWithExpectedTensors() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val labels = context.assets.open("labels.txt").bufferedReader().use { it.readLines() }
        assertEquals(80, labels.size)
        Interpreter(FileUtil.loadMappedFile(context, "yolov8n_float32.tflite")).use { model ->
            assertArrayEquals(intArrayOf(1, 640, 640, 3), model.getInputTensor(0).shape())
            assertArrayEquals(intArrayOf(1, 84, 8400), model.getOutputTensor(0).shape())
            assertEquals(DataType.FLOAT32, model.getInputTensor(0).dataType())
            assertEquals(DataType.FLOAT32, model.getOutputTensor(0).dataType())
            val input = ByteBuffer.allocateDirect(model.getInputTensor(0).numBytes())
                .order(ByteOrder.nativeOrder())
            val output = ByteBuffer.allocateDirect(model.getOutputTensor(0).numBytes())
                .order(ByteOrder.nativeOrder())
            model.run(input, output)
            val values = output.apply { rewind() }.asFloatBuffer()
            for (index in 0 until values.limit()) {
                val value = values.get(index)
                assertTrue("Nonfinite output at $index", value.isFinite())
                if (index >= 4 * 8400) {
                    assertTrue("Invalid class score at $index: $value", value in 0f..1f)
                }
            }
        }
    }

    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.example.yolodemo", appContext.packageName)
    }

    @Test
    fun croppedAndRotatedDetectionsMapToPreview() {
        val crop = Rect(80, 60, 560, 420)
        val factory = ImageProxyTransformFactory().apply {
            isUsingCropRect = true
            isUsingRotationDegrees = true
        }

        for (rotation in listOf(0, 90, 180, 270)) {
            val imageInfo = Proxy.newProxyInstance(
                ImageInfo::class.java.classLoader, arrayOf(ImageInfo::class.java)
            ) { _, method, _ ->
                when (method.name) {
                    "getRotationDegrees" -> rotation
                    else -> error("Unexpected ImageInfo method: ${method.name}")
                }
            } as ImageInfo
            val image = Proxy.newProxyInstance(
                ImageProxy::class.java.classLoader, arrayOf(ImageProxy::class.java)
            ) { _, method, _ ->
                when (method.name) {
                    "getWidth" -> 640
                    "getHeight" -> 480
                    "getCropRect" -> crop
                    "getImageInfo" -> imageInfo
                    else -> error("Unexpected ImageProxy method: ${method.name}")
                }
            } as ImageProxy

            val targetMatrix = Matrix().apply {
                setRotate(rotation.toFloat())
                postScale(540f, 960f)
                postTranslate(540f, 960f)
            }
            val target = OutputTransform(targetMatrix, Size(crop.width(), crop.height()))
            val transform = CoordinateTransform(factory.getOutputTransform(image), target)
            val rotatedWidth = if (rotation % 180 == 0) crop.width() else crop.height()
            val rotatedHeight = if (rotation % 180 == 0) crop.height() else crop.width()
            val bounds = RectF(0f, 0f, rotatedWidth.toFloat(), rotatedHeight.toFloat())
            transform.mapRect(bounds)
            assertRectEquals(RectF(0f, 0f, 1080f, 1920f), bounds)

            val centerBox = RectF(
                rotatedWidth * 0.25f, rotatedHeight * 0.25f,
                rotatedWidth * 0.75f, rotatedHeight * 0.75f
            )
            transform.mapRect(centerBox)
            assertRectEquals(RectF(270f, 480f, 810f, 1440f), centerBox)

            val cornerBox = RectF(0f, 0f, rotatedWidth * 0.25f, rotatedHeight * 0.25f)
            transform.mapRect(cornerBox)
            assertRectEquals(RectF(0f, 0f, 270f, 480f), cornerBox)
        }
    }

    private fun assertRectEquals(expected: RectF, actual: RectF) {
        assertEquals(expected.left, actual.left, 0.01f)
        assertEquals(expected.top, actual.top, 0.01f)
        assertEquals(expected.right, actual.right, 0.01f)
        assertEquals(expected.bottom, actual.bottom, 0.01f)
    }
}