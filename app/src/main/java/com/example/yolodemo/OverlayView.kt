package com.example.yolodemo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var detections: List<ObjectDetector.Detection> = emptyList()

    // Corner length as fraction of box size
    private val cornerLengthRatio = 0.2f
    private val cornerStrokeWidth = 8f
    private val boxStrokeWidth = 3f

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = boxStrokeWidth
        isAntiAlias = true
    }

    private val cornerPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = cornerStrokeWidth
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 42f
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        isAntiAlias = true
    }

    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // Color palette for different classes
    private val colorPalette = listOf(
        Color.parseColor("#FF3B30"), // red
        Color.parseColor("#007AFF"), // blue
        Color.parseColor("#34C759"), // green
        Color.parseColor("#FF9500"), // orange
        Color.parseColor("#AF52DE"), // purple
        Color.parseColor("#00C7BE"), // teal
        Color.parseColor("#FFD60A"), // yellow
        Color.parseColor("#FF2D55")  // pink
    )

    fun setDetections(results: List<ObjectDetector.Detection>) {
        detections = results
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for ((index, det) in detections.withIndex()) {
            val color = colorPalette[index % colorPalette.size]

            val scaledBox = RectF(det.boundingBox)

            // Clamp box to screen bounds
            scaledBox.left   = scaledBox.left.coerceAtLeast(0f)
            scaledBox.top    = scaledBox.top.coerceAtLeast(0f)
            scaledBox.right  = scaledBox.right.coerceAtMost(width.toFloat())
            scaledBox.bottom = scaledBox.bottom.coerceAtMost(height.toFloat())

            // Draw semi-transparent box fill
            bgPaint.color = color
            bgPaint.alpha = 30
            canvas.drawRoundRect(scaledBox, 12f, 12f, bgPaint)

            // Draw box border
            boxPaint.color = color
            boxPaint.alpha = 180
            canvas.drawRoundRect(scaledBox, 12f, 12f, boxPaint)

            // Draw corner accents
            cornerPaint.color = color
            cornerPaint.alpha = 255
            drawCorners(canvas, scaledBox)

            // Draw label
            val label = "${det.label} ${"%.0f".format(det.confidence * 100)}%"
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize
            val padding = 10f

            // Label background
            val labelBox = RectF(
                scaledBox.left,
                scaledBox.top - textHeight - padding * 2,
                scaledBox.left + textWidth + padding * 2,
                scaledBox.top
            )

            // Keep label inside screen
            if (labelBox.top < 0) {
                labelBox.offset(0f, -labelBox.top)
            }

            bgPaint.color = color
            bgPaint.alpha = 220
            canvas.drawRoundRect(labelBox, 8f, 8f, bgPaint)

            // Label text
            canvas.drawText(
                label,
                labelBox.left + padding,
                labelBox.bottom - padding,
                textPaint
            )
        }
    }

    private fun drawCorners(canvas: Canvas, box: RectF) {
        val cornerLength = minOf(box.width(), box.height()) * cornerLengthRatio

        // Top-left
        canvas.drawLine(box.left, box.top, box.left + cornerLength, box.top, cornerPaint)
        canvas.drawLine(box.left, box.top, box.left, box.top + cornerLength, cornerPaint)

        // Top-right
        canvas.drawLine(box.right, box.top, box.right - cornerLength, box.top, cornerPaint)
        canvas.drawLine(box.right, box.top, box.right, box.top + cornerLength, cornerPaint)

        // Bottom-left
        canvas.drawLine(box.left, box.bottom, box.left + cornerLength, box.bottom, cornerPaint)
        canvas.drawLine(box.left, box.bottom, box.left, box.bottom - cornerLength, cornerPaint)

        // Bottom-right
        canvas.drawLine(box.right, box.bottom, box.right - cornerLength, box.bottom, cornerPaint)
        canvas.drawLine(box.right, box.bottom, box.right, box.bottom - cornerLength, cornerPaint)
    }
}