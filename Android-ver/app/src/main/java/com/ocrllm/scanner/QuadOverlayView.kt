package com.ocrllm.scanner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import org.opencv.core.Point
import kotlin.math.min

/**
 * Draws the detected page outline over a camera preview shown with FIT_CENTER.
 * Corners are in analysis-frame pixels (frame already rotated upright).
 */
class QuadOverlayView @JvmOverloads constructor(
  context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {
  private var quad: List<Point>? = null
  private var frameW = 1
  private var frameH = 1
  private var steady = false
  private val path = Path()
  private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.STROKE
    strokeWidth = 4f * resources.displayMetrics.density
    strokeJoin = Paint.Join.ROUND
  }

  fun show(quad: List<Point>?, frameWidth: Int, frameHeight: Int, steady: Boolean) {
    this.quad = quad
    frameW = frameWidth
    frameH = frameHeight
    this.steady = steady
    invalidate()
  }

  override fun onDraw(canvas: Canvas) {
    val q = quad ?: return
    val scale = min(width.toFloat() / frameW, height.toFloat() / frameH)
    val dx = (width - frameW * scale) / 2
    val dy = (height - frameH * scale) / 2
    path.reset()
    q.forEachIndexed { i, p ->
      val x = dx + p.x.toFloat() * scale
      val y = dy + p.y.toFloat() * scale
      if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
    }
    path.close()
    // Green while tracking, amber once the page holds still (auto-capture is about to fire).
    paint.color = if (steady) Color.rgb(255, 200, 0) else Color.rgb(0, 200, 0)
    canvas.drawPath(path, paint)
  }
}
