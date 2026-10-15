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
 * Draws a plain viewfinder, plus the detected page outline, over a FIT_CENTER preview.
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
    strokeJoin = Paint.Join.ROUND
    strokeCap = Paint.Cap.SQUARE
  }
  private val bracket = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    style = Paint.Style.STROKE
    strokeCap = Paint.Cap.SQUARE
    color = Color.argb(180, 244, 241, 234)
  }

  fun show(quad: List<Point>?, frameWidth: Int, frameHeight: Int, steady: Boolean) {
    this.quad = quad
    frameW = frameWidth.coerceAtLeast(1)
    frameH = frameHeight.coerceAtLeast(1)
    this.steady = steady
    invalidate()
  }

  override fun onDraw(canvas: Canvas) {
    val density = resources.displayMetrics.density
    bracket.strokeWidth = 2f * density
    drawBrackets(canvas)
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
    paint.strokeWidth = 3f * density
    paint.color = if (steady) Color.rgb(244, 180, 60) else Color.rgb(32, 196, 160)
    canvas.drawPath(path, paint)
  }

  /** Four corner ticks so the preview reads as a scanner, even before a page is found. */
  private fun drawBrackets(canvas: Canvas) {
    val inset = min(width, height) * 0.08f
    val arm = min(width, height) * 0.07f
    val right = width - inset
    val bottom = height - inset
    canvas.drawLine(inset, inset, inset + arm, inset, bracket)
    canvas.drawLine(inset, inset, inset, inset + arm, bracket)
    canvas.drawLine(right, inset, right - arm, inset, bracket)
    canvas.drawLine(right, inset, right, inset + arm, bracket)
    canvas.drawLine(inset, bottom, inset + arm, bottom, bracket)
    canvas.drawLine(inset, bottom, inset, bottom - arm, bracket)
    canvas.drawLine(right, bottom, right - arm, bottom, bracket)
    canvas.drawLine(right, bottom, right, bottom - arm, bracket)
  }
}
