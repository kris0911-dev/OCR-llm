package com.ocrllm.scanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.graphics.Path
import android.net.Uri
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** Document scanning helpers (port of Desktop-ver/ocr_app/scanner.py). */
object Scanner {
  /**
   * Corners of the largest quadrilateral covering at least [minAreaRatio] of the frame,
   * in frame pixels, or null.
   */
  fun findDocument(frameBgr: Mat, minAreaRatio: Double = 0.2): List<Point>? {
    val h = frameBgr.rows()
    val w = frameBgr.cols()
    var scale = 500.0 / max(h, w)
    val small = if (scale < 1) Mat().also { Imgproc.resize(frameBgr, it, Size(), scale, scale) } else frameBgr
    scale = min(scale, 1.0)

    val gray = Mat()
    Imgproc.cvtColor(small, gray, Imgproc.COLOR_BGR2GRAY)
    Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
    val edges = Mat()
    Imgproc.Canny(gray, edges, 50.0, 150.0)
    val kernel = Mat.ones(3, 3, org.opencv.core.CvType.CV_8U)
    Imgproc.dilate(edges, edges, kernel)

    val contours = ArrayList<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
    val minArea = minAreaRatio * small.rows() * small.cols()
    var result: List<Point>? = null
    for (cnt in contours.sortedByDescending { Imgproc.contourArea(it) }.take(5)) {
      if (Imgproc.contourArea(cnt) < minArea) break
      val curve = MatOfPoint2f(*cnt.toArray())
      val approx = MatOfPoint2f()
      Imgproc.approxPolyDP(curve, approx, 0.02 * Imgproc.arcLength(curve, true), true)
      val corners = approx.toArray()
      val convex = corners.size == 4 && MatOfPoint(*corners).let { m ->
        Imgproc.isContourConvex(m).also { m.release() }
      }
      curve.release()
      approx.release()
      if (convex) {
        result = corners.map { Point(it.x / scale, it.y / scale) }
        break
      }
    }
    contours.forEach { it.release() }
    listOf(gray, edges, kernel, hierarchy).forEach { it.release() }
    if (small !== frameBgr) small.release()
    return result
  }

  /** Order as top-left, top-right, bottom-right, bottom-left. */
  fun orderCorners(pts: List<Point>): List<Point> {
    val sum = pts.map { it.x + it.y }
    val diff = pts.map { it.y - it.x }
    return listOf(
      pts[sum.indexOf(sum.min())], pts[diff.indexOf(diff.min())],
      pts[sum.indexOf(sum.max())], pts[diff.indexOf(diff.max())],
    )
  }

  /** Perspective-correct the page delimited by [corners]. */
  fun warpDocument(frameBgr: Mat, corners: List<Point>): Mat {
    val (tl, tr, br, bl) = orderCorners(corners)
    val width = max(dist(br, bl), dist(tr, tl)).toInt()
    val height = max(dist(tr, br), dist(tl, bl)).toInt()
    val src = MatOfPoint2f(tl, tr, br, bl)
    val dst = MatOfPoint2f(
      Point(0.0, 0.0), Point(width - 1.0, 0.0),
      Point(width - 1.0, height - 1.0), Point(0.0, height - 1.0),
    )
    val matrix = Imgproc.getPerspectiveTransform(src, dst)
    val out = Mat()
    Imgproc.warpPerspective(frameBgr, out, matrix, Size(width.toDouble(), height.toDouble()))
    listOf(src, dst, matrix).forEach { it.release() }
    return out
  }

  private fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)
}

/** Bitmap / Mat conversion and drawing helpers. */
object Images {
  /** Largest image side accepted (PaddleX max_side_limit). */
  const val MAX_SIDE = 4000

  fun bitmapToBgr(bitmap: Bitmap): Mat {
    val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)
    val rgba = Mat()
    Utils.bitmapToMat(argb, rgba)
    val bgr = Mat()
    Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
    rgba.release()
    return bgr
  }

  fun bgrToBitmap(bgr: Mat): Bitmap {
    val rgba = Mat()
    Imgproc.cvtColor(bgr, rgba, Imgproc.COLOR_BGR2RGBA)
    val bitmap = Bitmap.createBitmap(bgr.cols(), bgr.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(rgba, bitmap)
    rgba.release()
    return bitmap
  }

  /** Decodes a gallery / shared image, honoring EXIF rotation, at most [MAX_SIDE] px. */
  fun decode(context: Context, uri: Uri): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
      decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
      val w = info.size.width
      val h = info.size.height
      if (max(w, h) > MAX_SIDE) {
        val s = MAX_SIDE.toDouble() / max(w, h)
        decoder.setTargetSize((w * s).toInt(), (h * s).toInt())
      }
    }
  }

  /** Copy of [image] with the polygons drawn on it. */
  fun drawBoxes(image: Bitmap, boxes: List<List<Point>>, color: Int, strokeWidth: Float): Bitmap {
    val out = image.copy(Bitmap.Config.ARGB_8888, true)
    val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      style = Paint.Style.STROKE
      this.color = color
      this.strokeWidth = strokeWidth
    }
    val canvas = Canvas(out)
    for (box in boxes) {
      val path = Path()
      box.forEachIndexed { i, p ->
        if (i == 0) path.moveTo(p.x.toFloat(), p.y.toFloat()) else path.lineTo(p.x.toFloat(), p.y.toFloat())
      }
      path.close()
      canvas.drawPath(path, paint)
    }
    return out
  }
}
