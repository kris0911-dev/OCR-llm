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
import kotlin.math.abs
import kotlin.math.atan2
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

  /**
   * Order as top-left, top-right, bottom-right, bottom-left.
   * The sum/diff trick is kept when it yields four distinct corners.
   * Near a 45 degree tilt it collapses two corners and the warp goes black,
   * so that case walks the quad around its centroid instead.
   */
  fun orderCorners(pts: List<Point>): List<Point> {
    val legacy = legacyOrder(pts)
    if (distinctCorners(legacy) && quadArea(legacy) > 1.0) return legacy
    return robustOrder(pts) ?: legacy
  }

  /** Perspective-correct the page. Null when the quad is degenerate. */
  fun warpDocument(frameBgr: Mat, corners: List<Point>): Mat? {
    val ordered = orderCorners(corners)
    if (!distinctCorners(ordered) || quadArea(ordered) < 1.0) return null
    val (tl, tr, br, bl) = ordered
    val width = max(dist(br, bl), dist(tr, tl)).toInt()
    val height = max(dist(tr, br), dist(tl, bl)).toInt()
    if (width < 2 || height < 2) return null
    val src = MatOfPoint2f(tl, tr, br, bl)
    val dst = MatOfPoint2f(
      Point(0.0, 0.0),
      Point((width - 1).toDouble(), 0.0),
      Point((width - 1).toDouble(), (height - 1).toDouble()),
      Point(0.0, (height - 1).toDouble()),
    )
    val matrix = Imgproc.getPerspectiveTransform(src, dst)
    val out = Mat()
    return try {
      Imgproc.warpPerspective(frameBgr, out, matrix, Size(width.toDouble(), height.toDouble()))
      out
    } catch (e: Exception) {
      out.release()
      null
    } finally {
      listOf(src, dst, matrix).forEach { it.release() }
    }
  }

  private fun legacyOrder(pts: List<Point>): List<Point> {
    val sum = pts.map { it.x + it.y }
    val diff = pts.map { it.y - it.x }
    return listOf(
      pts[sum.indexOf(sum.min())], pts[diff.indexOf(diff.min())],
      pts[sum.indexOf(sum.max())], pts[diff.indexOf(diff.max())],
    )
  }

  private fun robustOrder(pts: List<Point>): List<Point>? {
    if (pts.size != 4) return null
    val cx = pts.sumOf { it.x } / 4.0
    val cy = pts.sumOf { it.y } / 4.0
    val ring = pts.sortedBy { atan2(it.y - cy, it.x - cx) }
    return bestRotation(ring) ?: bestRotation(ring.asReversed())
  }

  private fun bestRotation(ring: List<Point>): List<Point>? {
    var best: List<Point>? = null
    var bestTop = Double.POSITIVE_INFINITY
    var bestHoriz = Double.POSITIVE_INFINITY
    for (i in ring.indices) {
      val ordered = List(4) { ring[(i + it) % ring.size] }
      val (tl, tr, br, bl) = ordered
      val dx = tr.x - tl.x
      val dy = tr.y - tl.y
      if (dx <= 1e-3) continue
      val topY = (tl.y + tr.y) * 0.5
      val botY = (bl.y + br.y) * 0.5
      if (topY >= botY) continue
      val horiz = abs(dy) / (abs(dx) + 1e-6)
      if (topY < bestTop || (topY == bestTop && horiz < bestHoriz)) {
        bestTop = topY
        bestHoriz = horiz
        best = ordered
      }
    }
    return best
  }

  private fun distinctCorners(pts: List<Point>): Boolean {
    val kept = ArrayList<Point>()
    for (p in pts) {
      if (kept.none { hypot(p.x - it.x, p.y - it.y) <= 1.0 }) kept.add(p)
    }
    return kept.size == 4
  }

  private fun quadArea(pts: List<Point>): Double {
    var area = 0.0
    for (i in pts.indices) {
      val a = pts[i]
      val b = pts[(i + 1) % pts.size]
      area += a.x * b.y - b.x * a.y
    }
    return abs(area) * 0.5
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

  /** Shrink so the long side is at most [maxSide]. Recycles [bitmap] when a copy is made. */
  fun limit(bitmap: Bitmap, maxSide: Int): Bitmap {
    val longest = max(bitmap.width, bitmap.height)
    if (longest <= maxSide || longest == 0) return bitmap
    val scale = maxSide.toDouble() / longest
    val width = (bitmap.width * scale).toInt().coerceAtLeast(1)
    val height = (bitmap.height * scale).toInt().coerceAtLeast(1)
    val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
    if (scaled !== bitmap) bitmap.recycle()
    return scaled
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
