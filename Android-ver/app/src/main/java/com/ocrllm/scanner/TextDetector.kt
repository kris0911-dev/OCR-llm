package com.ocrllm.scanner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.RotatedRect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * PP-OCRv6 text detector (DB) with PaddleX's pre/post-processing:
 * DetResizeForTest -> NormalizeImage -> model -> DBPostProcess (quad) -> SortQuadBoxes.
 */
internal class TextDetector(
  private val env: OrtEnvironment,
  private val session: OrtSession,
  private val maxSide: Int,
) {
  // PaddleX OCR pipeline defaults (paddlex/configs/pipelines/OCR.yaml).
  private val thresh = 0.3
  private val boxThresh = 0.6
  private val unclipRatio = 1.5
  private val maxCandidates = 1000
  private val minSize = 3
  private val limitSideLen = 64

  private val inputName = session.inputNames.first()

  /** Returns quads (top-left, top-right, bottom-right, bottom-left) in image pixels. */
  fun detect(imageBgr: Mat): List<List<Point>> {
    val srcH = imageBgr.rows()
    val srcW = imageBgr.cols()
    val (resized, h, w) = resize(imageBgr)
    val pred: FloatArray
    try {
      OnnxTensor.createTensor(env, normalize(resized), longArrayOf(1, 3, h.toLong(), w.toLong())).use { input ->
        session.run(mapOf(inputName to input)).use { output ->
          val tensor = output[0] as OnnxTensor
          val buffer = tensor.floatBuffer
          pred = FloatArray(buffer.remaining()).also { buffer.get(it) }
        }
      }
    } finally {
      if (resized !== imageBgr) resized.release()
    }
    val predMat = Mat(h, w, CvType.CV_32F).apply { put(0, 0, pred) }
    try {
      return sortQuadBoxes(boxesFromBitmap(predMat, pred, w, h, srcW, srcH))
    } finally {
      predMat.release()
    }
  }

  /** DetResizeForTest (limit_type "min", 64) + max side limit; sides become multiples of 32. */
  private fun resize(image: Mat): Triple<Mat, Int, Int> {
    val h = image.rows()
    val w = image.cols()
    var ratio = if (min(h, w) < limitSideLen) limitSideLen.toDouble() / min(h, w) else 1.0
    var rh = (h * ratio).toInt()
    var rw = (w * ratio).toInt()
    if (max(rh, rw) > maxSide) {
      ratio = maxSide.toDouble() / max(rh, rw)
      rh = (rh * ratio).toInt()
      rw = (rw * ratio).toInt()
    }
    // Python's round() is round-half-to-even, like Math.rint.
    rh = max((Math.rint(rh / 32.0) * 32).toInt(), 32)
    rw = max((Math.rint(rw / 32.0) * 32).toInt(), 32)
    if (rh == h && rw == w) return Triple(image, h, w)
    val out = Mat()
    Imgproc.resize(image, out, Size(rw.toDouble(), rh.toDouble()))
    return Triple(out, rh, rw)
  }

  /** BGR uint8 HWC -> (x / 255 - mean) / std, CHW float. */
  private fun normalize(image: Mat): FloatBuffer {
    val h = image.rows()
    val w = image.cols()
    val plane = h * w
    val pixels = ByteArray(plane * 3)
    image.get(0, 0, pixels)
    val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val std = floatArrayOf(0.229f, 0.224f, 0.225f)
    val out = FloatArray(plane * 3)
    for (c in 0 until 3) {
      val m = mean[c]
      val s = std[c]
      val base = c * plane
      for (i in 0 until plane) {
        out[base + i] = ((pixels[i * 3 + c].toInt() and 0xFF) / 255f - m) / s
      }
    }
    return FloatBuffer.wrap(out)
  }

  private fun boxesFromBitmap(
    predMat: Mat, pred: FloatArray, width: Int, height: Int, destW: Int, destH: Int,
  ): List<List<Point>> {
    val bitmap = ByteArray(pred.size) { if (pred[it] > thresh) 255.toByte() else 0 }
    val bitmapMat = Mat(height, width, CvType.CV_8U).apply { put(0, 0, bitmap) }
    val contours = ArrayList<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(bitmapMat, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
    bitmapMat.release()
    hierarchy.release()

    val scaleX = destW.toDouble() / width
    val scaleY = destH.toDouble() / height
    val boxes = ArrayList<List<Point>>()
    for (contour in contours.take(maxCandidates)) {
      val (points, sside) = contour2f(contour).use { getMiniBoxes(Imgproc.minAreaRect(it)) }
      if (sside < minSize) continue
      val score = boxScoreFast(predMat, points)
      if (score < boxThresh) continue

      val expanded = MatOfPoint2f(*unclip(points, unclipRatio).toTypedArray())
      val (box, sside2) = expanded.use { getMiniBoxes(Imgproc.minAreaRect(it)) }
      if (sside2 < minSize + 2) continue
      boxes.add(box.map { p ->
        Point(
          Math.rint(p.x * scaleX).coerceIn(0.0, destW.toDouble()),
          Math.rint(p.y * scaleY).coerceIn(0.0, destH.toDouble()),
        )
      })
    }
    contours.forEach { it.release() }
    return boxes
  }

  /** Mean probability inside the box (box_score_fast). */
  private fun boxScoreFast(pred: Mat, box: List<Point>): Double {
    val h = pred.rows()
    val w = pred.cols()
    val xmin = floor(box.minOf { it.x }).toInt().coerceIn(0, w - 1)
    val xmax = ceil(box.maxOf { it.x }).toInt().coerceIn(0, w - 1)
    val ymin = floor(box.minOf { it.y }).toInt().coerceIn(0, h - 1)
    val ymax = ceil(box.maxOf { it.y }).toInt().coerceIn(0, h - 1)
    val mask = Mat.zeros(ymax - ymin + 1, xmax - xmin + 1, CvType.CV_8U)
    // numpy .astype(int32) truncates toward zero, as does toInt().
    val poly = MatOfPoint(*box.map { Point((it.x - xmin).toInt().toDouble(), (it.y - ymin).toInt().toDouble()) }.toTypedArray())
    Imgproc.fillPoly(mask, listOf(poly), org.opencv.core.Scalar(1.0))
    val roi = pred.submat(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
    val score = Core.mean(roi, mask).`val`[0]
    roi.release()
    mask.release()
    poly.release()
    return score
  }

  companion object {
    /** minAreaRect corners ordered top-left, top-right, bottom-right, bottom-left (get_mini_boxes). */
    fun getMiniBoxes(rect: RotatedRect): Pair<List<Point>, Double> {
      val corners = Mat()
      Imgproc.boxPoints(rect, corners)
      val raw = FloatArray(8).also { corners.get(0, 0, it) }
      corners.release()
      // boxPoints returns float32; keep that precision like numpy does.
      val pts = (0 until 4).map { Point(raw[it * 2].toDouble(), raw[it * 2 + 1].toDouble()) }.sortedBy { it.x }
      val (i1, i4) = if (pts[1].y > pts[0].y) 0 to 1 else 1 to 0
      val (i2, i3) = if (pts[3].y > pts[2].y) 2 to 3 else 3 to 2
      return listOf(pts[i1], pts[i2], pts[i3], pts[i4]) to min(rect.size.width, rect.size.height)
    }

    /** Expands the box by area * ratio / perimeter like pyclipper (round joins, integer grid). */
    fun unclip(box: List<Point>, ratio: Double): List<Point> {
      val poly = MatOfPoint2f(*box.toTypedArray())
      val distance = Imgproc.contourArea(poly) * ratio / Imgproc.arcLength(poly, true)
      poly.release()
      return ClipperOffset.offsetRound(box, distance)
    }

    /** PaddleX SortQuadBoxes: top-to-bottom, then left-to-right for boxes within 10 px vertically. */
    fun sortQuadBoxes(boxes: List<List<Point>>): List<List<Point>> {
      val b = boxes.sortedWith(compareBy({ it[0].y }, { it[0].x })).toMutableList()
      for (i in 0 until b.size - 1) {
        for (j in i downTo 0) {
          if (abs(b[j + 1][0].y - b[j][0].y) < 10 && b[j + 1][0].x < b[j][0].x) {
            val tmp = b[j]; b[j] = b[j + 1]; b[j + 1] = tmp
          } else {
            break
          }
        }
      }
      return b
    }

    private fun contour2f(contour: MatOfPoint) = MatOfPoint2f(*contour.toArray())

    private inline fun <T : Mat, R> T.use(block: (T) -> R): R = try {
      block(this)
    } finally {
      release()
    }
  }
}

/**
 * Port of ClipperOffset (Clipper 6.4.2, as used by pyclipper) for one closed convex
 * polygon with round joins and a positive delta. Coordinates snap to integers exactly
 * as in pyclipper, so the boxes match the desktop app to the pixel.
 */
internal object ClipperOffset {
  private const val ARC_TOLERANCE = 0.25

  private fun round(v: Double): Long = if (v < 0) (v - 0.5).toLong() else (v + 0.5).toLong()

  fun offsetRound(path: List<Point>, delta: Double): List<Point> {
    // pyclipper converts float input to integers by truncation.
    val pts = path.map { longArrayOf(it.x.toLong(), it.y.toLong()) }
    var hi = pts.size - 1
    while (hi > 0 && pts[0].contentEquals(pts[hi])) hi--
    val src = ArrayList<LongArray>()
    src.add(pts[0])
    for (i in 1..hi) if (!pts[i].contentEquals(src.last())) src.add(pts[i])
    if (src.size < 3 || delta <= 0) return src.map { Point(it[0].toDouble(), it[1].toDouble()) }

    val n = src.size
    var area = 0.0
    for (i in 0 until n) {
      val a = src[i]
      val b = src[(i + 1) % n]
      area += a[0].toDouble() * b[1] - b[0].toDouble() * a[1]
    }
    if (area < 0) src.reverse()

    val y = if (ARC_TOLERANCE > abs(delta) * 0.25) abs(delta) * 0.25 else ARC_TOLERANCE
    var steps = Math.PI / acos(1 - y / abs(delta))
    if (steps > abs(delta) * Math.PI) steps = abs(delta) * Math.PI
    val sinStep = sin(2 * Math.PI / steps)
    val cosStep = cos(2 * Math.PI / steps)
    val stepsPerRad = steps / (2 * Math.PI)

    val normals = (0 until n).map { j ->
      val a = src[j]
      val b = src[(j + 1) % n]
      val dx = (b[0] - a[0]).toDouble()
      val dy = (b[1] - a[1]).toDouble()
      if (dx == 0.0 && dy == 0.0) doubleArrayOf(0.0, 0.0)
      else (1 / sqrt(dx * dx + dy * dy)).let { f -> doubleArrayOf(dy * f, -dx * f) }
    }

    val out = ArrayList<Point>()
    fun add(x: Double, y2: Double) = out.add(Point(round(x).toDouble(), round(y2).toDouble()))
    var k = n - 1
    for (j in 0 until n) {
      val nk = normals[k]
      val nj = normals[j]
      val px = src[j][0].toDouble()
      val py = src[j][1].toDouble()
      var sinA = nk[0] * nj[1] - nj[0] * nk[1]
      if (abs(sinA * delta) < 1.0) {
        if (nk[0] * nj[0] + nj[1] * nk[1] > 0) {
          add(px + nk[0] * delta, py + nk[1] * delta)
          k = j
          continue
        }
      } else if (sinA > 1.0) {
        sinA = 1.0
      } else if (sinA < -1.0) {
        sinA = -1.0
      }
      if (sinA * delta < 0) {
        add(px + nk[0] * delta, py + nk[1] * delta)
        out.add(Point(px, py))
        add(px + nj[0] * delta, py + nj[1] * delta)
      } else {
        val a = atan2(sinA, nk[0] * nj[0] + nk[1] * nj[1])
        val count = max(round(stepsPerRad * abs(a)).toInt(), 1)
        var x = nk[0]
        var yy = nk[1]
        repeat(count) {
          add(px + x * delta, py + yy * delta)
          val x2 = x
          x = x * cosStep - sinStep * yy
          yy = x2 * sinStep + yy * cosStep
        }
        add(px + nj[0] * delta, py + nj[1] * delta)
      }
      k = j
    }
    return out
  }
}
