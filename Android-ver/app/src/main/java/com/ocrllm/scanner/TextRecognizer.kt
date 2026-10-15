package com.ocrllm.scanner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

internal data class Recognition(val text: String, val score: Float)

/**
 * Cuts one text line out of the image (PaddleX CropByPolys "quad" + get_rotate_crop_image).
 * Lines that are much taller than wide are rotated 90 degrees counter-clockwise.
 */
internal fun cropTextLine(imageBgr: Mat, quad: List<Point>): Mat {
  val intQuad = MatOfPoint(*quad.map { Point(it.x.toInt().toDouble(), it.y.toInt().toDouble()) }.toTypedArray())
  val rectPts = MatOfPoint2f(*intQuad.toArray())
  val (box, _) = TextDetector.getMiniBoxes(Imgproc.minAreaRect(rectPts))
  intQuad.release()
  rectPts.release()

  // numpy computes these norms in float32.
  fun dist(a: Point, b: Point): Float {
    val dx = (a.x - b.x).toFloat()
    val dy = (a.y - b.y).toFloat()
    return sqrt(dx * dx + dy * dy)
  }
  val width = max(dist(box[0], box[1]), dist(box[2], box[3])).toInt()
  val height = max(dist(box[0], box[3]), dist(box[1], box[2])).toInt()
  if (width < 1 || height < 1) return Mat()

  val src = MatOfPoint2f(*box.toTypedArray())
  val dst = MatOfPoint2f(
    Point(0.0, 0.0), Point(width.toDouble(), 0.0),
    Point(width.toDouble(), height.toDouble()), Point(0.0, height.toDouble()),
  )
  val matrix = Imgproc.getPerspectiveTransform(src, dst)
  val crop = Mat()
  Imgproc.warpPerspective(
    imageBgr, crop, matrix, Size(width.toDouble(), height.toDouble()),
    Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE,
  )
  src.release()
  dst.release()
  matrix.release()
  if (crop.rows().toFloat() / crop.cols() >= 1.5f) {
    val rotated = Mat()
    Core.rotate(crop, rotated, Core.ROTATE_90_COUNTERCLOCKWISE)
    crop.release()
    return rotated
  }
  return crop
}

/**
 * Text line recognizer (SVTR-style CTC model), matching PaddleX: crops are sorted by
 * aspect ratio and run in batches of 6, each resized to height 48 and zero-padded to
 * the widest crop in its batch; the output is greedy-decoded (CTCLabelDecode).
 */
internal class TextRecognizer(
  private val env: OrtEnvironment,
  private val session: OrtSession,
  private val characters: List<String>,
) {
  private val inputName = session.inputNames.first()

  fun recognize(crops: List<Mat>): List<Recognition> {
    val results = arrayOfNulls<Recognition>(crops.size)
    val valid = crops.indices.filter { !crops[it].empty() }
    for (i in crops.indices) if (crops[i].empty()) results[i] = Recognition("", 0f)
    val order = valid.sortedBy { ratio(crops[it]) }
    for (batch in order.chunked(BATCH_SIZE)) {
      runBatch(batch.map { crops[it] }).forEachIndexed { k, rec -> results[batch[k]] = rec }
    }
    return results.map { it!! }
  }

  private fun ratio(img: Mat) = img.cols() / img.rows().toDouble()

  /** Target tensor width for one crop: max(320, 48 * w / h), capped at 3200. */
  private fun targetWidth(img: Mat) = min((HEIGHT * max(BASE_RATIO, ratio(img))).toInt(), MAX_WIDTH)

  private fun runBatch(batch: List<Mat>): List<Recognition> {
    val batchWidth = batch.maxOf { targetWidth(it) }
    val plane = HEIGHT * batchWidth
    val data = FloatArray(batch.size * 3 * plane) // zero padding = normalized value 0
    batch.forEachIndexed { b, img ->
      val full = (HEIGHT * max(BASE_RATIO, ratio(img))).toInt()
      val resizedW = if (full > MAX_WIDTH) MAX_WIDTH else min(ceil(HEIGHT * ratio(img)).toInt(), full)
      val resized = Mat()
      Imgproc.resize(img, resized, Size(resizedW.toDouble(), HEIGHT.toDouble()))
      val pixels = ByteArray(resizedW * HEIGHT * 3)
      resized.get(0, 0, pixels)
      resized.release()
      val base = b * 3 * plane
      for (y in 0 until HEIGHT) {
        for (x in 0 until resizedW) {
          val p = (y * resizedW + x) * 3
          val o = base + y * batchWidth + x
          for (c in 0 until 3) {
            data[o + c * plane] = ((pixels[p + c].toInt() and 0xFF) / 255f - 0.5f) / 0.5f
          }
        }
      }
    }
    val shape = longArrayOf(batch.size.toLong(), 3, HEIGHT.toLong(), batchWidth.toLong())
    OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).use { input ->
      session.run(mapOf(inputName to input)).use { output ->
        val tensor = output[0] as OnnxTensor
        val dims = tensor.info.shape // [batch, steps, classes]
        val steps = dims[1].toInt()
        val classes = dims[2].toInt()
        check(classes == characters.size) {
          "Model outputs $classes classes but the dictionary has ${characters.size}"
        }
        val buffer = tensor.floatBuffer
        val probs = FloatArray(buffer.remaining()).also { buffer.get(it) }
        return batch.indices.map { decode(probs, it * steps * classes, steps, classes) }
      }
    }
  }

  /** Greedy CTC: argmax per step, drop repeats and blanks; score = mean prob of kept steps. */
  private fun decode(probs: FloatArray, offset: Int, steps: Int, classes: Int): Recognition {
    val text = StringBuilder()
    var sum = 0.0
    var kept = 0
    var prev = -1
    for (t in 0 until steps) {
      val row = offset + t * classes
      var best = 0
      var bestP = probs[row]
      for (c in 1 until classes) {
        if (probs[row + c] > bestP) {
          bestP = probs[row + c]
          best = c
        }
      }
      if (best != 0 && best != prev) {
        text.append(characters[best])
        sum += bestP
        kept++
      }
      prev = best
    }
    return Recognition(text.toString(), if (kept == 0) 0f else (sum / kept).toFloat())
  }

  private companion object {
    const val BATCH_SIZE = 6
    const val HEIGHT = 48
    const val BASE_RATIO = 320.0 / 48
    const val MAX_WIDTH = 3200
  }
}
