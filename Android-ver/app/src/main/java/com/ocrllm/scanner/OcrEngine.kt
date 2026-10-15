package com.ocrllm.scanner

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import org.opencv.core.Mat
import org.opencv.core.Point
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.max

/**
 * OCR engine: PP-OCRv6 text detection + recognition on ONNX Runtime.
 *
 * Android port of Desktop-ver/ocr_app/engine.py. It reproduces PaddleX's OCR
 * pipeline step by step (DB post-processing, Clipper unclip, crop, batched CTC
 * recognition), so the same image gives the same text as the desktop app.
 *
 * PP-OCRv6 recognizers cover Chinese, English, Japanese and Latin scripts, but
 * not Korean. Korean mode pairs the PP-OCRv6 detector (language-agnostic) with
 * PaddleOCR's dedicated Korean recognizer, which also reads English and digits.
 */
enum class Language(val label: String, val recModel: String?) {
  KOREAN("Korean (+English)", "korean_PP-OCRv5_mobile_rec"),
  // null -> use the PP-OCRv6 recognizer of the selected tier
  MULTI("PP-OCRv6 multilingual (CN/EN/JP/Latin)", null),
}

val TIERS = listOf("medium", "small", "tiny")

/** Detector input is scaled down so its longest side is at most this (the desktop uses 4000). */
val DET_SIDES = listOf(960, 1280, 1920, 2560, 4000)
const val DEFAULT_DET_SIDE = 1920

data class OcrLine(val text: String, val score: Float, val box: List<Point>)

data class OcrResult(val lines: List<OcrLine>, val seconds: Double) {
  val text: String get() = lines.joinToString("\n") { it.text }
}

class OcrEngine(
  context: Context,
  val language: Language = Language.KOREAN,
  val tier: String = "medium",
  val detMaxSide: Int = DEFAULT_DET_SIDE,
) : Closeable {
  val detModel = "PP-OCRv6_${tier}_det"
  val recModel = language.recModel ?: "PP-OCRv6_${tier}_rec"

  private val env = OrtEnvironment.getEnvironment()
  private val detSession: OrtSession
  private val recSession: OrtSession
  private val detector: TextDetector
  private val recognizer: TextRecognizer

  init {
    require(tier in TIERS) { "Unknown tier: $tier" }
    val app = context.applicationContext
    for (name in listOf(detModel, recModel)) {
      require(isModelBundled(app, name)) { "Model $name is not bundled in this APK" }
    }
    detSession = createSession(app, detModel)
    recSession = try {
      createSession(app, recModel)
    } catch (e: Exception) {
      detSession.close()
      throw e
    }
    detector = TextDetector(env, detSession, detMaxSide)
    recognizer = TextRecognizer(env, recSession, loadCharacters(app, recModel))
    warmup()
  }

  /** First inference compiles kernels. Do it while the status still says "loading". */
  private fun warmup() {
    val tiny = org.opencv.core.Mat.zeros(64, 64, org.opencv.core.CvType.CV_8UC3)
    try {
      recognize(tiny, 0.99f)
    } catch (_: Exception) {
    } finally {
      tiny.release()
    }
  }

  fun recognize(imageBgr: Mat, minScore: Float = 0.5f): OcrResult {
    val start = SystemClock.elapsedRealtimeNanos()
    val boxes = detector.detect(imageBgr)
    val crops = boxes.map { cropTextLine(imageBgr, it) }
    val recognized = try {
      recognizer.recognize(crops)
    } finally {
      crops.forEach { it.release() }
    }
    val lines = boxes.zip(recognized)
      .filter { (_, rec) -> rec.score >= minScore && rec.text.isNotBlank() }
      .map { (box, rec) -> OcrLine(rec.text, rec.score, box) }
    val seconds = (SystemClock.elapsedRealtimeNanos() - start) / 1e9
    return OcrResult(readingOrder(lines), seconds)
  }

  override fun close() {
    detSession.close()
    recSession.close()
  }

  // ONNX Runtime's CPU provider. (The XNNPACK provider crashes on these models, and
  // NNAPI can't run their dynamic input shapes, so neither is offered.)
  private fun createSession(context: Context, name: String): OrtSession {
    val options = OrtSession.SessionOptions().apply {
      setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
      // Big cores only on typical big.LITTLE phones.
      setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
      // Hide harmless warnings (unused initializers in the converted models).
      setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)
    }
    return options.use { env.createSession(mapModel(context, name), it) }
  }

  companion object {
    private const val MODELS_DIR = "models"

    /** Models packed into the APK by the bundleOcrModels Gradle task. */
    fun bundledModels(context: Context): Set<String> =
      context.assets.list(MODELS_DIR)?.toSet() ?: emptySet()

    fun isModelBundled(context: Context, name: String) = name in bundledModels(context)

    /** Tiers whose detector (and multilingual recognizer) are bundled. */
    fun bundledTiers(context: Context): List<String> {
      val models = bundledModels(context)
      return TIERS.filter { "PP-OCRv6_${it}_det" in models }
    }

    /** Memory-maps the (uncompressed) ONNX file straight from the APK. */
    private fun mapModel(context: Context, name: String): ByteBuffer =
      context.assets.openFd("$MODELS_DIR/$name/inference.onnx").use { fd ->
        FileInputStream(fd.fileDescriptor).channel.use { channel ->
          channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
      }

    /**
     * CTC character table: "blank" + PostProcess.character_dict from inference.yml + " ".
     * The dictionary is a YAML list of plain or single-quoted scalars, one per line.
     */
    private val characterCache = HashMap<String, List<String>>()

    fun loadCharacters(context: Context, name: String): List<String> {
      synchronized(characterCache) { characterCache[name]?.let { return it } }
      val chars = mutableListOf("blank")
      var inDict = false
      context.assets.open("$MODELS_DIR/$name/inference.yml").bufferedReader(Charsets.UTF_8).useLines { lines ->
        for (line in lines) {
          if (!inDict) {
            inDict = line.trim() == "character_dict:"
            continue
          }
          if (!line.startsWith("  - ")) break
          var value = line.substring(4)
          if (value.length >= 2 && value.first() == '\'' && value.last() == '\'') {
            value = value.substring(1, value.length - 1).replace("''", "'")
          }
          chars.add(value)
        }
      }
      check(chars.size > 1) { "No character_dict in $name/inference.yml" }
      chars.add(" ")
      synchronized(characterCache) { characterCache[name] = chars }
      return chars
    }

    /** Sort lines top-to-bottom, then left-to-right within the same row (as on desktop). */
    fun readingOrder(lines: List<OcrLine>): List<OcrLine> {
      if (lines.isEmpty()) return lines
      fun meanY(l: OcrLine) = l.box.sumOf { it.y } / l.box.size
      fun minX(l: OcrLine) = l.box.minOf { it.x }
      val heights = lines.map { l -> max(1.0, l.box.maxOf { it.y } - l.box.minOf { it.y }) }.sorted()
      val mid = heights.size / 2
      val median = if (heights.size % 2 == 1) heights[mid] else (heights[mid - 1] + heights[mid]) / 2
      val rowTol = median * 0.5
      val ordered = lines.sortedWith(compareBy({ meanY(it) }, { minX(it) }))
      val rows = mutableListOf<MutableList<OcrLine>>()
      for (line in ordered) {
        val row = rows.lastOrNull()
        if (row != null && abs(meanY(row[0]) - meanY(line)) <= rowTol) row.add(line) else rows.add(mutableListOf(line))
      }
      return rows.flatMap { row -> row.sortedBy { minX(it) } }
    }
  }
}
