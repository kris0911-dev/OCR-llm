package com.ocrllm.scanner

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import java.util.Locale
import kotlin.math.abs
import kotlin.math.min

/**
 * Runs the Android OCR pipeline on the test pages in androidTest/assets/testimg and
 * compares it with the text the desktop app (PaddleOCR) produced for the same images
 * (androidTest/assets/expected). Run with: gradlew connectedDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class OcrEngineTest {
  private val app = InstrumentationRegistry.getInstrumentation().targetContext
  private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets

  @Test fun koreanMedium() = compare(Language.KOREAN, "medium")
  @Test fun koreanSmall() = compare(Language.KOREAN, "small")
  @Test fun koreanTiny() = compare(Language.KOREAN, "tiny")
  @Test fun multilingualTiny() = compare(Language.MULTI, "tiny")

  /** Page detection + flattening on the camera-like images, then OCR. */
  @Test
  fun scanCameraPages() {
    OcrEngine(app, Language.KOREAN, "small", detMaxSide = Images.MAX_SIDE).use { engine ->
      val stats = Stats()
      for ((name, image) in images().filter { it.first.startsWith("camera_") }) {
        val expectedQuad = read("expected/quad_$name.txt").trim().split(" ").map { it.toDouble() }
        val quad = Scanner.findDocument(image)
        assertNotNull("page not found in $name", quad)
        val got = quad!!.flatMap { listOf(it.x, it.y) }
        val maxErr = got.zip(expectedQuad).maxOf { (a, b) -> abs(a - b) }
        assertTrue("$name corners off by $maxErr px", maxErr < 1.0)

        val page = Scanner.warpDocument(image, quad)
        assertNotNull("warp failed for $name", page)
        stats.add(name, engine.recognize(page!!, 0.5f), read("expected/scan_korean_small_$name.txt"))
        page.release()
        image.release()
      }
      stats.check("scan korean small")
    }
  }

  @Test
  fun dictionarySizesMatchModels() {
    for ((model, size) in listOf(
      "korean_PP-OCRv5_mobile_rec" to 11947, "PP-OCRv6_tiny_rec" to 6906, "PP-OCRv6_small_rec" to 18710,
    )) {
      if (OcrEngine.isModelBundled(app, model)) {
        assertEquals(model, size, OcrEngine.loadCharacters(app, model).size)
      }
    }
  }

  private fun compare(language: Language, tier: String) {
    if ("PP-OCRv6_${tier}_det" !in OcrEngine.bundledModels(app)) {
      Log.w(TAG, "$tier tier not bundled, skipping")
      return
    }
    val lang = language.name.lowercase(Locale.US)
    // detMaxSide = 4000 reproduces the desktop pipeline (no downscaling of these pages).
    OcrEngine(app, language, tier, detMaxSide = Images.MAX_SIDE).use { engine ->
      val stats = Stats()
      for ((name, image) in images()) {
        stats.add(name, engine.recognize(image, 0.5f), read("expected/${lang}_${tier}_$name.txt"))
        image.release()
      }
      stats.check("$lang $tier")
    }
  }

  private class Stats {
    var pages = 0
    var identical = 0
    var edits = 0
    var chars = 0
    val millis = mutableListOf<Double>()

    fun add(name: String, result: OcrResult, expected: String) {
      pages++
      millis += result.seconds * 1000
      if (result.text == expected) {
        identical++
      } else {
        Log.i(TAG, "$name differs from desktop:\n${result.text}\n--- desktop ---\n$expected")
      }
      edits += editDistance(result.text, expected)
      chars += expected.length
    }

    fun check(label: String) {
      val cer = 100.0 * edits / chars.coerceAtLeast(1)
      val summary = String.format(
        Locale.US, "%s: %d/%d pages identical to desktop, char diff %.2f%%, mean %.0f ms/page",
        label, identical, pages, cer, millis.average())
      Log.i(TAG, summary)
      println(summary)
      // Floating point differences between CPUs may flip a rare character; allow 1%.
      assertTrue(summary, cer < 1.0)
    }
  }

  private fun images(): List<Pair<String, Mat>> =
    testAssets.list("testimg")!!.filter { it.endsWith(".png") }.sorted().map { file ->
      val bitmap = testAssets.open("testimg/$file").use { BitmapFactory.decodeStream(it) }
      file.removeSuffix(".png") to Images.bitmapToBgr(bitmap)
    }

  private fun read(path: String) = testAssets.open(path).use { it.readBytes().toString(Charsets.UTF_8) }

  companion object {
    private const val TAG = "OcrEngineTest"

    @JvmStatic
    @BeforeClass
    fun loadOpenCv() {
      assertTrue(OpenCVLoader.initLocal())
    }

    fun editDistance(a: String, b: String): Int {
      val x = a.codePoints().toArray()
      val y = b.codePoints().toArray()
      var prev = IntArray(y.size + 1) { it }
      for (i in 1..x.size) {
        val cur = IntArray(y.size + 1)
        cur[0] = i
        for (j in 1..y.size) {
          cur[j] = min(min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + if (x[i - 1] == y[j - 1]) 0 else 1)
        }
        prev = cur
      }
      return prev[y.size]
    }
  }
}
