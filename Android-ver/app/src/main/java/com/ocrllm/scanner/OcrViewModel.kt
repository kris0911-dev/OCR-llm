package com.ocrllm.scanner

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Core
import org.opencv.core.Mat
import java.util.Locale

data class OcrSettings(
  val language: Language,
  val tier: String,
  val detMaxSide: Int,
  val minScore: Float,
)

data class UiState(
  /** Image shown instead of the camera preview (captured or opened), with text boxes once done. */
  val image: Bitmap? = null,
  val text: String = "",
  val status: String = "",
  val busy: Boolean = false,
  val error: String? = null,
)

private class Pending(val bitmap: Bitmap, val flatten: Boolean, val settings: OcrSettings)

class OcrViewModel(app: Application) : AndroidViewModel(app) {
  private val _state = MutableStateFlow(UiState())
  val state: StateFlow<UiState> = _state

  private val queueLock = Any()
  private var running = false
  private var generation = 0
  private var pending: Pending? = null
  private var source: Bitmap? = null
  private var orphan: Bitmap? = null

  val hasImage: Boolean get() = source != null
  val busy: Boolean get() = _state.value.busy

  /** True when [bitmap] is still needed for a rerun or a queued job. */
  fun retains(bitmap: Bitmap): Boolean {
    if (bitmap === source) return true
    val queued = synchronized(queueLock) { pending?.bitmap }
    return bitmap === queued
  }

  /** Runs OCR on [bitmap]; with [flatten], first detects the page and corrects perspective. */
  fun process(bitmap: Bitmap, flatten: Boolean, settings: OcrSettings) {
    val myGen = synchronized(queueLock) {
      if (running) {
        pending = Pending(bitmap, flatten, settings)
        _state.update { it.copy(image = bitmap, status = "This image will run next.") }
        return
      }
      running = true
      ++generation
    }
    _state.update { it.copy(image = bitmap, busy = true, error = null, status = "Preparing image...") }
    viewModelScope.launch {
      var job = Pending(bitmap, flatten, settings)
      try {
        while (true) {
          withContext(Dispatchers.Default) { run(job.bitmap, job.flatten, job.settings) }
          val next = synchronized(queueLock) {
            val queued = pending
            pending = null
            queued
          } ?: break
          job = next
          _state.update { it.copy(image = job.bitmap, status = "Preparing image...") }
        }
      } catch (e: Throwable) {
        Log.e(TAG, "OCR failed", e)
        synchronized(queueLock) { pending = null }
        _state.update { it.copy(status = "OCR failed", error = e.message ?: e.toString()) }
      } finally {
        val clear = synchronized(queueLock) {
          if (generation == myGen) {
            running = false
            true
          } else {
            false
          }
        }
        if (clear) _state.update { it.copy(busy = false) }
      }
    }
  }

  /** Re-runs OCR on the last image (e.g. after changing settings). */
  fun rerun(settings: OcrSettings) {
    source?.let { process(it, flatten = false, settings = settings) }
  }

  fun clearError() = _state.update { it.copy(error = null) }

  fun clearImage() {
    if (busy) return
    source?.let { stash(it) }
    source = null
    _state.update { it.copy(image = null, text = "") }
  }

  /**
   * Drops a bitmap the screen has moved on from. Call after the ImageView has
   * been given [UiState.image], so a bitmap still on screen is never recycled.
   */
  fun releaseUnused() {
    synchronized(queueLock) {
      val old = orphan ?: return
      if (old === source || old === _state.value.image || old === pending?.bitmap) return
      orphan = null
      if (!old.isRecycled) old.recycle()
    }
  }

  private fun run(bitmap: Bitmap, flatten: Boolean, settings: OcrSettings) {
    var image = Images.bitmapToBgr(bitmap)
    if (flatten) {
      Scanner.findDocument(image)?.let { quad ->
        val warped = Scanner.warpDocument(image, quad)
        if (warped != null) {
          image.release()
          image = warped
        }
      }
    }
    var page = if (flatten) Images.bgrToBitmap(image) else bitmap
    try {
      replaceSource(page)
      _state.update { it.copy(image = page) }

      val key = listOf(settings.language, settings.tier, settings.detMaxSide)
      val current = engine?.takeIf { engineKey == key } ?: run {
        _state.update { it.copy(status = "Loading OCR models...") }
        engine?.close()
        engine = null
        OcrEngine(getApplication(), settings.language, settings.tier, settings.detMaxSide)
          .also { engine = it; engineKey = key }
      }

      _state.update { it.copy(status = "Recognizing text...") }
      var result = current.recognize(image, settings.minScore)
      var avg = averageScore(result)
      // A low score usually means the page is upside down. Try once, keep the better read.
      if (avg < UPSIDE_DOWN_RETRY) {
        val flipped = Mat()
        Core.rotate(image, flipped, Core.ROTATE_180)
        val second = current.recognize(flipped, settings.minScore)
        val secondAvg = averageScore(second)
        if (secondAvg > avg + 0.05f) {
          result = second
          avg = secondAvg
          page = Images.bgrToBitmap(flipped)
          _state.update { it.copy(image = page) }
          replaceSource(page)
        }
        flipped.release()
      }
      val boxed = Images.drawBoxes(page, result.lines.map { it.box }, BOX_COLOR, max(2f, page.width / 400f))
      val status = String.format(
        Locale.US, "%d lines recognized (avg confidence %.2f) in %.2fs on CPU, %s tier",
        result.lines.size, avg, result.seconds, settings.tier,
      )
      _state.update { it.copy(image = boxed, text = result.text, status = status) }
    } finally {
      image.release()
    }
  }

  private fun replaceSource(page: Bitmap) {
    val previous = source
    source = page
    if (previous != null && previous !== page) stash(previous)
  }

  private fun stash(bitmap: Bitmap) {
    synchronized(queueLock) {
      val old = orphan
      orphan = bitmap
      if (old != null && old !== bitmap && old !== source && old !== _state.value.image &&
        old !== pending?.bitmap && !old.isRecycled
      ) {
        old.recycle()
      }
    }
  }

  private fun averageScore(result: OcrResult): Float =
    if (result.lines.isEmpty()) 0f else result.lines.map { it.score }.average().toFloat()

  override fun onCleared() {
    engine?.close()
    engine = null
    source?.recycle()
    source = null
    synchronized(queueLock) {
      pending?.bitmap?.let { if (!it.isRecycled) it.recycle() }
      pending = null
      orphan?.let { if (!it.isRecycled) it.recycle() }
      orphan = null
    }
  }

  private var engine: OcrEngine? = null
  private var engineKey: List<Any>? = null

  private fun max(a: Float, b: Float) = if (a > b) a else b

  companion object {
    private const val TAG = "OcrViewModel"
    private const val UPSIDE_DOWN_RETRY = 0.62f
    private val BOX_COLOR = Color.rgb(0, 200, 0)
  }
}
