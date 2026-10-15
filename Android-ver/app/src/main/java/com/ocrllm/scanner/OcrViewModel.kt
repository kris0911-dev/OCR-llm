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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
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

class OcrViewModel(app: Application) : AndroidViewModel(app) {
  private val _state = MutableStateFlow(UiState())
  val state: StateFlow<UiState> = _state

  private val lock = Mutex()
  private var engine: OcrEngine? = null
  private var engineKey: List<Any>? = null
  private var source: Bitmap? = null

  val hasImage: Boolean get() = source != null
  val busy: Boolean get() = _state.value.busy

  /** Runs OCR on [bitmap]; with [flatten], first detects the page and corrects perspective. */
  fun process(bitmap: Bitmap, flatten: Boolean, settings: OcrSettings) {
    if (busy) {
      _state.update { it.copy(status = "Still working on the previous image...") }
      return
    }
    _state.update { it.copy(image = bitmap, busy = true, error = null, status = "Preparing image...") }
    viewModelScope.launch {
      try {
        lock.withLock { withContext(Dispatchers.Default) { run(bitmap, flatten, settings) } }
      } catch (e: Throwable) {
        Log.e(TAG, "OCR failed", e)
        _state.update { it.copy(busy = false, status = "OCR failed", error = e.message ?: e.toString()) }
      }
    }
  }

  /** Re-runs OCR on the last image (e.g. after changing settings). */
  fun rerun(settings: OcrSettings) {
    source?.let { process(it, flatten = false, settings = settings) }
  }

  fun clearError() = _state.update { it.copy(error = null) }

  fun clearImage() {
    if (!busy) _state.update { it.copy(image = null) }
  }

  private fun run(bitmap: Bitmap, flatten: Boolean, settings: OcrSettings) {
    var image = Images.bitmapToBgr(bitmap)
    if (flatten) {
      Scanner.findDocument(image)?.let { quad ->
        val warped = Scanner.warpDocument(image, quad)
        image.release()
        image = warped
      }
    }
    try {
      val page = if (flatten) Images.bgrToBitmap(image) else bitmap
      source = page
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
      val result = current.recognize(image, settings.minScore)
      val avg = if (result.lines.isEmpty()) 0f else result.lines.map { it.score }.average().toFloat()
      val boxed = Images.drawBoxes(page, result.lines.map { it.box }, BOX_COLOR, max(2f, page.width / 400f))
      val status = String.format(
        Locale.US, "%d lines recognized (avg confidence %.2f) in %.2fs on CPU, %s tier",
        result.lines.size, avg, result.seconds, settings.tier,
      )
      _state.update { it.copy(image = boxed, text = result.text, status = status, busy = false) }
    } finally {
      image.release()
    }
  }

  override fun onCleared() {
    engine?.close()
    engine = null
  }

  private fun max(a: Float, b: Float) = if (a > b) a else b

  companion object {
    private const val TAG = "OcrViewModel"
    private val BOX_COLOR = Color.rgb(0, 200, 0)
  }
}
