package com.ocrllm.scanner

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraState
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.isVisible
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.ocrllm.scanner.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max

class MainActivity : AppCompatActivity() {
  private lateinit var binding: ActivityMainBinding
  private val viewModel: OcrViewModel by viewModels()
  private val prefs by lazy { getSharedPreferences("settings", MODE_PRIVATE) }

  private lateinit var cameraExecutor: ExecutorService
  private var cameraProvider: ProcessCameraProvider? = null
  private var cameraLive = false
  private var tiers: List<String> = emptyList()
  private var shownText: String? = null

  // Read by the analysis thread.
  @Volatile private var flattenEnabled = true
  @Volatile private var autoCaptureEnabled = false
  @Volatile private var captureRequested = false
  private var anchorQuad: List<Point>? = null
  private var steadySince = 0L

  private val requestCamera =
    registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
      if (granted) startCamera() else binding.stage.hint.setText(R.string.hint_start)
    }

  private val pickImage =
    registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let(::openImage) }

  private val saveText =
    registerForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
      uri?.let(::writeText)
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    check(OpenCVLoader.initLocal()) { "OpenCV native library failed to load" }
    binding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(binding.root)
    binding.toolbar.title = getString(R.string.app_name)
    ViewCompat.setOnApplyWindowInsetsListener(binding.root) { v, insets ->
      val bars = insets.getInsets(
        WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout() or
          WindowInsetsCompat.Type.ime())
      v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
      WindowInsetsCompat.CONSUMED
    }
    cameraExecutor = Executors.newSingleThreadExecutor()

    setupSettings()
    setupButtons()

    lifecycleScope.launch {
      repeatOnLifecycle(Lifecycle.State.STARTED) { viewModel.state.collect(::render) }
    }

    if (savedInstanceState == null) handleShareIntent(intent)
    if (viewModel.state.value.image == null) startCamera()
  }

  override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    handleShareIntent(intent)
  }

  override fun onDestroy() {
    super.onDestroy()
    cameraExecutor.shutdown()
  }

  // ------------------------------------------------------------------ UI

  private fun setupButtons() {
    binding.actions.cameraButton.setOnClickListener {
      viewModel.clearImage()
      startCamera()
    }
    binding.actions.scanButton.setOnClickListener { capture() }
    binding.actions.galleryButton.setOnClickListener {
      pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
    val panel = binding.panel
    panel.rerunButton.setOnClickListener { viewModel.rerun(currentSettings()) }
    panel.copyButton.setOnClickListener {
      val clipboard = getSystemService(ClipboardManager::class.java)
      clipboard.setPrimaryClip(ClipData.newPlainText("OCR text", currentText()))
      Toast.makeText(this, R.string.copied, Toast.LENGTH_SHORT).show()
    }
    panel.shareButton.setOnClickListener {
      val text = currentText()
      if (text.isEmpty()) {
        Toast.makeText(this, R.string.nothing_to_share, Toast.LENGTH_SHORT).show()
        return@setOnClickListener
      }
      val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, text)
      startActivity(Intent.createChooser(send, getString(R.string.share)))
    }
    panel.saveButton.setOnClickListener { saveText.launch("ocr.txt") }
  }

  private fun setupSettings() {
    val panel = binding.panel
    tiers = OcrEngine.bundledTiers(this)
    if (tiers.isEmpty()) {
      showError("No OCR models are bundled in this APK. Rebuild it with the models folder available.")
    }

    bindSpinner(panel.languageSpinner, Language.entries.map { it.label }, KEY_LANGUAGE,
      Language.entries.indexOf(Language.KOREAN))
    // small: about 3x faster than medium on phone CPUs, and usually the same text.
    bindSpinner(panel.tierSpinner, tiers, KEY_TIER, tiers.indexOf("small").coerceAtLeast(0))
    bindSpinner(panel.detSideSpinner,
      DET_SIDES.map { if (it == Images.MAX_SIDE) "$it px (same as desktop)" else "$it px" },
      KEY_DET_SIDE, DET_SIDES.indexOf(DEFAULT_DET_SIDE))

    panel.minScoreSlider.value = prefs.getFloat(KEY_MIN_SCORE, 0.5f)
    updateMinScoreLabel()
    panel.minScoreSlider.addOnChangeListener { _, value, _ ->
      prefs.edit().putFloat(KEY_MIN_SCORE, value).apply()
      updateMinScoreLabel()
    }

    panel.flattenSwitch.isChecked = prefs.getBoolean(KEY_FLATTEN, true)
    panel.autoSwitch.isChecked = prefs.getBoolean(KEY_AUTO, false)
    flattenEnabled = panel.flattenSwitch.isChecked
    autoCaptureEnabled = panel.autoSwitch.isChecked
    panel.flattenSwitch.setOnCheckedChangeListener { _, checked ->
      flattenEnabled = checked
      prefs.edit().putBoolean(KEY_FLATTEN, checked).apply()
      if (!checked) binding.stage.overlay.show(null, 1, 1, false)
    }
    panel.autoSwitch.setOnCheckedChangeListener { _, checked ->
      autoCaptureEnabled = checked
      prefs.edit().putBoolean(KEY_AUTO, checked).apply()
    }
  }

  private fun bindSpinner(spinner: Spinner, items: List<String>, key: String, default: Int) {
    spinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, items)
    val saved = prefs.getInt(key, default)
    spinner.setSelection(if (saved in items.indices) saved else default.coerceIn(0, max(0, items.size - 1)))
    spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
      override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
        prefs.edit().putInt(key, position).apply()
      }

      override fun onNothingSelected(parent: AdapterView<*>?) = Unit
    }
  }

  private fun updateMinScoreLabel() {
    binding.panel.minScoreLabel.text = String.format(
      Locale.US, "%s: %.2f", getString(R.string.min_confidence), binding.panel.minScoreSlider.value)
  }

  private fun currentSettings(): OcrSettings {
    val panel = binding.panel
    return OcrSettings(
      language = Language.entries[panel.languageSpinner.selectedItemPosition],
      tier = tiers.getOrElse(panel.tierSpinner.selectedItemPosition) { tiers.firstOrNull() ?: "tiny" },
      detMaxSide = DET_SIDES[panel.detSideSpinner.selectedItemPosition],
      minScore = panel.minScoreSlider.value,
    )
  }

  private fun currentText() = binding.panel.resultText.text?.toString().orEmpty()

  private fun render(state: UiState) {
    val stage = binding.stage
    val showImage = state.image != null
    stage.resultImage.isVisible = showImage
    stage.resultImage.setImageBitmap(state.image)
    stage.preview.isVisible = !showImage
    stage.overlay.isVisible = !showImage
    stage.hint.isVisible = !showImage && !cameraLive
    stage.progress.isVisible = state.busy
    if (state.status.isNotEmpty()) binding.actions.status.text = state.status
    if (state.text != shownText) {
      shownText = state.text
      binding.panel.resultText.setText(state.text)
    }
    binding.actions.scanButton.isEnabled = cameraLive && !state.busy && !showImage
    binding.panel.rerunButton.isEnabled = viewModel.hasImage && !state.busy
    state.error?.let {
      viewModel.clearError()
      showError(it)
    }
  }

  private fun showError(message: String) {
    MaterialAlertDialogBuilder(this)
      .setTitle(R.string.ocr_error)
      .setMessage(message)
      .setPositiveButton(android.R.string.ok, null)
      .show()
  }

  // -------------------------------------------------------------- input

  private fun handleShareIntent(intent: Intent?) {
    if (intent?.action != Intent.ACTION_SEND || intent.type?.startsWith("image/") != true) return
    IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java)?.let(::openImage)
  }

  private fun openImage(uri: Uri) {
    lifecycleScope.launch {
      val bitmap = try {
        withContext(Dispatchers.IO) { Images.decode(this@MainActivity, uri) }
      } catch (e: Exception) {
        showError("Cannot read image:\n${e.message}")
        return@launch
      }
      stopCamera()
      // Like the desktop app, opened images are used as-is (no page flattening).
      viewModel.process(bitmap, flatten = false, settings = currentSettings())
    }
  }

  private fun writeText(uri: Uri) {
    try {
      contentResolver.openOutputStream(uri)?.use { it.write(currentText().toByteArray(Charsets.UTF_8)) }
      Toast.makeText(this, R.string.saved, Toast.LENGTH_SHORT).show()
    } catch (e: Exception) {
      showError("Cannot save:\n${e.message}")
    }
  }

  // ------------------------------------------------------------- camera

  private fun startCamera() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
      requestCamera.launch(Manifest.permission.CAMERA)
      return
    }
    val future = ProcessCameraProvider.getInstance(this)
    future.addListener({
      val provider = try {
        future.get()
      } catch (e: Exception) {
        Log.e(TAG, "Camera unavailable", e)
        binding.stage.hint.setText(R.string.hint_no_camera)
        return@addListener
      }
      cameraProvider = provider
      if (viewModel.state.value.image != null) return@addListener // an image arrived meanwhile
      val selector = listOf(CameraSelector.DEFAULT_BACK_CAMERA, CameraSelector.DEFAULT_FRONT_CAMERA)
        .firstOrNull { runCatching { provider.hasCamera(it) }.getOrDefault(false) }
      if (selector == null) {
        binding.stage.hint.setText(R.string.hint_no_camera)
        render(viewModel.state.value)
        return@addListener
      }
      // Only Preview + ImageAnalysis: a combination every device supports. Like the desktop
      // app, Scan takes the next analysis frame instead of using ImageCapture, which some
      // phones fail with "Camera is closed" when three streams run at once.
      val preview = Preview.Builder()
        .setResolutionSelector(resolution(Size(1280, 960)))
        .build()
        .also { it.surfaceProvider = binding.stage.preview.surfaceProvider }
      val analysis = ImageAnalysis.Builder()
        .setResolutionSelector(resolution(Size(1920, 1440)))
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .build()
        .also { it.setAnalyzer(cameraExecutor, ::analyze) }
      try {
        provider.unbindAll()
        val camera = provider.bindToLifecycle(this, selector, preview, analysis)
        camera.cameraInfo.cameraState.observe(this) { state ->
          state.error?.let { error ->
            Log.e(TAG, "Camera error ${error.code}", error.cause)
            binding.actions.status.text = cameraErrorMessage(error.code)
          }
        }
        cameraLive = true
        captureRequested = false
        anchorQuad = null
        steadySince = 0L
        binding.actions.status.setText(R.string.status_ready)
      } catch (e: Exception) {
        Log.e(TAG, "Camera bind failed", e)
        binding.stage.hint.setText(R.string.hint_no_camera)
        binding.actions.status.text = "Cannot open the camera: ${e.message}"
      }
      render(viewModel.state.value)
    }, ContextCompat.getMainExecutor(this))
  }

  private fun cameraErrorMessage(code: Int) = when (code) {
    CameraState.ERROR_CAMERA_IN_USE, CameraState.ERROR_MAX_CAMERAS_IN_USE ->
      "Camera is in use by another app. Close it, then tap Camera."
    CameraState.ERROR_CAMERA_DISABLED -> "Camera is disabled on this device (policy)."
    CameraState.ERROR_DO_NOT_DISTURB_MODE_ENABLED -> "Camera is blocked by Do Not Disturb."
    else -> "Camera error ($code). Tap Camera to restart it."
  }

  private fun resolution(target: Size) = ResolutionSelector.Builder()
    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
    .setResolutionStrategy(ResolutionStrategy(target, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
    .build()

  private fun stopCamera() {
    cameraProvider?.unbindAll()
    cameraLive = false
    captureRequested = false
    binding.stage.overlay.show(null, 1, 1, false)
  }

  /** Scan: the analyzer hands over the next full-resolution frame. */
  private fun capture() {
    if (!cameraLive || captureRequested || viewModel.busy) return
    captureRequested = true
    binding.actions.scanButton.isEnabled = false
    binding.actions.status.text = "Capturing..."
    binding.root.postDelayed({
      if (captureRequested) { // no frame arrived
        captureRequested = false
        binding.actions.status.text = "The camera is not sending frames. Tap Camera to restart it."
        render(viewModel.state.value)
      }
    }, CAPTURE_TIMEOUT_MS)
  }

  /** Grabs frames for Scan, and runs live page detection for the outline and auto-capture. */
  private fun analyze(proxy: ImageProxy) {
    proxy.use {
      if (captureRequested) {
        val frame = rotate(it.toBitmap(), it.imageInfo.rotationDegrees)
        runOnUiThread {
          if (!captureRequested) return@runOnUiThread
          captureRequested = false
          stopCamera() // freeze on the captured shot
          viewModel.process(frame, flattenEnabled, currentSettings())
        }
        return
      }
      if (!flattenEnabled) return
      // Page detection works at 500 px anyway, so shrink the frame first.
      val full = it.toBitmap()
      val scale = DETECT_SIDE.toFloat() / max(full.width, full.height)
      val small = if (scale < 1) {
        Bitmap.createScaledBitmap(full, (full.width * scale).toInt(), (full.height * scale).toInt(), true)
      } else {
        full
      }
      val bgr = Mat()
      val rgba = Mat()
      org.opencv.android.Utils.bitmapToMat(small, rgba)
      Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
      rgba.release()
      when (it.imageInfo.rotationDegrees) {
        90 -> Core.rotate(bgr, bgr, Core.ROTATE_90_CLOCKWISE)
        180 -> Core.rotate(bgr, bgr, Core.ROTATE_180)
        270 -> Core.rotate(bgr, bgr, Core.ROTATE_90_COUNTERCLOCKWISE)
      }
      val quad = Scanner.findDocument(bgr)
      val frameW = bgr.cols()
      val frameH = bgr.rows()
      bgr.release()

      // The page counts as steady while no corner moves more than 2% of the frame
      // from where the current streak started.
      val now = SystemClock.elapsedRealtime()
      val anchor = anchorQuad
      if (quad == null) {
        anchorQuad = null
        steadySince = 0L
      } else if (anchor == null || drift(quad, anchor) > STEADY_TOLERANCE * max(frameW, frameH)) {
        anchorQuad = quad
        steadySince = now
      }
      val steadyMs = if (quad != null) now - steadySince else 0L
      val fire = quad != null && autoCaptureEnabled && steadyMs >= AUTO_CAPTURE_MS
      runOnUiThread {
        if (!cameraLive) return@runOnUiThread
        binding.stage.overlay.show(quad, frameW, frameH, steadyMs >= AUTO_CAPTURE_MS / 2)
        if (fire && !viewModel.busy && !captureRequested) {
          anchorQuad = null
          capture()
        }
      }
    }
  }

  /** Largest corner movement, comparing x and y coordinates sorted independently (as on desktop). */
  private fun drift(a: List<Point>, b: List<Point>): Double {
    val ax = a.map { it.x }.sorted()
    val bx = b.map { it.x }.sorted()
    val ay = a.map { it.y }.sorted()
    val by = b.map { it.y }.sorted()
    return ax.indices.maxOf { i -> max(abs(ax[i] - bx[i]), abs(ay[i] - by[i])) }
  }

  private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
    if (degrees == 0) return bitmap
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
  }

  companion object {
    private const val TAG = "OcrScanner"
    private const val AUTO_CAPTURE_MS = 1200L
    private const val CAPTURE_TIMEOUT_MS = 3000L
    private const val DETECT_SIDE = 640
    private const val STEADY_TOLERANCE = 0.02
    private const val KEY_LANGUAGE = "language"
    private const val KEY_TIER = "tier"
    private const val KEY_DET_SIDE = "det_side"
    private const val KEY_MIN_SCORE = "min_score"
    private const val KEY_FLATTEN = "flatten"
    private const val KEY_AUTO = "auto_capture"
  }
}
