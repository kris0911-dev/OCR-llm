# PP-OCRv6 Scanner: Android version

Android port of [`Desktop-ver`](../Desktop-ver). It uses the same PP-OCRv6 and Korean models and the same
pipeline, and it runs fully offline on the phone with
[ONNX Runtime](https://onnxruntime.ai) and OpenCV. Written in Kotlin, built with Android Studio.

## Features
- **Camera scan**: live preview with page-edge detection, perspective flattening, and optional
  auto-capture when the page holds still. Tap **Scan** to capture.
- **Image input**: **Gallery** (system photo picker), or **Share** an image to the app from any other app.
- **Korean support** (Korean recognizer + PP-OCRv6 detector), plus the PP-OCRv6 multilingual model for
  Chinese, English, Japanese and Latin-script languages.
- Model tier (medium / small / tiny) and detection size are selectable. Inference runs on the CPU
  (ONNX Runtime). The XNNPACK and NNAPI accelerators were tried: XNNPACK crashes on these models,
  and NNAPI can't run their dynamic input shapes.
- Text boxes are drawn on the image. Text comes out in reading order and can be edited, copied, shared, or saved as UTF-8 `.txt`.
- Settings are remembered. Portrait and landscape/tablet layouts.

## Install the prebuilt APK
`release/PP-OCRv6-Scanner.apk` (264 MB; Android 9+, arm64 phones and x86_64 emulators; stored with Git LFS).
Most of the size is the bundled models, so it works offline. Build with `-PocrTiers=small,tiny` for an APK about 120 MB smaller.
```powershell
adb install -r release\PP-OCRv6-Scanner.apk
```
Or copy the file to the phone and open it (allow "Install unknown apps" for your file manager).
The APK is signed with a local debug key, so it installs directly. Use your own key before publishing it to a store.

## Build with Android Studio
1. Make sure `../Desktop-ver/models` holds the models. They're in this repo via **Git LFS**
   (`git lfs install` before cloning). To fetch them again, run `python download_models.py --formats onnx` in `Desktop-ver`.
2. **File > Open** the `Android-ver` folder and let Gradle sync.
3. Run on a device, or **Build > Generate App Bundles or APKs > Build APK(s)**.

The `bundleOcrModels` Gradle task copies the ONNX weights and `inference.yml` (character
dictionary) of each model into the APK assets at build time, so the weights aren't stored twice in the repo.
Options in `gradle.properties` (or `-P` on the command line):

| property | default | meaning |
|---|---|---|
| `ocrTiers` | `medium,small,tiny` | detector tiers to bundle. `small,tiny` makes the APK about 120 MB smaller |
| `ocrModelsDir` | `../Desktop-ver/models` | where the models are read from |

Command line (uses Android Studio's JDK):
```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
.\gradlew assembleRelease            # app\build\outputs\apk\release\app-release.apk
.\gradlew connectedDebugAndroidTest  # on-device test vs desktop output (needs a device/emulator)
```

## How it matches the desktop app
`OcrEngine.kt` reimplements PaddleX's OCR pipeline step by step, so the same image gives the same
text as `Desktop-ver`:

| step | PaddleX (desktop) | Android |
|---|---|---|
| detection resize / normalize | `DetResizeForTest`, `NormalizeImage` | `TextDetector.resize/normalize` |
| box extraction | `DBPostProcess` (thresh 0.3, box 0.6, unclip 1.5) | `TextDetector.boxesFromBitmap` |
| box expansion | pyclipper `PyclipperOffset` (round joins) | `ClipperOffset` (port of Clipper 6.4.2) |
| box order | `SortQuadBoxes` | `TextDetector.sortQuadBoxes` |
| line crop | `CropByPolys` + `get_rotate_crop_image` | `cropTextLine` |
| recognition | batches of 6 sorted by aspect ratio, height 48 | `TextRecognizer` |
| decoding | `CTCLabelDecode` | `TextRecognizer.decode` |
| reading order | `engine._reading_order` | `OcrEngine.readingOrder` |

The instrumented test (`OcrEngineTest`) runs the app's engine on 12 Korean+English test pages
(clean and camera-like) and compares the result with the desktop app's output for the same pages
(`androidTest/assets/expected`). Results on an x86_64 Android 14 emulator:

| config | pages identical to desktop | mean time / page (emulator) |
|---|---|---|
| Korean, medium | 12 / 12 | 9.2 s |
| Korean, small | 12 / 12 | 2.8 s |
| Korean, tiny | 12 / 12 | 0.6 s |
| Multilingual, tiny | 12 / 12 | 0.4 s |
| camera pages: page detection + flatten + OCR (small) | 6 / 6, corners within 1 px | 0.6 s |

Emulator timings only show the relative speed of the tiers; a recent phone is usually faster. The app
defaults to **small**: it's about 3x faster than medium and gave the same text on these pages.

**Detection size**: the desktop app runs detection at full resolution (up to 4000 px). On a phone the
default is 1920 px for speed. Choose "4000 px (same as desktop)" for identical results on large images.

## Layout
```
app/src/main/java/com/ocrllm/scanner/
  OcrEngine.kt        engine API, languages/tiers, model loading, reading order
  TextDetector.kt     PP-OCRv6 detector + DB post-processing + Clipper offset port
  TextRecognizer.kt   line crop, batched recognition, CTC decoding
  Scanner.kt          page detection / perspective warp (port of scanner.py), image helpers
  OcrViewModel.kt     runs OCR off the UI thread, keeps state across rotation
  MainActivity.kt     CameraX preview/capture, live page outline, auto-capture, settings
  QuadOverlayView.kt  page outline overlay
app/src/androidTest/  on-device comparison with the desktop app
```
