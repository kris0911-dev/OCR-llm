import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
  alias(libs.plugins.android.application)
}

android {
  namespace = "com.ocrllm.scanner"
  compileSdk { this.version = release(37) { minorApiLevel = 0 } }

  defaultConfig {
    applicationId = "com.ocrllm.scanner"
    minSdk = 28
    targetSdk = 37
    versionCode = 1
    versionName = "1.0"
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    ndk {
      // Phones (arm64) + emulators (x86_64). Add "armeabi-v7a" for old 32-bit devices.
      abiFilters += listOf("arm64-v8a", "x86_64")
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      // Signed with the local debug key so the prebuilt APK installs directly.
      // Replace with your own signingConfig before publishing to a store.
      signingConfig = signingConfigs.getByName("debug")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  buildFeatures {
    viewBinding = true
  }
  androidResources {
    // Models are memory-mapped straight from the APK, which requires them uncompressed.
    noCompress += "onnx"
  }
  packaging {
    // Uncompressed native libs map straight from the APK (faster first launch).
    jniLibs.useLegacyPackaging = false
  }
}

kotlin {
  compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

/** Copies the ONNX weights + inference.yml (char dictionary) of every model the app uses into assets. */
abstract class BundleOcrModels : DefaultTask() {
  @get:Internal abstract val modelsDir: DirectoryProperty
  @get:Input abstract val modelNames: ListProperty<String>
  @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE)
  abstract val modelFiles: ConfigurableFileCollection
  @get:OutputDirectory abstract val outputDir: DirectoryProperty

  @TaskAction
  fun copy() {
    val out = outputDir.get().asFile
    out.deleteRecursively()
    for (name in modelNames.get()) {
      for (file in listOf("inference.onnx", "inference.yml")) {
        val src = modelsDir.get().file("$name/$file").asFile
        if (!src.isFile) {
          throw GradleException(
            "Missing $src\nRun 'python download_models.py --formats onnx' in Desktop-ver, " +
              "or point -PocrModelsDir at a folder with the models.")
        }
        src.copyTo(File(out, "models/$name/$file"), overwrite = true)
      }
    }
  }
}

val ocrTiers = providers.gradleProperty("ocrTiers").get().split(",").map { it.trim() }.filter { it.isNotEmpty() }
val ocrModelNames = ocrTiers.flatMap { listOf("PP-OCRv6_${it}_det", "PP-OCRv6_${it}_rec") } +
  "korean_PP-OCRv5_mobile_rec"
val ocrModelsDir = rootProject.layout.projectDirectory.dir(providers.gradleProperty("ocrModelsDir").get())

val bundleOcrModels = tasks.register<BundleOcrModels>("bundleOcrModels") {
  modelsDir.set(ocrModelsDir)
  modelNames.set(ocrModelNames)
  modelFiles.from(ocrModelNames.flatMap { name ->
    listOf(ocrModelsDir.file("$name/inference.onnx"), ocrModelsDir.file("$name/inference.yml"))
  })
}

androidComponents {
  onVariants { variant ->
    variant.sources.assets?.addGeneratedSourceDirectory(bundleOcrModels, BundleOcrModels::outputDir)
  }
}

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.appcompat)
  implementation(libs.androidx.activity.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.material)
  implementation(libs.kotlinx.coroutines.android)
  implementation(libs.camerax.core)
  implementation(libs.camerax.camera2)
  implementation(libs.camerax.lifecycle)
  implementation(libs.camerax.view)
  implementation(libs.onnxruntime.android)
  implementation(libs.opencv)

  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.test.runner)
}
