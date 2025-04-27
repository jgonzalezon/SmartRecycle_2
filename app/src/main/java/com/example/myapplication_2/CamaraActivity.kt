package com.example.myapplication_2

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.graphics.drawable.ShapeDrawable
import android.graphics.drawable.shapes.RectShape
import android.os.Bundle
import android.provider.MediaStore
import android.view.MotionEvent
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

class CamaraActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvResult: TextView
    private lateinit var tvResultMulti: TextView

    private lateinit var tfliteBinary: Interpreter
    private lateinit var tfliteMulti: Interpreter

    private var focusRect: Rect? = null

    companion object {
        private const val MODEL_BINARY = "waste_model_V3.tflite"
        private const val MODEL_MULTI  = "waste_model_V3_multi.tflite"
        private const val IMAGE_SIZE   = 224
        private val IMAGE_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGE_STD  = floatArrayOf(0.229f, 0.224f, 0.225f)
        private const val THRESHOLD = 0.5f
        private const val POSITIVE_LABEL = "Reciclable"
        private const val NEGATIVE_LABEL = "Orgánica"
        private val MULTI_LABELS = listOf(
            "Basura general", "Carton", "Aluminio", "Organica", "Papel",
            "Plastico", "Vidrio", "Vegetacion", "Textil"
        )
    }

    @SuppressLint("ClickableViewAccessibility", "SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        previewView = findViewById(R.id.previewView)
        tvResult = findViewById(R.id.tvResult)
        tvResultMulti = findViewById(R.id.tvResultMulti)

        tfliteBinary = Interpreter(loadModelFile(MODEL_BINARY))
        tfliteMulti  = Interpreter(loadModelFile(MODEL_MULTI))

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 123)
        } else {
            startCameraPreview()
        }

        // Toque: recuadro, guardar y clasificar
        previewView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                val fullBitmap = previewView.bitmap
                fullBitmap?.let { bmp ->
                    // 1) guardar foco completo
                    guardarEnGaleria(bmp)
                    Toast.makeText(this, "Foto guardada", Toast.LENGTH_SHORT).show()

                    // 2) recortar según focusRect y clasificar
                    focusRect?.let { r ->
                        val w = r.width().coerceAtMost(bmp.width - r.left)
                        val h = r.height().coerceAtMost(bmp.height - r.top)
                        val cropped = Bitmap.createBitmap(bmp, r.left, r.top, w, h)
                        val probPos = classifyBinary(cropped)
                        val isPos = probPos >= THRESHOLD
                        val label = if (isPos) POSITIVE_LABEL else NEGATIVE_LABEL
                        val pct = "%.1f".format(if (isPos) probPos*100 else (1f-probPos)*100)
                        tvResult.text = "$label: $pct%"

                        if (isPos) {
                            val probs = classifyMulti(cropped)
                            tvResultMulti.text = buildString {
                                MULTI_LABELS.forEachIndexed { i, lbl ->
                                    append("$lbl: ${"%.1f".format(probs[i]*100)}%\n")
                                }
                            }
                        } else {
                            tvResultMulti.text = ""
                        }
                    }
                }
            }
            true
        }
    }

    private fun startCameraPreview() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            val previewUseCase = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, previewUseCase)
            drawFocusOverlay()
        }, ContextCompat.getMainExecutor(this))
    }

    // Dibuja un recuadro centrado en pantalla
    private fun drawFocusOverlay() {
        previewView.post {
            val w = previewView.width
            val h = previewView.height
            val size = min(w, h) / 2
            val left = (w - size) / 2
            val top = (h - size) / 2
            val rect = Rect(left, top, left + size, top + size)
            focusRect = rect

            val shape = ShapeDrawable(RectShape()).apply {
                paint.color = Color.GREEN
                paint.style = android.graphics.Paint.Style.STROKE
                paint.strokeWidth = 8f
            }
            shape.setBounds(rect)
            previewView.overlay.clear()
            previewView.overlay.add(shape)
        }
    }

    private fun loadModelFile(name: String): ByteBuffer {
        assets.openFd(name).use { afd ->
            val input = afd.createInputStream()
            val size = afd.declaredLength.toInt()
            val buffer = ByteBuffer.allocateDirect(size)
            buffer.order(ByteOrder.nativeOrder())
            input.channel.read(buffer)
            buffer.rewind()
            return buffer
        }
    }

    private fun classifyBinary(bitmap: Bitmap): Float {
        val img = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val buf = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
            .apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        img.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            buf.putFloat((r - IMAGE_MEAN[0]) / IMAGE_STD[0])
            buf.putFloat((g - IMAGE_MEAN[1]) / IMAGE_STD[1])
            buf.putFloat((b - IMAGE_MEAN[2]) / IMAGE_STD[2])
        }
        buf.rewind()
        val output = Array(1) { FloatArray(1) }
        tfliteBinary.run(buf, output)
        return output[0][0]
    }

    private fun classifyMulti(bitmap: Bitmap): FloatArray {
        val img = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val buf = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
            .apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        img.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            buf.putFloat((r - IMAGE_MEAN[0]) / IMAGE_STD[0])
            buf.putFloat((g - IMAGE_MEAN[1]) / IMAGE_STD[1])
            buf.putFloat((b - IMAGE_MEAN[2]) / IMAGE_STD[2])
        }
        buf.rewind()
        val output = Array(1) { FloatArray(MULTI_LABELS.size) }
        tfliteMulti.run(buf, output)
        return output[0]
    }

    private fun guardarEnGaleria(bitmap: Bitmap) {
        val filename = "MiFoto_${System.currentTimeMillis()}.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/MyApp")
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        contentResolver.openOutputStream(uri)?.use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
        }
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
        }
    }
}
