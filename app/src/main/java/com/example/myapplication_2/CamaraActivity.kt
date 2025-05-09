package com.example.myapplication_2

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.ContentValues.TAG
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.graphics.drawable.ShapeDrawable
import android.graphics.drawable.shapes.RectShape
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
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
import androidx.core.graphics.scale
import org.tensorflow.lite.support.common.ops.NormalizeOp
import java.io.File
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer      // si lo utilizas

/**
 * Cámara + clasificación TFLite que expone la
 * firma `infer` (salida "output") de 9 clases.
 */
class CamaraActivity : AppCompatActivity() {

    // UI
    private lateinit var previewView: PreviewView
    private lateinit var tvTop: TextView
    private lateinit var tvAll: TextView

    // TFLite
    private lateinit var tflite: Interpreter

    // zona de recorte
    private var focusRect: Rect? = null

    companion object {
        private const val MODEL_FILE = "continual_trainable.tflite"
        internal var activeClasses = 9      // 9 al inicio
        private const val MAX_CLASSES = 15
    }

    // ──────────────────────────────────────────────────────────────
    // ──────────────────────────────────────────────────────────────
    @SuppressLint("ClickableViewAccessibility", "SetTextI18n")override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        previewView = findViewById(R.id.previewView)
        tvTop       = findViewById(R.id.tvTop)
        tvAll       = findViewById(R.id.tvAll)


        // 1) Cargar e inicializar el intérprete
        tflite = Interpreter(loadModelFile(MODEL_FILE))
        tflite.allocateTensors()

        // 2) Ruta del checkpoint en internal storage
        val ckptFile = File(filesDir, "model.ckpt")

        if (ckptFile.exists()) {
            try {
                // 3) Prepara los mapas para la firma "restore"
                val inputs: Map<String, Any> = mapOf(
                    "path" to arrayOf(ckptFile.absolutePath)
                )
                val outputs: MutableMap<String, Any> = mutableMapOf()

                // 4) Ejecuta la firma
                tflite.runSignature(inputs, outputs, "restore")
                Toast.makeText(this, "Checkpoint restaurado", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e(TAG, "Error al restaurar checkpoint", e)
                Toast.makeText(this, "No se pudo restaurar checkpoint", Toast.LENGTH_SHORT).show()
            }
        } else {
            Log.i(TAG, "No hay checkpoint previo → pesos iniciales")
        }

        // 5) Pedir permiso de cámara o arrancar preview
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 123)
        } else {
            startCameraPreview()
        }

        previewView.setOnTouchListener { _, ev ->
            if (ev.action == MotionEvent.ACTION_UP) {
                previewView.bitmap?.let { bmp ->
                    guardarEnGaleria(bmp)
                    focusRect?.let { r ->
                        val w = r.width().coerceAtMost(bmp.width - r.left)
                        val h = r.height().coerceAtMost(bmp.height - r.top)
                        val crop = Bitmap.createBitmap(bmp, r.left, r.top, w, h)

                        val probs = classify(crop)
                        val (idx, prob) = probs.withIndex().maxBy { it.value }

                        tvTop.text = "${AppConfig.labels[idx]}: ${"%.1f".format(prob * 100)}%"
                        tvAll.text = buildString {
                            AppConfig.labels.forEachIndexed { i, lbl ->
                                append("$lbl: ${"%.1f".format(probs[i] * 100)}%\n")
                            }
                        }
                    }
                    Toast.makeText(this, "Foto guardada y clasificada", Toast.LENGTH_SHORT).show()
                }
            }
            true
        }
    }

    // ─────────────────────── Cámara ──────────────────────────────
    private fun startCameraPreview() {
        val pf = ProcessCameraProvider.getInstance(this)
        pf.addListener({
            val provider = pf.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview)
            drawFocusOverlay()
        }, ContextCompat.getMainExecutor(this))
    }


    private fun loadLabelsFromAssets(name: String): List<String> =
        assets.open(name).bufferedReader()
            .useLines { seq ->
                seq
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                    .toList()
            }

    private fun drawFocusOverlay() {
        previewView.post {
            val size = min(previewView.width, previewView.height) / 2
            val left = (previewView.width - size) / 2
            val top = (previewView.height - size) / 2
            focusRect = Rect(left, top, left + size, top + size)

            val shape = ShapeDrawable(RectShape()).apply {
                paint.color = Color.GREEN
                paint.style = android.graphics.Paint.Style.STROKE
                paint.strokeWidth = 8f
                setBounds(focusRect!!)
            }
            previewView.overlay.clear()
            previewView.overlay.add(shape)
        }
    }

    // ─────────────────────── IO helper ───────────────────────────
    private fun loadModelFile(name: String): ByteBuffer =
        assets.openFd(name).use { afd ->
            val buffer = ByteBuffer.allocateDirect(afd.declaredLength.toInt())
                .order(ByteOrder.nativeOrder())
            afd.createInputStream().channel.read(buffer)
            buffer.rewind()
            buffer
        }

    // ────────────────────── Inferencia ────────────────────────────
    private fun classify(bitmap: Bitmap): FloatArray {

        /* ---------- 1) Pre-procesado idéntico a Python ---------- */
        val tensor  = TensorImage.fromBitmap(bitmap)               // RGB-888
        val input   = ImageProcessor.Builder()
            .add(ResizeOp(AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            // (x / 127.5 − 1) ⇒ [-1,1] tal como EfficientNet espera
            .add(NormalizeOp(127.5f, 127.5f))
            .build()
            .process(tensor)                                       // (1,260,260,3) FP32

        /* ---------- 2) Inferencia con la Signature API ---------- */
        val outputs = hashMapOf<String, Any>(
            "output" to Array(1) { FloatArray(AppConfig.MaxClasses) }
        )
        tflite.runSignature(mapOf("x" to input.buffer), outputs, "infer")

        @Suppress("UNCHECKED_CAST")
        val logits = (outputs["output"] as Array<FloatArray>)[0]

        /* ---------- 3) Devolver sólo las clases activas ---------- */
        return logits.copyOfRange(0, AppConfig.activeClasses)
    }



    // ─────────────── Guardar en galería (sin cambios) ──────────────
    private fun guardarEnGaleria(bmp: Bitmap) {
        val filename = "MiFoto_${System.currentTimeMillis()}.jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/MyApp")
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q)
                put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        contentResolver.openOutputStream(uri)?.use { os ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, os)
        }
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
        }
    }
}