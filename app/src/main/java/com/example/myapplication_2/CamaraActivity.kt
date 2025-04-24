package com.example.myapplication_2

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.launch
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class CamaraActivity : AppCompatActivity() {

    private lateinit var tfliteBinary: Interpreter
    private lateinit var tfliteMulti: Interpreter

    companion object {
        private const val MODEL_BINARY = "waste_model_V3.tflite"
        private const val MODEL_MULTI  = "waste_model_V3_multi.tflite"
        private const val IMAGE_SIZE   = 224

        // Normalización ImageNet
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

    private lateinit var tvResult: TextView
    private lateinit var tvResultMulti: TextView

    private val takePhotoLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) {
            guardarEnGaleria(bitmap)

            // 1) inferencia binaria
            val probPos = classifyBinary(bitmap)
            val isPos = probPos >= THRESHOLD
            val label = if (isPos) POSITIVE_LABEL else NEGATIVE_LABEL
            val pct = "%.1f".format(
                if (isPos) probPos * 100 else (1f - probPos) * 100
            )
            tvResult.text = "$label: $pct%"

            // 2) si reciclable, inferencia multiclase
            if (isPos) {
                val probs = classifyMulti(bitmap)
                // construir texto multilinea con etiquetas y prob
                tvResultMulti.text = buildString {
                    MULTI_LABELS.forEachIndexed { i, lbl ->
                        append("$lbl: ${"%.1f".format(probs[i] * 100)}%\n")
                    }
                }
            } else {
                tvResultMulti.text = ""
            }
        } else {
            Toast.makeText(this, "No se obtuvo ninguna foto", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        tvResult = findViewById(R.id.tvResult)
        tvResultMulti = findViewById(R.id.tvResultMulti)
        val btnTakePhoto = findViewById<Button>(R.id.btnTakePhoto)

        // Carga ambos modelos
        tfliteBinary = Interpreter(loadModelFile(MODEL_BINARY))
        tfliteMulti  = Interpreter(loadModelFile(MODEL_MULTI))

        btnTakePhoto.setOnClickListener {
            if (ContextCompat.checkSelfPermission(
                    this, Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this, arrayOf(Manifest.permission.CAMERA), 123
                )
            } else {
                takePhotoLauncher.launch()
            }
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

    // Inferencia binaria
    private fun classifyBinary(bitmap: Bitmap): Float {
        val img = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
            .apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        img.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr  8) and 0xFF) / 255f
            val b = ( p         and 0xFF) / 255f
            buffer.putFloat((r - IMAGE_MEAN[0]) / IMAGE_STD[0])
            buffer.putFloat((g - IMAGE_MEAN[1]) / IMAGE_STD[1])
            buffer.putFloat((b - IMAGE_MEAN[2]) / IMAGE_STD[2])
        }
        buffer.rewind()
        val output = Array(1) { FloatArray(1) }
        tfliteBinary.run(buffer, output)
        return output[0][0]
    }

    // Inferencia multiclase (9 clases)
    private fun classifyMulti(bitmap: Bitmap): FloatArray {
        val img = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
            .apply { order(ByteOrder.nativeOrder()) }
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        img.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr  8) and 0xFF) / 255f
            val b = ( p         and 0xFF) / 255f
            buffer.putFloat((r - IMAGE_MEAN[0]) / IMAGE_STD[0])
            buffer.putFloat((g - IMAGE_MEAN[1]) / IMAGE_STD[1])
            buffer.putFloat((b - IMAGE_MEAN[2]) / IMAGE_STD[2])
        }
        buffer.rewind()
        val output = Array(1) { FloatArray(MULTI_LABELS.size) }
        tfliteMulti.run(buffer, output)
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
        val uri = contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
        ) ?: run { Toast.makeText(this, "No se pudo crear la entrada en MediaStore", Toast.LENGTH_SHORT).show(); return }
        try {
            contentResolver.openOutputStream(uri)?.use { out ->
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)) throw Exception("compress() devolvió false")
            }
        } catch (e: Exception) {
            contentResolver.delete(uri, null, null)
            Toast.makeText(this, "Error al escribir la imagen: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
        }
        Toast.makeText(this, "Foto guardada en galería", Toast.LENGTH_SHORT).show()
    }
}
