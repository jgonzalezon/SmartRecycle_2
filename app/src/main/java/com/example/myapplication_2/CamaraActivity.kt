package com.example.myapplication_2

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.content.res.AssetFileDescriptor
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
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel




class CamaraActivity : AppCompatActivity() {

    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>
    companion object {
        private const val MODEL_FILENAME = "waste_model_V3.tflite"
        private const val LABELS_FILENAME = "labels.txt"
        private const val IMAGE_SIZE = 224
        private const val IMAGE_MEAN = 0.485f
        private const val IMAGE_STD = 0.229f
    }

    private lateinit var tvResult: TextView

    // 1) launcher para tomar foto
    private val takePhotoLauncher =
        registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap != null) {
                // 1) Guardar en galería
                guardarEnGaleria(bitmap)
                // 2) Clasificarla
                val probs = classify(bitmap)
                // 3) Mostrar resultados
                tvResult.text = buildString {
                    labels.forEachIndexed { i, lbl ->
                        append("$lbl: ${"%.1f".format(probs[i] * 100)}%\n")
                    }
                }
            } else {
                Toast.makeText(this, "No se obtuvo ninguna foto", Toast.LENGTH_SHORT).show()
            }
        }
    // Carga el .tflite desde assets en un ByteBuffer
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
    // Carga labels.txt (una etiqueta por línea)
    private fun loadLabels(name: String): List<String> {
        val out = mutableListOf<String>()
        assets.open(name).bufferedReader().useLines { lines ->
            lines.forEach { line ->
                if (line.isNotBlank()) out += line.trim()
            }
        }
        return out
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        tvResult = findViewById(R.id.tvResult)
        val btnTakePhoto = findViewById<Button>(R.id.btnTakePhoto)

        // 1) Carga modelo y labels
        tflite = Interpreter(loadModelFile(MODEL_FILENAME))
        labels = loadLabels(LABELS_FILENAME)

        btnTakePhoto.setOnClickListener {
            // Pedir permiso si no lo tenemos
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    123
                )
            } else {
                takePhotoLauncher.launch()
            }
        }
    }

    /** Rota 90° si el thumbnail sale en orientación incorrecta (opcional). */
    private fun rotateBitmapIfNeeded(bm: Bitmap): Bitmap {
        val matrix = Matrix().apply { postRotate(90f) }
        return Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, matrix, true)
    }

    /** Clasifica un Bitmap y devuelve la etiqueta más probable. */
    private fun classify(bitmap: Bitmap): FloatArray {
        // 1) redimensiona
        val img = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        // 2) prepara ByteBuffer de entrada
        val byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
            .apply { order(ByteOrder.nativeOrder()) }
        val intValues = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        img.getPixels(intValues, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        // 3) normaliza y rellena
        for (pixel in intValues) {
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f
            byteBuffer.putFloat((r - IMAGE_MEAN) / IMAGE_STD)
            byteBuffer.putFloat((g - IMAGE_MEAN) / IMAGE_STD)
            byteBuffer.putFloat((b - IMAGE_MEAN) / IMAGE_STD)
        }
        // 4) resultado (1×N)
        val output = Array(1) { FloatArray(labels.size) }
        // 5) inferencia
        tflite.run(byteBuffer, output)
        // 6) devolvemos el array de probabilidades
        return output[0]
    }


// Guarda el bitmap en la galería
    private fun guardarEnGaleria(bitmap: Bitmap) {
        val filename = "MiFoto_${System.currentTimeMillis()}.jpg"

        // 1) Preparamos los metadatos
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            // Para Android Q+ (Scoped Storage): carpeta DCIM/MyApp
            put(MediaStore.Images.Media.RELATIVE_PATH, "DCIM/MyApp")
            // marcamos pending hasta que terminemos de escribir
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        // 2) Insertamos y obtenemos URI
        val uri = contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            values
        )

        if (uri == null) {
            Toast.makeText(this, "No se pudo crear la entrada en MediaStore", Toast.LENGTH_SHORT).show()
            return
        }

        // 3) Escribimos el bitmap sobre el OutputStream
        try {
            contentResolver.openOutputStream(uri)?.use { out ->
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)) {
                    throw Exception("compress() devolvió false")
                }
            }
        } catch (e: Exception) {
            // si algo falla, borramos la entrada que creamos
            contentResolver.delete(uri, null, null)
            Toast.makeText(this, "Error al escribir la imagen: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }

        // 4) Liberamos el flag pending para que aparezca en la galería
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
        }

        Toast.makeText(this, "Foto guardada en galería", Toast.LENGTH_SHORT).show()
    }

}