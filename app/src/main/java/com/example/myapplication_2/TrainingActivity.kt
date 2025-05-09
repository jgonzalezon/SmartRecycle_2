
package com.example.myapplication_2

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import androidx.core.graphics.scale
import org.tensorflow.lite.Delegate

class TrainingActivity : AppCompatActivity() {
    companion object {
        internal const val MODEL_FILE     = "continual_trainable.tflite"
        private const val IMG_SIZE       = 260
        private const val TOTAL_CLASSES  = 15
    }
    private lateinit var progressBar: ProgressBar
    private lateinit var btnPick: Button
    private lateinit var btnTrainNow: Button
    private lateinit var recycler: RecyclerView

    private lateinit var tflite: Interpreter
    private lateinit var adapter: ImageAdapter


    // Contract para seleccionar múltiples imágenes
    private val pickImages =
        registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isNotEmpty()) addUris(uris)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_training)

        progressBar  = findViewById(R.id.progressBar)
        btnPick      = findViewById(R.id.btnPick)
        btnTrainNow  = findViewById(R.id.btnTrainNow)
        recycler     = findViewById(R.id.recycler)

        /* 1 · instantiate the delegate */
        val modelBuf = loadAsset(MODEL_FILE)

        /* 2 · prepare options */
        val options = Interpreter.Options().apply {
            // attach the delegate ➜ enables all Select-TF-Ops
            setNumThreads(4)
        }
        tflite = Interpreter(modelBuf, options)

        // 2) Recycler + Adapter
        adapter = ImageAdapter(
            labels = AppConfig.labels,
            onLabelChange = ::onLabelChanged
        )
        recycler.layoutManager = LinearLayoutManager(this)
        recycler.adapter = adapter

        // 3) Botón “Elegir fotos”
        btnPick.setOnClickListener {
            pickImages.launch(arrayOf("image/*"))
        }

        // 4) Botón “Entrenar ahora”
        btnTrainNow.setOnClickListener {
            trainAll()
        }
    }

    /** Añade las URIs seleccionadas al adaptador, infiere y muestra miniaturas */
    private fun addUris(uris: List<Uri>) {
        for (uri in uris) {
            val bmp = uriToBitmap(uri) ?: continue
            val probs = classifyBitmap(bmp)
            val best  = probs.indices.maxBy { probs[it] }
            adapter.add(ImageItem(uri, bmp, best))
        }
    }

    /** Callback que recibe la posición y la nueva etiqueta del spinner */
    private fun onLabelChanged(position: Int, newIdx: Int) {
        adapter.items[position].labelIdx = newIdx
    }

    /**
     * Lanza el entrenamiento en background y actualiza la barra
     * de progreso por cada imagen procesada.
     */
    private fun trainAll() {
        if (adapter.items.isEmpty()) {
            Toast.makeText(this, "No hay imágenes para entrenar", Toast.LENGTH_SHORT).show()
            return
        }

        // Deshabilita el botón mientras entrena
        btnTrainNow.isEnabled = false

        // Prepara la ProgressBar en UI
        val total = adapter.items.size
        progressBar.max = total
        progressBar.progress = 0
        progressBar.visibility = View.VISIBLE

        // Entrenamiento en hilo aparte
        Thread {
            adapter.items.forEachIndexed { idx, item ->
                // 1) Prepara entradas
                val xBuf = bitmapToBuffer(item.bitmap)
                val yBuf = oneHot(item.labelIdx, TOTAL_CLASSES)

                // 2) Preparar el mapa de outputs
                val lossArr = FloatArray(1)
                val inputs  = mapOf("x" to xBuf, "y" to yBuf)
                val outputs = mutableMapOf<String, Any>("loss" to lossArr)

                // 3) Llamada a la firma "train"
                tflite.runSignature(inputs, outputs, "train")

                val loss = (outputs["loss"] as FloatArray)[0]
                Log.d("TrainingActivity", "Img #$idx etiqueta=${item.labelIdx} loss=$loss")

                // 4) Actualiza la barra en el hilo de UI
                runOnUiThread {
                    progressBar.progress = idx + 1
                }
            }


            runOnUiThread {
                progressBar.visibility = View.GONE
                setResult(Activity.RESULT_OK)
                Toast.makeText(
                    this@TrainingActivity,
                    "Entrenamiento completado",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }.start()
    }

    /** Preprocesa un Bitmap a ByteBuffer normalizado en [-1,1] */
    private fun bitmapToBuffer(bmp: Bitmap): ByteBuffer {
        // 1) Escalado con filtrado (bilineal) para evitar aliasing
        val img = Bitmap.createScaledBitmap(bmp, IMG_SIZE, IMG_SIZE, true)

        // 2) Reserva exacta de bytes: width*height*3 canales * Float.SIZE_BYTES (==4)
        val buf = ByteBuffer
            .allocateDirect(IMG_SIZE * IMG_SIZE * 3 * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())

        // 3) Extrae cada píxel y NORMALIZA a [-1, +1]
        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        img.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)
        for (p in pixels) {
            // usa ushr para desplazamiento sin signo
            val r = ((p ushr 16) and 0xFF).toFloat()
            val g = ((p ushr  8) and 0xFF).toFloat()
            val b = ( p              and 0xFF).toFloat()
            buf.putFloat(r / 127.5f - 1f)
            buf.putFloat(g / 127.5f - 1f)
            buf.putFloat(b / 127.5f - 1f)
        }

        // 4) Vuelve al principio para que el Interpreter lea desde byte 0
        buf.rewind()
        return buf
    }

    /** Construye un one-hot de longitud length con un 1.0f en idx */
    private fun oneHot(idx: Int, length: Int): Array<FloatArray> {
        val vec = FloatArray(length) { 0f }.apply { this[idx] = 1f }
        return arrayOf(vec)
    }

    /** Decodifica una URI a Bitmap seguro desde ContentResolver */
    private fun uriToBitmap(uri: Uri): Bitmap? =
        try {
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }

    /** Carga el modelo TFLite de assets a ByteBuffer */
    private fun loadAsset(name: String): ByteBuffer =
        assets.openFd(name).use { fd ->
            ByteBuffer.allocateDirect(fd.declaredLength.toInt())
                .order(ByteOrder.nativeOrder())
                .also { fd.createInputStream().channel.read(it); it.rewind() }
        }

    private fun classifyBitmap(bmp: Bitmap): FloatArray {
        // 1) Pre-procesado: Bitmap → ByteBuffer [-1,1]
        val buf = bitmapToBuffer(bmp)
        buf.rewind()

        // 2) Inferencia: pedimos siempre 15 floats de salida
        val outputs = mutableMapOf<String, Any>()
        outputs["output"] = Array(1) { FloatArray(15) }
        tflite.runSignature(mapOf("x" to buf), outputs, "infer")

        // 3) Extraer las primeras ACTIVE_CLASSES y devolver probabilidades
        @Suppress("UNCHECKED_CAST")
        val fullLogits = (outputs["output"] as Array<FloatArray>)[0]
        val logits9   = fullLogits.copyOfRange(0, AppConfig.activeClasses)

        return   logits9

    }
}