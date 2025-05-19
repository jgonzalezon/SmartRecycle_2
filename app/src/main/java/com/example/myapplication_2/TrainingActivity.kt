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
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream
import kotlin.math.min

class TrainingActivity : AppCompatActivity() {

    companion object {
        private const val MODEL_FILE       = "continual_trainable_B0.tflite"
        private const val REPLAY_ASSET     = "replay_buffer_.npz"
        private const val REPLAY_LOCAL     = "replay_buffer_local.npz"
        private const val IMG_SIZE         = 224
        private const val MAX_CLASSES      = 15
        private const val BATCH            = 1
    }

    /* UI */
    private lateinit var progressBar: ProgressBar
    private lateinit var btnPick: Button
    private lateinit var btnTrain: Button
    private lateinit var recycler: RecyclerView

    /* ML */
    private lateinit var tflite: Interpreter
    private lateinit var adapter: ImageAdapter

    /* Replay in-memory:  class → mutableListOf<Pair<imgVec, oneHot>> */
    private val replay = mutableMapOf<Int, MutableList<Pair<FloatArray, FloatArray>>>()

    /* -------------------------------- lifecycle -------------------------------- */
    private lateinit var imgProcessor: ImageProcessor

    private val pickImages =
        registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isNotEmpty()) addUris(uris)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_training)

        progressBar = findViewById(R.id.progressBar)
        btnPick     = findViewById(R.id.btnPick)
        btnTrain    = findViewById(R.id.btnTrainNow)
        recycler    = findViewById(R.id.recycler)

        /* 1) Intérprete */
        tflite = Interpreter(loadAsset(MODEL_FILE), Interpreter.Options().apply { setNumThreads(4) })
        val inputShape = tflite.getInputTensor(0).shape()
        val h = inputShape[1]
        val w = inputShape[2]
        imgProcessor = ImageProcessor.Builder()
            .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
            .add(CastOp(DataType.FLOAT32))
            .add(NormalizeOp(127.5f, 127.5f))
            .build()

        /* 2) Replay (assets → local) */
        loadReplay()

        /* 3) Recycler */
        adapter = ImageAdapter(
            labels = AppConfig.labels,
            onLabelChange = ::onLabelChanged
        )
        recycler.layoutManager = LinearLayoutManager(this)
        recycler.adapter       = adapter

        /* 4) Botones */
        btnPick.setOnClickListener { pickImages.launch(arrayOf("image/*")) }
        btnTrain.setOnClickListener { trainAll() }
    }


    private fun runSignatureReset(
        inputs: Map<String, Any>,
        outputs: Map<String, Any>,
        signatureKey: String
    ) {
        tflite.runSignature(inputs, outputs, signatureKey)
        // esto vuelve a preparar (y vacía) el subgrafo para la siguiente firma
        tflite.allocateTensors()
    }

    /* ---------------------------- recoger imágenes ---------------------------- */

    private fun addUris(uris: List<Uri>) {
        for (uri in uris) {
            val bmp = uriToBitmap(uri) ?: continue
            val probs = classifyBitmap(bmp)
            val best  = probs.withIndex().maxByOrNull { it.value }?.index ?: 0
            adapter.add(ImageItem(uri, bmp, best))
        }
    }



    private fun onLabelChanged(pos: Int, newIdx: Int) {
        adapter.items[pos].labelIdx = newIdx
    }

    /* -------------------------------- entrenamiento -------------------------- */

    private fun trainAll() {
        // 1) Si no hay nada, lo avisamos
        if (adapter.items.isEmpty()) {
            toast("No hay imágenes para entrenar")
            return
        }

        // 2) Preparación de la UI ANTES de lanzar el thread
        progressBar.max = adapter.items.size             // ← establecemos el total
        progressBar.progress = 0                         // ← empezamos en 0
        progressBar.visibility = View.VISIBLE            // ← la hacemos visible
        btnTrain.isEnabled = false                       // ← bloqueamos el botón



        Thread {
            var processed = 0
            val items = adapter.items

            while (processed < items.size) {
                // ---- 1) Preparamos el batch de BITMAPS frescos ----
                val slice = items.subList(processed, min(items.size, processed + BATCH))
                val freshBmps = slice.map { it.bitmap to it.labelIdx }

                // ---- 2) Añadimos replay si hace falta ----
                val need = BATCH - freshBmps.size
                val rep = if (need > 0) sampleReplay(need) else emptyList()

                // ---- 3) Creamos los buffers planos ----
                val batchSize = freshBmps.size + rep.size
                val imgLen = IMG_SIZE * IMG_SIZE * 3
                // Buffer para todas las imágenes
                val xs = FloatArray(batchSize * imgLen)
                val ys = Array(batchSize) { FloatArray(MAX_CLASSES) }

                // Procesamos cada elemento (Bitmap o FloatArray)
                var off = 0
                (freshBmps + rep).forEachIndexed { i, (bmpOrVec, lblOrIdx) ->
                    val imgVec: FloatArray
                    val lblVec: FloatArray

                    if (bmpOrVec is Bitmap) {
                        // ——— aquí usamos ImageProcessor ———
                        val ti = TensorImage.fromBitmap(bmpOrVec)
                        val buf = imgProcessor.process(ti).buffer
                        val fb = buf.asFloatBuffer()
                        imgVec = FloatArray(imgLen).also { fb.get(it) }

                        lblVec = oneHot(lblOrIdx as Int)
                        // guardamos en replay
                        val cls = lblVec.indexOfFirst { it == 1f }
                        replay.getOrPut(cls) { mutableListOf() }.add(imgVec to lblVec)
                    } else {
                        imgVec = bmpOrVec as FloatArray
                        lblVec = lblOrIdx as FloatArray
                    }

                    // copia al array plano
                    System.arraycopy(imgVec, 0, xs, off, imgLen)
                    off += imgLen
                    ys[i] = lblVec
                }

                // ---- 4) Llamada a la firma train ----
                val xBuf = ByteBuffer
                    .allocateDirect(xs.size * Float.SIZE_BYTES)
                    .order(ByteOrder.nativeOrder())
                    .apply { asFloatBuffer().put(xs); rewind() }

                val lossArr = FloatArray(1)
                runSignatureReset(
                    inputs  = mapOf("x" to xBuf, "y" to ys),
                    outputs = mutableMapOf("loss" to lossArr),
                    signatureKey = "train"
                )
                Log.d("Training", "batch loss=${lossArr[0]}")

                processed += freshBmps.size
                runOnUiThread { progressBar.progress = processed }
            }
            saveReplay()

            runOnUiThread {
                progressBar.visibility = View.GONE       // ← ocultamos
                btnTrain.isEnabled = true               // ← reactivamos el botón
                toast("Entrenamiento completado")
                setResult(Activity.RESULT_OK)
                finish()
            }
        }.start()
    }

    /* ----------------------------- replay helpers ---------------------------- */

    private fun loadReplay() {
        val localFile = File(filesDir, REPLAY_LOCAL)

        // 0) Si existe y no es ZIP válido, bórralo
        if (localFile.exists()) {
            try {
                ZipFile(localFile).use { /* si llega aquí, está OK */ }
            } catch (e: Exception) {
                // no es un ZIP de binarios planos → lo borramos
                localFile.delete()
            }
        }

        // 1) Primera vez o tras borrado → copia el asset
        if (!localFile.exists()) {
            assets.open(REPLAY_ASSET).use { stream ->
                FileOutputStream(localFile).use { out ->
                    stream.copyTo(out)
                }
            }
        }

        // 2) Ahora sí abrimos como ZIP de .bin plano
        ZipFile(localFile).use { zip ->
            replay.clear()
            zip.entries().asSequence().forEach { entry ->
                // cada entry es "0.bin", "1.bin", ...
                val cls = entry.name.removeSuffix(".bin").toIntOrNull()
                    ?: return@forEach
                val data = zip.getInputStream(entry).readBytes()
                replay[cls] = readClassBin(data).toMutableList()
            }
        }
    }

    private fun saveReplay() {
        val tmp = File(filesDir, "$REPLAY_LOCAL.tmp")
        ZipOutputStream(FileOutputStream(tmp)).use { zos ->
            replay.forEach { (cls, list) ->
                zos.putNextEntry(ZipEntry("$cls.bin"))
                zos.write(writeClassBin(list))
                zos.closeEntry()
            }
        }
        tmp.renameTo(File(filesDir, REPLAY_LOCAL))
    }

    private fun sampleReplay(k: Int): List<Pair<FloatArray, FloatArray>> {
        val out = mutableListOf<Pair<FloatArray, FloatArray>>()
        val keys = replay.keys.toList().shuffled()
        var idx = 0
        while (out.size < k && idx < keys.size) {
            val lst = replay[keys[idx++]] ?: continue
            if (lst.isNotEmpty()) out += lst.random()
        }
        return out
    }

    /* bin-format helpers */
    private fun writeClassBin(list: List<Pair<FloatArray, FloatArray>>): ByteArray {
        val imgLen   = IMG_SIZE * IMG_SIZE * 3
        val elemSize = imgLen + MAX_CLASSES          // nº de floats por ejemplo
        val totalFloats = list.size * elemSize
        val bytes = ByteBuffer.allocate(4 + totalFloats * 4)
            .order(ByteOrder.LITTLE_ENDIAN)
        bytes.putInt(list.size)
        list.forEach { (img, lbl) ->
            img.forEach { bytes.putFloat(it) }
            lbl.forEach { bytes.putFloat(it) }
        }
        return bytes.array()
    }

    private fun readClassBin(bytes: ByteArray): List<Pair<FloatArray, FloatArray>> {

        // --- header LE ---
        val nExamples = ByteBuffer.wrap(bytes, 0, 4)
            .order(ByteOrder.LITTLE_ENDIAN)
            .int
        require(nExamples in 1..1000) { "nExamples fuera de rango: $nExamples" }

        val imgLen   = IMG_SIZE * IMG_SIZE * 3
        val floatBuf = ByteBuffer.wrap(bytes, 4, bytes.size - 4)
            .order(ByteOrder.LITTLE_ENDIAN)
            .asFloatBuffer()

        val labelLen = floatBuf.capacity() / nExamples - imgLen
        require(labelLen in 1..MAX_CLASSES) {
            "Formato de replay inconsistente (labelLen=$labelLen)"
        }

        val list = mutableListOf<Pair<FloatArray, FloatArray>>()
        repeat(nExamples) {
            val img = FloatArray(imgLen).also { floatBuf.get(it) }
            val lblTmp = FloatArray(labelLen).also { floatBuf.get(it) }

            // pad a 15 si hace falta
            val lbl = if (labelLen == MAX_CLASSES) lblTmp
            else FloatArray(MAX_CLASSES).also {
                System.arraycopy(lblTmp, 0, it, 0, labelLen)
            }
            list += img to lbl
        }
        return list
    }

    /* ------------------------- utilidades de imagen -------------------------- */

    private fun bitmapToArray(bmp: Bitmap): FloatArray {
        val img = Bitmap.createScaledBitmap(bmp, IMG_SIZE, IMG_SIZE, true)
        val pixels = IntArray(IMG_SIZE * IMG_SIZE).also {
            img.getPixels(it, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)
        }
        val out = FloatArray(pixels.size * 3)
        var p = 0
        pixels.forEach {
            out[p++] = ((it ushr 16) and 0xFF) / 127.5f - 1f
            out[p++] = ((it ushr 8)  and 0xFF) / 127.5f - 1f
            out[p++] = ( it          and 0xFF) / 127.5f - 1f
        }
        return out
    }

    private fun uriToBitmap(uri: Uri): Bitmap? =
        try {
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
        } catch (e: Exception) { null }

    private fun loadAsset(name: String): ByteBuffer =
        assets.openFd(name).use { fd ->
            ByteBuffer.allocateDirect(fd.declaredLength.toInt())
                .order(ByteOrder.nativeOrder())
                .also { fd.createInputStream().channel.read(it); it.rewind() }
        }

    /* --------------------------- infer simple (top-1) ------------------------ */
    private fun classifyBitmap(bitmap: Bitmap): FloatArray {
        val shape       = tflite.getInputTensor(0).shape()   // [1,H,W,3]
        val h = shape[1]; val w = shape[2]

        val tensorImg   = TensorImage.fromBitmap(bitmap)
        val inputBuffer = ImageProcessor.Builder()
            .add(ResizeOp(h, w, ResizeOp.ResizeMethod.BILINEAR))
            .add(CastOp(org.tensorflow.lite.DataType.FLOAT32))
            .build()
            .process(tensorImg)
            .buffer

        val output = Array(1) { FloatArray(AppConfig.MaxClasses) }
        tflite.runSignature(
            mapOf("x" to inputBuffer),
            mapOf("output" to output),
            "infer"
        )
        // reset del intérprete para la siguiente llamada
        tflite.allocateTensors()

        return output[0].copyOfRange(0, AppConfig.activeClasses)
    }

    private fun oneHot(idx: Int) = FloatArray(MAX_CLASSES) { if (it == idx) 1f else 0f }
    private fun toast(msg: String) = runOnUiThread {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }
}
