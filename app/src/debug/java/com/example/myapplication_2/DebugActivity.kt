package com.example.myapplication_2

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import kotlin.math.exp
import kotlin.math.min

/** Actividad de depuración: prueba un .tflite sobre una imagen fija. */
class DebugActivity : AppCompatActivity() {

    companion object {
        private const val TAG        = "TFL_DEBUG"

        /** ficheros en app/src/main/assets/ */
        private const val MODEL_FILE = "continual_trainable.tflite"
        private const val TEST_IMG   = "test.jpg"

        private const val IMG_SIZE   = 260   // EfficientNet-B2
        private const val MAX_OUT    = 15    // columnas totales en el modelo
        private const val ACTIVE_OUT = 9     // clases entrenadas
    }

    private lateinit var tflite: Interpreter

    // ────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        tflite = Interpreter(loadAsset(MODEL_FILE))
        runDebugInference()

        finish()            // cerramos actividad de debug
    }

    // ────────────────────────────────────────────────────────────
    /** Lanza la firma infer y vuelca resultados a Logcat. */
    private fun runDebugInference() {
        /* 1) leer + escalar imagen */
        val bmp = loadBitmap(TEST_IMG).let {
            Bitmap.createScaledBitmap(it, IMG_SIZE, IMG_SIZE, true)
        }

        /* 2) imagen → ByteBuffer (x/127.5 − 1) */
        val buffer = ByteBuffer.allocateDirect(4 * IMG_SIZE * IMG_SIZE * 3)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        bmp.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)
        for (p in pixels) {
            val r = (p shr 16 and 0xFF).toFloat()
            val g = (p shr 8 and 0xFF).toFloat()
            val b = (p and 0xFF).toFloat()
            buffer.putFloat(r / 127.5f - 1f)
            buffer.putFloat(g / 127.5f - 1f)
            buffer.putFloat(b / 127.5f - 1f)
        }
        buffer.rewind()

        /* 3) inferencia */
        val outputs = mutableMapOf<String, Any>()
        outputs["output"] = Array(1) { FloatArray(MAX_OUT) }

        tflite.runSignature(mapOf("x" to buffer), outputs, "infer")

        @Suppress("UNCHECKED_CAST")
        val logitsFull = (outputs["output"] as Array<FloatArray>)[0]
        val logits9 = logitsFull.copyOfRange(0, ACTIVE_OUT)
        val probs9 = softmax(logits9)

        /* 4) Logcat */
        Log.i(TAG, "logits[0..14] = ${logitsFull.joinToString(prefix = "[", postfix = "]")}")
        Log.i(TAG, "probs [0..8 ] = ${probs9.joinToString(prefix = "[", postfix = "]")}")
    }
    // ───── utilidades ───────────────────────────────────────────
    private fun softmax(v: FloatArray): FloatArray {
        val expVals = v.map { exp(it.toDouble()).toFloat() }
        val sum = expVals.sum()
        return FloatArray(v.size) { i -> expVals[i] / sum }
    }

    private fun loadAsset(name: String): ByteBuffer =
        assets.openFd(name).use { fd ->
            val buf = ByteBuffer.allocateDirect(fd.declaredLength.toInt())
                .order(ByteOrder.nativeOrder())
            fd.createInputStream().channel.read(buf)
            buf.rewind()
            buf
        }

    private fun loadBitmap(name: String): Bitmap =
        assets.open(name).use { BitmapFactory.decodeStream(it) }
}
