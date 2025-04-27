package com.example.myapplication_2

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.InputStreamReader

@RunWith(AndroidJUnit4::class)
class TestImageClassifier {

    private val MODEL_FILE = "waste_model_V3.tflite"
    private val IMAGE_SIZE = 224
    private val THRESHOLD = 0.5f
    private val POS_LABEL = "Reciclable"
    private val NEG_LABEL = "Organica"

    @Test fun accuracyOverTestSet() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val assets: AssetManager = ctx.assets
        // 1) Carga el modelo TFLite
        val fid = assets.openFd(MODEL_FILE)
        val model = Interpreter(fid.createInputStream().channel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            fid.startOffset,
            fid.declaredLength))

        // 2) Lee el listado de ground-truth
        val pairs = mutableListOf<Pair<String,String>>()
        assets.open("test_binary/test_labels.txt").bufferedReader().useLines { lines ->
            lines.forEach {
                val (name, lbl) = it.trim().split(",")
                pairs += Pair(name, lbl)
            }
        }

        var hits = 0
        pairs.forEach { (fileName, trueLabel) ->
            // 3) Carga la imagen (224x224, RGB float normalizado)…
            val bmp = BitmapFactory.decodeStream(assets.open("test_binary/test_images/$fileName"))
            val input = preprocess(bmp, IMAGE_SIZE)   // tu función de normalizar

            // 4) Invoca al intérprete
            val output = Array(1) { FloatArray(1) }
            model.run(input, output)

            // 5) Post-process
            val pred = if (output[0][0] > THRESHOLD) POS_LABEL else NEG_LABEL
            if (pred == trueLabel) hits++
        }

        val accuracy = hits.toFloat() / pairs.size
        println(">>> Test set accuracy = $accuracy")

        // 6) Asserción para que el test pase/falle
        assertEquals("Accuracy < 0.80", true, accuracy >= 0.80f)
    }

    // Ejemplo de pre-processing: bitmap a [1,224,224,3] float
    private fun preprocess(b: Bitmap, size: Int): Array<Array<Array<FloatArray>>> {
        val scaled = Bitmap.createScaledBitmap(b, size, size, true)
        val inBuffer = Array(1){ Array(size){ Array(size){ FloatArray(3) } } }
        for (y in 0 until size) for (x in 0 until size) {
            val px = scaled.getPixel(x,y)
            inBuffer[0][y][x][0] = (Color.red(px)/255f - 0.485f)/0.229f
            inBuffer[0][y][x][1] = (Color.green(px)/255f - 0.456f)/0.224f
            inBuffer[0][y][x][2] = (Color.blue(px)/255f - 0.406f)/0.225f
        }
        return inBuffer
    }
}