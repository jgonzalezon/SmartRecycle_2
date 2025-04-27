package com.example.myapplication_2


import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

@RunWith(AndroidJUnit4::class)
class TfliteMultiClassInstrumentedTest {

    // Nombre de tu archivo .tflite dentro de /src/androidTest/assets/
    private val MODEL_FILE = "waste_model_V3_multi.tflite"
    private val IMAGE_SIZE = 224

    // Lista de etiquetas en orden exacto como las entrenaste (9 clases)
    private val CLASS_NAMES = listOf(
        "Basura general",
        "Carton",
        "Aluminio",
        "Organica",
        "Papel",
        "Plastico",
        "Vidrio",
        "Vegetacion",
        "Textil"
    )

    @Test
    fun multiClassAccuracy() {
        // Contexto y AssetManager
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val assets: AssetManager = ctx.assets

        // 1) Abrir y mapear el modelo TFLite
        val fd = assets.openFd(MODEL_FILE)
        val modelBuffer: MappedByteBuffer = fd.createInputStream().channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
        val interpreter = Interpreter(modelBuffer)

        // 2) Leer el fichero test_labels.txt
        val pairs = mutableListOf<Pair<String,Int>>()
        assets.open("test_multiclase/test_labels.txt").bufferedReader().useLines { lines ->
            lines.forEach { line ->
                val (fileName, lblStr) = line.trim().split(",")
                val lbl = lblStr.toIntOrNull() ?: return@forEach
                pairs += fileName to lbl
            }
        }

        var hits = 0
        // 3) Recorremos cada par (imagen y etiqueta real)
        pairs.forEach { (fileName, trueLabel) ->
            // 3.1) Carga y preprocessing de la bitmap
            val bmp: Bitmap = BitmapFactory.decodeStream(assets.open("test_multiclase/images/$fileName"))
            val input = preprocess(bmp, IMAGE_SIZE)

            // 4) Crear array de salida [1][9]
            val output = Array(1) { FloatArray(CLASS_NAMES.size) }

            // 5) Ejecutar la inferencia
            interpreter.run(input, output)

            // 6) Tomar argmax como clase predicha
            val scores = output[0]
            val predIndex = scores.indices.maxByOrNull { scores[it] } ?: 0

            if (predIndex == trueLabel) hits++
        }

        // 7) Calcular accuracy y aserción
        val accuracy = hits.toFloat() / pairs.size
        println(">>> Test set multi-class accuracy = $accuracy  ($hits / ${pairs.size})")
        assertTrue("Accuracy mínima esperada 0.70, pero fue $accuracy", accuracy >= 0.7)
    }

    /** Normaliza la imagen en array [1][224][224][3] con mean/std de ImageNet */
    private fun preprocess(b: Bitmap, size: Int): Array<Array<Array<FloatArray>>> {
        val scaled = Bitmap.createScaledBitmap(b, size, size, true)
        val input = Array(1) { Array(size) { Array(size) { FloatArray(3) } } }
        for (y in 0 until size) for (x in 0 until size) {
            val px = scaled.getPixel(x, y)
            // Normalización ImageNet
            input[0][y][x][0] = (Color.red(px)   / 255f - 0.485f) / 0.229f
            input[0][y][x][1] = (Color.green(px) / 255f - 0.456f) / 0.224f
            input[0][y][x][2] = (Color.blue(px)  / 255f - 0.406f) / 0.225f
        }
        return input
    }
}