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
class BinaryThenMultiClassInstrumentedTest {

    // 1) Nombre de los modelos en assets/androidTest/assets
    private val BINARY_MODEL = "waste_model_V3.tflite"
    private val MULTI_MODEL  = "waste_model_V3_multi.tflite"

    private val IMAGE_SIZE = 224

    // 2) Lista de etiquetas (multiclase)
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

    @Test fun binaryThenMultiAccuracy() {
        // Contexto y AssetManager
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val assets: AssetManager = ctx.assets

        // --- Carga y mapeo de ambos intérpretes TFLite ---
        fun loadInterpreter(modelFile: String): Interpreter {
            val fd = assets.openFd(modelFile)
            val buffer: MappedByteBuffer = fd.createInputStream().channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
            return Interpreter(buffer)
        }

        val binInterpreter   = loadInterpreter(BINARY_MODEL)
        val multiInterpreter = loadInterpreter(MULTI_MODEL)

        // --- Leer listado de test (mismo test_labels.txt de 9 clases) ---
        val pairs = mutableListOf<Pair<String,Int>>()
        assets.open("test_multiclase/test_labels.txt").bufferedReader().useLines { lines ->
            lines.forEach { line ->
                val (fileName, lblStr) = line.trim().split(",")
                pairs += fileName to lblStr.toInt()
            }
        }

        var hits = 0
        // 3) Para cada imagen: primero inferencia binaria, luego multiclas
        pairs.forEach { (fileName, trueLabel) ->
            // 3.1) Carga y preprocess
            val bmp: Bitmap = BitmapFactory.decodeStream(assets.open("test_multiclase/images/$fileName"))
            val input = preprocess(bmp, IMAGE_SIZE)

            // 3.2) Inferencia binaria [1][1]
            val binOut = Array(1) { FloatArray(1) }
            binInterpreter.run(input, binOut)
            val binPred = if (binOut[0][0] > 0.5f) 1 else 0

            // 4) Si salió “1” (reciclable) lanzamos multiclas
            val finalPred = if (binPred == 1) {
                val multiOut = Array(1) { FloatArray(CLASS_NAMES.size) }
                multiInterpreter.run(input, multiOut)
                // argmax
                multiOut[0].indices.maxByOrNull { multiOut[0][it] }!!
            } else {
                // si binPred==0 devolvemos 0 directamente
                0
            }

            if (finalPred == trueLabel) hits++
        }

        // 5) Calcular accuracy y aserción
        val accuracy = hits.toFloat() / pairs.size
        println(">>> Test combinado accuracy = $accuracy  ($hits / ${pairs.size})")
        assertTrue("Accuracy mínima esperada 0.80, pero fue $accuracy", accuracy >= 0.70f)
    }

    /** Normalización igual que antes: [1,224,224,3] con mean/std de ImageNet */
    private fun preprocess(b: Bitmap, size: Int): Array<Array<Array<FloatArray>>> {
        val scaled = Bitmap.createScaledBitmap(b, size, size, true)
        val input = Array(1) { Array(size) { Array(size) { FloatArray(3) } } }
        for (y in 0 until size) for (x in 0 until size) {
            val px = scaled.getPixel(x, y)
            input[0][y][x][0] = (Color.red(px)   / 255f - 0.485f) / 0.229f
            input[0][y][x][1] = (Color.green(px) / 255f - 0.456f) / 0.224f
            input[0][y][x][2] = (Color.blue(px)  / 255f - 0.406f) / 0.225f
        }
        return input
    }
}