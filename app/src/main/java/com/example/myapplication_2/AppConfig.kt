package com.example.myapplication_2


import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder



object AppConfig {

    private const val LABELS_FILE = "labels.txt"

    /** Etiquetas de usuario */
    lateinit var labels: List<String>
        private set
    /** Clases activas = tamaño de labels */
    var activeClasses: Int = 0
        private set
    const val MaxClasses: Int = 15
    const val IMG_SIZE: Int = 224

    fun init(context: Context) {
        val destFile = File(context.filesDir, LABELS_FILE)

        if (!destFile.exists()) {
            // Copiamos el original desde assets al internal storage
            context.assets.open(LABELS_FILE).use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            }
        }



        // Leemos siempre el fichero de internal storage
        labels = destFile.readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        // Actualizamos el número de clases activas
        activeClasses = labels.size
    }

    /**
     * Guarda una nueva lista de etiquetas en disco y actualiza
     * labels y activeClasses en memoria.
     */
    fun saveLabels(context: Context, newLabels: List<String>) {
        val destFile = File(context.filesDir, LABELS_FILE)
        destFile.printWriter().use { pw ->
            newLabels.forEach { pw.println(it) }
        }
        labels = newLabels
        activeClasses = labels.size
    }


}