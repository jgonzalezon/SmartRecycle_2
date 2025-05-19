package com.example.myapplication_2

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.widget.PopupMenu

class MainActivity : AppCompatActivity() {


    private val MODEL_FILE      = "continual_trainable_B0.tflite"
    // Flag que indica si hay cambios de entrenamiento sin guardar
    private var isDirty = false
    private lateinit var tflite: Interpreter
    private lateinit var trainingLauncher: ActivityResultLauncher<Intent>

    private fun runSignatureReset(
        inputs: Map<String, Any>,
        outputs: Map<String, Any>,
        signatureKey: String
    ) {
        tflite.runSignature(inputs, outputs, signatureKey)
        tflite.allocateTensors()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_menu)
        AppConfig.init(this)

        // Seguir configuración del sistema :
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM)

        // Carga modelo
        val options = Interpreter.Options().apply {
            // 1. Añade el Flex delegate para ops no nativas

            // 2. (opcional) ajusta número de hilos
            setNumThreads(4)
        }
        tflite = Interpreter(loadAsset(MODEL_FILE), options)



        // Referencia a los botones
        val btnMenu = findViewById<Button>(R.id.btnPopupMenu)
        val btnEjecutar = findViewById<Button>(R.id.btnEjecutar)
        val btnSalir    = findViewById<Button>(R.id.btnSalir)
        val btnTrain    = findViewById<Button>(R.id.btnTrain)
        val btnSave     = findViewById<Button>(R.id.btnSave)
        // 2) Al hacer click, crea y muestra el PopupMenu
        btnMenu.setOnClickListener { view ->
            PopupMenu(this, view).apply {
                // infla el xml que creamos en /res/menu/menu_main.xml
                menuInflater.inflate(R.menu.menu_main, menu)

                // 3) Maneja los clicks de cada opción
                setOnMenuItemClickListener { item ->
                    when (item.itemId) {
                        R.id.action_help -> {
                            // Mostrar diálogo de “Ayuda futura”
                            AlertDialog.Builder(this@MainActivity)
                                .setTitle(R.string.help_title)
                                .setMessage(R.string.help_message)
                                .setPositiveButton(R.string.ok, null)
                                .show()
                            true
                        }
                        R.id.action_factory_reset -> {
                            // diálogo de confirmación
                            showFactoryResetDialog()
                            true
                        }
                        else -> false
                    }
                }

                // 4) finalmente…
                show()
            }
        }


        // Registra el lanzador
        trainingLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == RESULT_OK) {
                isDirty = true
                btnSave.isEnabled = true
                Toast.makeText(this, "Entrenamiento on-device completado", Toast.LENGTH_SHORT).show()
            }
        }
        // Acción para el botón "Ejecutar la Aplicación"
        btnEjecutar.setOnClickListener {
            startActivity(Intent(this, CamaraActivity::class.java))
            Toast.makeText(this, "Ejecutando la aplicación...", Toast.LENGTH_SHORT).show()
        }

        btnTrain.setOnClickListener {
            trainingLauncher.launch(Intent(this, TrainingActivity::class.java))
        }

        // Acción para el botón "Ejecutar la Aplicación"
        btnSave.setOnClickListener {
            showSaveDialog(onConfirm = {
                // Una vez confirmado y guardado:
                Toast.makeText(this, "Entrenamiento guardado", Toast.LENGTH_SHORT).show()
                isDirty = false
                btnSave.isEnabled = false
            })
        }
        // Acción para el botón "Salir"
        btnSalir.setOnClickListener {
            handleExit()
        }
    }

    /** Ejecuta la firma “save” y vuelca el checkpoint al path elegido */
    fun saveCheckpoint(path: String) {
        // Preparamos inputs/outputs
        val inputs = mapOf("path" to arrayOf(path))
        val outputs: MutableMap<String, Any> = mutableMapOf(
            "checkpoint_path" to arrayOf("")  // placeholder
        )

        // Ejecutamos la firma y reseteamos el intérprete
        runSignatureReset(inputs, outputs, "save")

        // Extraemos la ruta devuelta
        @Suppress("UNCHECKED_CAST")
        val savedPath = (outputs["checkpoint_path"] as Array<String>)[0]

        Toast.makeText(this,
            "Checkpoint guardado en:\n$savedPath",
            Toast.LENGTH_LONG
        ).show()
    }

    private fun showSaveDialog(onConfirm: () -> Unit) {
        AlertDialog.Builder(this)
            .setTitle("Guardar checkpoint")
            .setMessage("¿Deseas guardar los cambios entrenados?")
            .setPositiveButton("Guardar") { _, _ ->
                val targetFile = File(filesDir, "model.ckpt")
                saveCheckpoint(targetFile.absolutePath)
                onConfirm()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

    private fun handleExit() {
        if (isDirty) {
            AlertDialog.Builder(this)
                .setTitle("Salir")
                .setMessage("Hay cambios sin guardar. ¿Guardar antes de salir?")
                .setPositiveButton("Sí") { _, _ ->
                    showSaveDialog {
                        finish()
                    }
                }
                .setNegativeButton("No") { _, _ ->
                    finish()
                }
                .show()
        } else {
            finish()
        }
    }


    /** Si hay cambios sin guardar, al pulsar atrás pedimos confirmación */
    override fun onBackPressed() {
        if (isDirty) {
            showSaveDialog(onConfirm = { super.onBackPressedDispatcher })
        } else {
            super.onBackPressedDispatcher
        }
    }
    private fun loadAsset(name: String): ByteBuffer =
        assets.openFd(name).use { fd ->
            ByteBuffer.allocateDirect(fd.declaredLength.toInt())
                .order(ByteOrder.nativeOrder())
                .also { fd.createInputStream().channel.read(it); it.rewind() }
        }


    // Diálogo de confirmación para “Fábrica”
    private fun showFactoryResetDialog() {
        AlertDialog.Builder(this)
            .setTitle("Restablecer fábrica")
            .setMessage("¿Seguro que quieres eliminar el checkpoint, el buffer de replay y restablecer etiquetas de fábrica?")
            .setPositiveButton("Eliminar") { _, _ ->
                // 1) Borrar checkpoint
                val ckptFile = File(filesDir, "model.ckpt")
                val ckptDeleted = if (ckptFile.exists()) ckptFile.delete() else false

                // 2) Borrar fichero de labels del usuario
                val labelsFile = File(filesDir, "labels.txt")
                val labelsDeleted = if (labelsFile.exists()) labelsFile.delete() else false

                // 3) Borrar buffer de replay local
                val replayFile = File(filesDir, "replay_buffer_local.npz")
                val replayDeleted = if (replayFile.exists()) replayFile.delete() else false

                // 4) Notificar al usuario
                val message = when {
                    ckptDeleted && labelsDeleted && replayDeleted ->
                        "Checkpoint, etiquetas y replay buffer restaurados a fábrica"
                    ckptDeleted && labelsDeleted ->
                        "Checkpoint y etiquetas restaurados; replay buffer no existía"
                    ckptDeleted && replayDeleted ->
                        "Checkpoint y replay buffer borrados; etiquetas en fábrica"
                    labelsDeleted && replayDeleted ->
                        "Etiquetas y replay buffer restaurados; checkpoint no existía"
                    ckptDeleted ->
                        "Checkpoint borrado; etiquetas y replay buffer en fábrica"
                    labelsDeleted ->
                        "Etiquetas restauradas a fábrica; ni checkpoint ni replay buffer previos"
                    replayDeleted ->
                        "Replay buffer restaurado a fábrica; checkpoint y etiquetas en fábrica"
                    else ->
                        "No había checkpoint, etiquetas personalizadas ni replay buffer"
                }
                Toast.makeText(this, message, Toast.LENGTH_SHORT).show()

                // 5) Reinicializar AppConfig para recargar las labels de assets
                AppConfig.init(this)

                // 6) Refrescar UI si usas menús o vistas basados en activeClasses
                invalidateOptionsMenu()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }


}