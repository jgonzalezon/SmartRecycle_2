package com.example.myapplication_2

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Delegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.widget.PopupMenu
import com.example.myapplication_2.AppConfig

class MainActivity : AppCompatActivity() {


    private val MODEL_FILE      = "continual_trainable.tflite"
    // Flag que indica si hay cambios de entrenamiento sin guardar
    private var isDirty = false
    private lateinit var tflite: Interpreter
    private lateinit var trainingLauncher: ActivityResultLauncher<Intent>



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


        // Registras el lanzador
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
        // Entrada: 'path' → array de String con la ruta destino
        val inputs  = mapOf("path" to arrayOf(path))
        // Salida: 'checkpoint_path' → array donde se devolverá la ruta escrita
        // 2) Outputs como MutableMap<String, Any>
        val outputs: MutableMap<String, Any> = mutableMapOf(
            "checkpoint_path" to arrayOf("")  // inicializamos vacío
        )
        // Lanza la firma "save"
        tflite.runSignature(inputs, outputs, "save")

        // Recoge la ruta confirmada
        @Suppress("UNCHECKED_CAST")
        val savedPath = (outputs["checkpoint_path"] as Array<String>)[0]
        Toast.makeText(this,
            "Checkpoint guardado en:\n$savedPath",
            Toast.LENGTH_LONG
        ).show()
    }

    /** Diálogo genérico de “¿Guardar cambios?” */
    private fun showSaveDialog(onConfirm: () -> Unit) {
        AlertDialog.Builder(this)
            .setTitle("Guardar checkpoint")
            .setMessage("¿Deseas guardar los cambios entrenados?")
            .setPositiveButton("Guardar") { _, _ ->
                // Guarda el checkpoint
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
            .setMessage("¿Seguro que quieres eliminar el checkpoint y restablecer etiquetas de fábrica?")
            .setPositiveButton("Eliminar") { _, _ ->
                // 1) Borrar checkpoint
                val ckptFile = File(filesDir, "model.ckpt")
                val ckptDeleted = if (ckptFile.exists()) ckptFile.delete() else false

                // 2) Borrar fichero de labels del usuario
                val labelsFile = File(filesDir, "labels.txt")
                val labelsDeleted = if (labelsFile.exists()) labelsFile.delete() else false

                // 3) Notificar al usuario
                when {
                    ckptDeleted && labelsDeleted ->
                        Toast.makeText(this, "Checkpoint y etiquetas restaurados a fábrica", Toast.LENGTH_SHORT).show()
                    ckptDeleted ->
                        Toast.makeText(this, "Checkpoint borrado; etiquetas en fábrica", Toast.LENGTH_SHORT).show()
                    labelsDeleted ->
                        Toast.makeText(this, "Etiquetas restauradas a fábrica; no había checkpoint", Toast.LENGTH_SHORT).show()
                    else ->
                        Toast.makeText(this, "No había checkpoint ni etiquetas personalizadas", Toast.LENGTH_SHORT).show()
                }

                // 4) Reinicializar AppConfig para recargar las labels de assets
                AppConfig.init(this@MainActivity)

                // 5) Si tenías un menú o botones dependientes de isDirty/activeClasses, recarga UI:
                invalidateOptionsMenu()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }

}