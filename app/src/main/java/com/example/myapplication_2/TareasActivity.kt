package com.example.myapplication_2


import android.content.Context
import android.os.Bundle
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication_2.Tarea
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

class TareasActivity : AppCompatActivity() {

    companion object {
        private const val PREFS_NAME = "MisTareasPrefs"
        private const val KEY_TAREAS = "KEY_LISTA_TAREAS"
    }

    private lateinit var rvTareas: RecyclerView
    private lateinit var adapter: TareasAdapter

    // Lista de tareas en memoria
    private val listaTareas = mutableListOf<Tarea>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tareas)

        rvTareas = findViewById(R.id.rvTareas)
        rvTareas.layoutManager = LinearLayoutManager(this)

        // Creamos el adapter y definimos la acción para borrar una tarea
        adapter = TareasAdapter(listaTareas) { tareaAEliminar ->
            adapter.removeTarea(tareaAEliminar)
            guardarTareasEnPrefs()
        }
        rvTareas.adapter = adapter

        // Cargar tareas si existen en SharedPreferences
        cargarTareasDePrefs()

        // FAB para añadir una nueva tarea
        val fabAddTask = findViewById<FloatingActionButton>(R.id.fabAddTask)
        fabAddTask.setOnClickListener {
            mostrarDialogoNuevaTarea()
        }
    }

    /**
     * Muestra un AlertDialog con un EditText para introducir el nombre de la tarea
     */
    private fun mostrarDialogoNuevaTarea() {
        val editText = EditText(this)
        editText.hint = "Nombre de la tarea"

        AlertDialog.Builder(this)
            .setTitle("Nueva Tarea")
            .setView(editText)
            .setPositiveButton("Guardar") { dialog, _ ->
                val nombreTarea = editText.text.toString().trim()
                if (nombreTarea.isNotEmpty()) {
                    val nuevaTarea = Tarea(
                        id = System.currentTimeMillis(),
                        nombre = nombreTarea
                    )
                    adapter.addTarea(nuevaTarea)
                    guardarTareasEnPrefs() // Guardar cambios en memoria persistente
                } else {
                    Toast.makeText(this, "El nombre no puede estar vacío", Toast.LENGTH_SHORT).show()
                }
                dialog.dismiss()
            }
            .setNegativeButton("Cancelar") { dialog, _ ->
                dialog.dismiss()
            }
            .show()
    }

    /**
     * Convierte la lista de tareas a JSON y la guarda en SharedPreferences
     */
    private fun guardarTareasEnPrefs() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val editor = prefs.edit()

        val gson = Gson()
        val jsonTareas = gson.toJson(listaTareas)
        editor.putString(KEY_TAREAS, jsonTareas)
        editor.apply()  // o editor.commit()
    }

    /**
     * Carga la lista de tareas desde SharedPreferences y actualiza la lista en memoria
     */
    private fun cargarTareasDePrefs() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val jsonTareas = prefs.getString(KEY_TAREAS, null) ?: return

        val gson = Gson()
        val type = object : TypeToken<MutableList<Tarea>>() {}.type
        val listaCargada: MutableList<Tarea> = gson.fromJson(jsonTareas, type)

        listaTareas.clear()
        listaTareas.addAll(listaCargada)
        adapter.notifyDataSetChanged()
    }
}