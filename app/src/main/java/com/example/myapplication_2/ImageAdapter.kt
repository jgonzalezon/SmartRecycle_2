package com.example.myapplication_2

import android.app.AlertDialog
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication_2.databinding.RowImageBinding
import android.view.MotionEvent


data class ImageItem(
    val uri: Uri,
    val bitmap: Bitmap,
    var labelIdx: Int
)

class ImageAdapter(
    private val onLabelChange: (position: Int, newIdx: Int) -> Unit,
    labels: List<String>
) : RecyclerView.Adapter<ImageAdapter.VH>() {

    val items = mutableListOf<ImageItem>()

    inner class VH(val binding: RowImageBinding) : RecyclerView.ViewHolder(binding.root)




    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val binding = RowImageBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return VH(binding)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val item = items[position]
        with(holder.binding) {
            imgThumb.setImageBitmap(item.bitmap)

            // 1) Base de etiquetas + “Añadir clase”
            val baseLabels = AppConfig.labels.toMutableList().apply { add("Añadir clase") }
            val spinnerAdapter = ArrayAdapter(
                root.context,
                android.R.layout.simple_spinner_item,
                baseLabels
            ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            spinner.adapter = spinnerAdapter

            // 2) Selección inicial
            spinner.setSelection(item.labelIdx)

            // 3) Listener de cambio (lo definimos una sola vez y lo reutilizamos)
            val selectionListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>, view: View?, idx: Int, id: Long) {
                    val addPos = parent.adapter.count - 1
                    if (idx == addPos) {
                        // **Aquí** abrimos el diálogo
                        if (AppConfig.labels.size >= AppConfig.MaxClasses) {
                            AlertDialog.Builder(root.context)
                                .setTitle("Límite alcanzado")
                                .setMessage("Ya tienes ${AppConfig.MaxClasses} clases. Restablece fábrica para más.")
                                .setPositiveButton("OK", null)
                                .show()
                        } else {
                            showAddClassDialog(root.context) { newLabel ->
                                // 1) Guardar
                                val newLabels = AppConfig.labels + newLabel
                                AppConfig.saveLabels(root.context, newLabels)
                                // 2) Reconstruir spinner
                                val updated = newLabels.toMutableList().apply { add("Añadir clase") }
                                val newAdapter = ArrayAdapter(
                                    root.context,
                                    android.R.layout.simple_spinner_item,
                                    updated
                                ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
                                spinner.adapter = newAdapter
                                // 3) Seleccionar la recién creada
                                val newIdx = newLabels.lastIndex
                                spinner.setSelection(newIdx)
                                // 4) Actualizar modelo y callback
                                item.labelIdx = newIdx
                                onLabelChange(position, newIdx)
                            }
                        }
                    } else if (item.labelIdx != idx) {
                        // Cambio normal
                        item.labelIdx = idx
                        onLabelChange(position, idx)
                    }
                }
                override fun onNothingSelected(parent: AdapterView<*>) = Unit
            }

            // 4) Asignamos el listener
            spinner.onItemSelectedListener = selectionListener

            // 5) OnTouch para refrescar etiquetas justo antes de abrir
            spinner.setOnTouchListener { _, event ->
                if (event.action == MotionEvent.ACTION_DOWN) {
                    val dynamic = AppConfig.labels.toMutableList().apply { add("Añadir clase") }
                    val refreshed = ArrayAdapter(
                        root.context,
                        android.R.layout.simple_spinner_item,
                        dynamic
                    ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
                    spinner.adapter = refreshed
                    // Necesitamos re-asignar el mismo listener tras cambiar adapter:
                    spinner.onItemSelectedListener = selectionListener
                    // Y volvemos a fijar la selección actual:
                    spinner.setSelection(item.labelIdx)
                }
                false
            }
        }
    }

    override fun getItemCount(): Int = items.size

    fun add(item: ImageItem) {
        items.add(item)
        notifyItemInserted(items.lastIndex)
    }

    // ——— Helpers ———

    /** Muestra un diálogo con un EditText para introducir el nombre de la nueva etiqueta */
    private fun showAddClassDialog(
        context: Context,
        onNewLabel: (String) -> Unit
    ) {
        val edit = EditText(context).apply { hint = "Nombre de la nueva clase" }
        // Creamos el AlertDialog en vez de usar .show() directo
        val dialog = AlertDialog.Builder(context)
            .setTitle("Añadir nueva clase")
            .setView(edit)
            .setPositiveButton("Añadir", null)   // listener lo configuramos más abajo
            .setNegativeButton("Cancelar", null)
            .create()

        dialog.setOnShowListener {
            // Sobrescribimos el click del botón “Añadir”
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val text = edit.text.toString().trim()
                if (text.isEmpty()) {
                    Toast.makeText(context, "El nombre no puede estar vacío", Toast.LENGTH_SHORT).show()
                } else {
                    // Llamamos al callback y cerramos
                    onNewLabel(text)
                    dialog.dismiss()
                }
            }
        }
        dialog.show()
    }
}