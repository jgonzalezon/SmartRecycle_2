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

            // 1) Lista dinámica: etiquetas + “Añadir clase”
            val baseLabels = AppConfig.labels.toMutableList()
            baseLabels.add("Añadir clase")

            // 2) Adapter
            val spinnerAdapter = ArrayAdapter(
                root.context,
                android.R.layout.simple_spinner_item,
                baseLabels
            ).also {
                it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }
            spinner.adapter = spinnerAdapter

            // 3) Selección sin disparar evento
            spinner.setSelection(item.labelIdx, false)

            // 4) Listener
            spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>, view: View?, idx: Int, id: Long) {
                    val ctx = root.context

                    // Usuario pinchó “Añadir clase” (último elemento)
                    if (idx == baseLabels.lastIndex) {
                        if (AppConfig.labels.size >= 15) {
                            // Ya en límite: informar que restaure fábrica
                            AlertDialog.Builder(ctx)
                                .setTitle("Límite alcanzado")
                                .setMessage("Ya tienes 15 clases. Para añadir más etiquetas debes restablecer la configuración de fábrica.")
                                .setPositiveButton("OK", null)
                                .show()
                        } else {
                            // Todavía caben: abrir diálogo para nombre
                            showAddClassDialog(ctx) { newLabel ->
                                val newLabels = AppConfig.labels.toMutableList().apply { add(newLabel) }
                                AppConfig.saveLabels(ctx, newLabels)

                                // Rebuild spinner adapter
                                val updated = newLabels.toMutableList().apply { add("Añadir clase") }
                                val newAdapter = ArrayAdapter(
                                    ctx,
                                    android.R.layout.simple_spinner_item,
                                    updated
                                ).also {
                                    it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                                }
                                spinner.adapter = newAdapter

                                // Seleccionar la nueva
                                val newIdx = newLabels.lastIndex
                                spinner.setSelection(newIdx, false)

                                // Actualizar item y callback
                                item.labelIdx = newIdx
                                onLabelChange(position, newIdx)
                            }
                        }
                    }
                    // Cambio normal de etiqueta existente
                    else if (item.labelIdx != idx) {
                        item.labelIdx = idx
                        onLabelChange(position, idx)
                    }
                }

                override fun onNothingSelected(parent: AdapterView<*>) = Unit
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
        AlertDialog.Builder(context)
            .setTitle("Añadir nueva clase")
            .setView(edit)
            .setPositiveButton("Añadir") { _, _ ->
                val text = edit.text.toString().trim()
                if (text.isNotEmpty()) onNewLabel(text)
                else Toast.makeText(context, "El nombre no puede estar vacío", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancelar", null)
            .show()
    }
}