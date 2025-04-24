package com.example.myapplication_2


import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.myapplication_2.Tarea

class TareasAdapter(
    private val tareas: MutableList<Tarea>,
    private val onDelete: (Tarea) -> Unit
) : RecyclerView.Adapter<TareasAdapter.TareaViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TareaViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_tarea, parent, false)
        return TareaViewHolder(view)
    }

    override fun onBindViewHolder(holder: TareaViewHolder, position: Int) {
        val tarea = tareas[position]
        holder.bind(tarea, onDelete)
    }

    override fun getItemCount() = tareas.size

    fun addTarea(tarea: Tarea) {
        tareas.add(tarea)
        notifyItemInserted(tareas.size - 1)
    }

    fun removeTarea(tarea: Tarea) {
        val index = tareas.indexOf(tarea)
        if (index != -1) {
            tareas.removeAt(index)
            notifyItemRemoved(index)
        }
    }

    class TareaViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvNombre: TextView = itemView.findViewById(R.id.tvNombre)
        private val btnDelete: Button = itemView.findViewById(R.id.btnDelete)

        // Si tienes contadores P1/P2:
        private val btnP1Plus: Button = itemView.findViewById(R.id.btnP1Plus)
        private val btnP1Minus: Button = itemView.findViewById(R.id.btnP1Minus)
        private val tvCountP1: TextView = itemView.findViewById(R.id.tvCountP1)

        private val btnP2Plus: Button = itemView.findViewById(R.id.btnP2Plus)
        private val btnP2Minus: Button = itemView.findViewById(R.id.btnP2Minus)
        private val tvCountP2: TextView = itemView.findViewById(R.id.tvCountP2)

        fun bind(tarea: Tarea, onDelete: (Tarea) -> Unit) {
            tvNombre.text = tarea.nombre

            // Contador persona 1
            tvCountP1.text = tarea.countP1.toString()
            btnP1Plus.setOnClickListener {
                tarea.countP1++
                tvCountP1.text = tarea.countP1.toString()
            }
            btnP1Minus.setOnClickListener {
                if (tarea.countP1 > 0) {
                    tarea.countP1--
                    tvCountP1.text = tarea.countP1.toString()
                }
            }

            // Contador persona 2
            tvCountP2.text = tarea.countP2.toString()
            btnP2Plus.setOnClickListener {
                tarea.countP2++
                tvCountP2.text = tarea.countP2.toString()
            }
            btnP2Minus.setOnClickListener {
                if (tarea.countP2 > 0) {
                    tarea.countP2--
                    tvCountP2.text = tarea.countP2.toString()
                }
            }

            // Eliminar tarea
            btnDelete.setOnClickListener {
                onDelete(tarea)
            }
        }
    }
}