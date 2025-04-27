package com.example.myapplication_2

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate


class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_segunda)
        // Seguir configuración del sistema :
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM)

        // Referencia a los botones
        val btnEjecutar = findViewById<Button>(R.id.btnEjecutar)
        val btnAyuda = findViewById<Button>(R.id.btnAyuda)
        val btnSalir = findViewById<Button>(R.id.btnSalir)
        //val btnTareas = findViewById<Button>(R.id.btnTareas)

        // Acción para el botón "Ejecutar la Aplicación"
        btnEjecutar.setOnClickListener {
            // Lanzar la CameraActivity
            val intent = Intent(this, CamaraActivity::class.java)
            startActivity(intent)
            Toast.makeText(this, "Ejecutando la aplicación...", Toast.LENGTH_SHORT).show()
        }


/*
btnTareas.setOnClickListener {
    // Abrir la com.example.myapplication_2.TareasActivity
    val intent = Intent(this, TareasActivity::class.java)
    startActivity(intent)
}
*/

// Acción para el botón "Ayuda"
btnAyuda.setOnClickListener {
    // Mostrar un diálogo de ayuda
    AlertDialog.Builder(this)
        .setTitle("Ayuda")
        .setMessage("Esta es la ayuda de la aplicación.\n\nAquí puedes incluir instrucciones o información relevante.")
        .setPositiveButton("OK", null)
        .show()
}

// Acción para el botón "Salir"
btnSalir.setOnClickListener {
    // Finaliza la Activity (cierra la app si es la única Activity)
    finish()
}
}
}
