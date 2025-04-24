package com.example.myapplication_2;


/**
 * Representa una tarea con un nombre y dos contadores, uno para cada persona.
 *
 * @param id Identificador único de la tarea.
 * @param nombre Nombre o descripción de la tarea.
 * @param countP1 Contador para la persona 1, valor inicial 0.
 * @param countP2 Contador para la persona 2, valor inicial 0.
 */
data class Tarea(
        val id: Long,
        val nombre: String,
        var countP1: Int = 0,
        var countP2: Int = 0
)