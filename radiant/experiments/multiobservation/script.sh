#!/bin/bash

# Verifica que se haya proporcionado un argumento
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 {train|play}"
    exit 1
fi

# Guarda el argumento en una variable
mode=$1

# Define los arrays de valores de n y delta
n_values=(4 10 20)

# Verifica si el modo es train o play
if [ "$mode" = "train" ]; then
    script_name="train.py"
elif [ "$mode" = "play" ]; then
    script_name="play.py"
else
    echo "Modo no válido: $mode. Usa 'train' o 'play'."
    exit 1
fi

# Recorre cada combinación de n y delta
for n in "${n_values[@]}"; do
  python "$script_name" "$n"
done
