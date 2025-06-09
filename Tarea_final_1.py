# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
#Importación de librerías
import numpy as np

#Definición de hiper parámetros alpha y gamma
gamma = 0.75
alpha = 0.90

#PARTE 1 - DEFINICIÓN DEL ENTORNO

#Definición de los estados
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6, 
                     'H': 7, 
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# Definición de prioridades en el sistema
# 1 representa la prioridad más alta y así sucesivamente
priority_state = {
    1: 'G',   # Prioridad 1
    2: 'K',   # Prioridad 2  
    3: 'L',   # Prioridad 3
    4: 'J',   # Prioridad 4
    5: 'A',   # Prioridad 5
    6: 'I',   # Prioridad 6
    7: 'H',   # Prioridad 7
    8: 'C',   # Prioridad 8
    9: 'B',   # Prioridad 9
    10: 'D',  # Prioridad 10
    11: 'F',  # Prioridad 11
    12: 'E'   # Prioridad 12
}

#Definición de las acciones
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Definición de las recompensas
# Columnas:    A,B,C,D,E,F,G,H,I,J,K,L
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0], # A
              [1,0,1,0,0,1,0,0,0,0,0,0], # B
              [0,1,0,0,0,0,1,0,0,0,0,0], # C
              [0,0,0,0,0,0,0,1,0,0,0,0], # D
              [0,0,0,0,0,0,0,0,1,0,0,0], # E
              [0,1,0,0,0,0,0,0,0,1,0,0], # F
              [0,0,1,0,0,0,1,1,0,0,0,0], # G
              [0,0,0,1,0,0,1,0,0,0,0,1], # H
              [0,0,0,0,1,0,0,0,0,1,0,0], # I
              [0,0,0,0,0,1,0,0,1,0,1,0], # J
              [0,0,0,0,0,0,0,0,0,1,0,1], # K
              [0,0,0,0,0,0,0,1,0,0,1,0]])# L

#PARTE 2 - CONSTRUCCIÓN DE LA SOLUCIÓN DE IA CON Q-LEARNING

#Transformación inversa de estados a ubicaciones
state_to_location = {state : location for location, state in location_to_state.items()}

#Adquiere el número de prioridad por cada letra en el diccionario
"""
    -priority_state.items() devuelve todos los items (clave, valor) del diccionario.
    
    -for priority, state in priority_state.items() va a iterar por cada clave (número/priority) y dentro de
    cada una de esas claves que itera, va a devolver devolver el valor (state/letra).
    
    -state: priority va a invertir las claves y valores, ahora las claves serán las letras (state) y el valor serán
    los números de prioridad (priority).
"""
state_priority = {state: priority for priority, state in priority_state.items()}

#Crear la función final que nos devuelva la ruta óptima
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    
    #Inicialización de los valores Q
    Q = np.array(np.zeros([12, 12]))
    
    
    for state_name, priority in state_priority.items():
        state_index = location_to_state[state_name]
        # Mayor prioridad (número menor) = mayor bonus
        bonus = (13 - priority) * 5  # Ajusta el multiplicador según necesites
        
        # Aplicar bonus a todas las transiciones QUE LLEVAN A ese estado prioritario
        for i in range(12):
            if R_new[i, state_index] > 0:  # Si existe conexión hacia el estado prioritario
                R_new[i, state_index] += bonus

    #Implementación del proceso de Q-Learning
    for i in range(1000):
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD
    
    
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state, ])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

#PARTE 3 - PONER EL MODELO EN PRODUCCIÓN

# Imprimir la ruta final
print("Ruta Elegida:")
print(route('E','G'))
