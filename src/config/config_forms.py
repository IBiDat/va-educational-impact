# Nomenclatura:
# tc = test conocimiento
# ta = test autoeficacia
# tcc = test carga congnitiva (solo en el cuestionario post-test)

# Global Variables
TC_TIPOS = ['retencion', 'transferencia']
PERIODOS = ['pre', 'post']
MAX_PUNTUACION_TC = 10
MAX_PUNTUACION_TC_TIPO = 5
MAX_PUNTUACION_TA = 7
MAX_PUNTUACION_TA = 10
TC_RESPUESTAS_CORRECTAS = {

    'pre': {

        'retencion': [
            'Describir y resumir un conjunto de datos recogidos sin sacar conclusiones sobre una población más amplia.',
            'Coche',
            6,
            'La distancia media aproximada entre cada dato concreto y la media del conjunto.',
            'Para analizar y comparar la relación entre dos variables categóricas simultáneamente.'
        ],

        'transferencia': [
            'La mediana, porque es resistente a los valores extremos y representa mejor al group mayoritario.',
            'Que el 50% de los alumnos sacó notas entre 50 y 80',
            'No, porque la estadística descriptiva sólo resume los datos recogidos (los 20 pacientes) sin sacar conclusiones de la población total.',
            'El Grupo B, porque sus valores están más dispersos y alejados de su media que los del Grupo A.',
            'Gráfico circular / tarta (Pie chart)'
        ]
    },

    'post': {

        'retencion': [
            'Resumir y describir los tiempos de esos 10 atletas específicamente.',
            'Rojo.',
            '25 m2 ambos.',
            'El grado medio de dispersión o separación de los valores alrededor de la media.',
            'Mostrar con qué frecuencia aparece cada valor distinto en una sola variable.'
        ],

        'transferencia': [
            'Porque la media será mucho más alta que el precio real de la mayoría de las casas debido al valor atípico.',
            'Que el 50% de los empleados tiene sueldos comprendidos entre 1500 € y 2500 €.',
            'No, la estadística descriptiva sólo describe el group analizado (Clase A); extenderlo a la Clase B sería inferencia.',
            'Ruta 2 tiene mayor desviación; es menos predecible y los datos están más lejos de la media.',
            'Un gráfico / diagrama de barras'
        ]
    }
}