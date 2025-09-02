Propagacion de una Enfermedad Epidemica
Utiliza los modelos matematicos SIR (Susceptible, Infectado, Recuperado) y SEIR (Susceptible, Expuesto, Infectado, Recuperado)

Los parámetros iniciales configurados son:
  EXCEL_PATH: La ruta al archivo de Excel (p1.xlsx) que contiene los datos de casos.
  SMOOTH_WINDOW: El tamaño de la ventana para suavizar los datos, lo que ayuda a eliminar el "ruido" de las fluctuaciones diarias.
  N: La población total de Bolivia, que se usa como base para los modelos.

normalize_potosi_cols: 
  Estandariza los nombres de columnas que contienen la palabra "Potosí" para asegurar que los datos de ese departamento sean procesados correctamente.
sheet_to_national_series: 
  Lee una hoja de un archivo Excel, convierte la columna de fechas a un formato adecuado y suma los casos de todos los departamentos para obtener una serie de tiempo de casos a nivel nacional.
detect_and_convert_cumulative: 
  Es una función crucial que determina si los datos de casos son acumulativos (el total de casos hasta una fecha) o de incidencia diaria (casos nuevos por día). Si son acumulativos, los convierte a incidencia diaria para que sean compatibles con los modelos SIR/SEIR.
load_bolivia_from_all_sheets: 
  Orquesta el proceso de carga, leyendo todas las hojas del archivo de Excel, combinando los datos y aplicando la detección y conversión de datos acumulativos.


Los modelos se definen como un sistema de ecuaciones diferenciales ordinarias (ode).
  sir_ode: 
    Define el modelo SIR con tres compartimentos: S (población susceptible), I (población infecciosa) y R (población recuperada/retirada).
  seir_ode: 
    Define el modelo SEIR, que es una extensión del SIR que añade un compartimento E (población expuesta). Este compartimento representa a las personas que han sido infectadas pero aún no son infecciosas (período de incubación).
  simulate_sir y simulate_seir: 
    Estas funciones resuelven las ecuaciones de los modelos para simular la trayectoria de la epidemia a lo largo del tiempo, dada una serie de parámetros iniciales.


Las simulaciones de SIR y SEIR se ajusten a los datos reales de la siguiente manera
  gamma_kernel: 
    Utiliza una distribución de probabilidad de tipo gamma para modelar el retraso entre la fecha de infección y la fecha en que se confirma el caso.
  apply_gamma_delay: 
    Aplica la función de retraso a la simulación del modelo.
  fit_sir_window y fit_seir_window: 
    Utilizan la función curve_fit de SciPy para encontrar los parámetros que mejor "encajan" con los datos. Estos parámetros incluyen la tasa de contagio (beta), el tamaño de la población efectiva (fN) y el retraso (tau_days), etc.



  
