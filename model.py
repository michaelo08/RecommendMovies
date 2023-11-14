# # Datos de ejemplo: Usuarios, películas y calificaciones
# data = {
#     'Usuario1': {'Pelicula1': 5, 'Pelicula2': 3, 'Pelicula3': 4, 'Pelicula4': 1},
#     'Usuario2': {'Pelicula1': 3, 'Pelicula2': 5, 'Pelicula3': 4, 'Pelicula4': 2},
#     'Usuario3': {'Pelicula1': 4, 'Pelicula2': 2, 'Pelicula3': 1, 'Pelicula4': 5},
#     'Usuario4': {'Pelicula1': 1, 'Pelicula2': 5, 'Pelicula3': 4, 'Pelicula4': 3},
#     'Usuario5': {'Pelicula2': 5, 'Pelicula3': 4, 'Pelicula4': 2},
# }

# # Mapeo de ID de película a nombre de película
# peliculas = {
#     'Pelicula1': 'El Gran Escape',
#     'Pelicula2': 'La Aventura Comienza',
#     'Pelicula3': 'El Misterio del Pasado',
#     'Pelicula4': 'Amanecer en la Ciudad',
# }

# # ... (código previo)
# # Función para obtener recomendaciones basadas en filtrado colaborativo
# def recomendacion_usuario(data, usuario):
#     puntuaciones_usuario = data[usuario]
#     similitudes = {}
#     total = {}

#     for otro_usuario in data:
#         if otro_usuario != usuario:
#             sim = 0
#             tiene_peliculas_en_comun = False
#             for pelicula in data[otro_usuario]:
#                 if pelicula in puntuaciones_usuario:
#                     sim += (puntuaciones_usuario[pelicula] - data[otro_usuario][pelicula])**2
#                     tiene_peliculas_en_comun = True
#             if tiene_peliculas_en_comun:
#                 similitudes[otro_usuario] = 1 / (1 + sim)
#                 for pelicula in data[otro_usuario]:
#                     if pelicula not in puntuaciones_usuario or puntuaciones_usuario[pelicula] == 0:
#                         total.setdefault(pelicula, 0)
#                         total[pelicula] += data[otro_usuario][pelicula] * similitudes[otro_usuario]

#     peliculas_recomendadas = []
#     for pelicula, similitud_total in total.items():
#         peliculas_recomendadas.append((pelicula, similitud_total))

#     peliculas_recomendadas.sort(key=lambda x: x[1], reverse=True)

#     # Convertir IDs de películas a nombres de películas en las recomendaciones
#     recomendaciones_con_nombre = [(peliculas[peli_id], similitud) for peli_id, similitud in peliculas_recomendadas]

#     return recomendaciones_con_nombre

# # Obtener recomendaciones para un usuario específico (por ejemplo, 'Usuario5')
# usuario_a_recomendar = 'Usuario5'
# recomendaciones = recomendacion_usuario(data, usuario_a_recomendar)

# print(f"Las recomendaciones para {usuario_a_recomendar} son:")
# for recomendacion in recomendaciones:
#     print(f"Película: {recomendacion[0]} - Similitud: {recomendacion[1]}")


# Codigo 2

# import numpy as np

# # Datos de ejemplo: Usuarios, películas y calificaciones
# data = np.array([
#     [5, 3, 4, 1],
#     [3, 5, 4, 2],
#     [4, 2, 1, 5],
#     [1, 5, 4, 3],
#     [0, 5, 4, 2],  # Usuario5 con datos incompletos
# ])

# # Mapeo de ID de película a nombre de película
# peliculas = {
#     0: 'El Gran Escape',
#     1: 'La Aventura Comienza',
#     2: 'El Misterio del Pasado',
#     3: 'Amanecer en la Ciudad',
# }

# # Función para predecir calificaciones usando SVD
# def svd_recomendacion(data, usuario, peliculas):
#     U, sigma, Vt = np.linalg.svd(data, full_matrices=False)
#     sigma = np.diag(sigma)
    
#     # Reducción de dimensionalidad para hacer las predicciones
#     dimension = 3  # Número de dimensiones a considerar
#     Usk = U[:, :dimension]
#     sigmak = sigma[:dimension, :dimension]
#     Vtk = Vt[:dimension, :]
    
#     # Predicción para el usuario dado
#     usuario_prediccion = np.dot(np.dot(Usk[usuario], sigmak), Vtk)
#     return usuario_prediccion

# # Obtener recomendaciones para un usuario específico (por ejemplo, 'Usuario4')
# usuario_a_recomendar = 3  # Índice del usuario en los datos
# recomendacion_usuario = svd_recomendacion(data, usuario_a_recomendar, peliculas)

# print(f"Las recomendaciones para el usuario {usuario_a_recomendar} son:")
# for i, calificacion in enumerate(recomendacion_usuario):
#     nombre_pelicula = peliculas[i]
#     print(f"Película: {nombre_pelicula} - Calificación: {calificacion:.2f}")

#Codigo 3 
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Datos de ejemplo: Usuarios, películas y calificaciones
data = np.array([
    [5, 4, 3, 2],
    [4, 5, 3, 2],
    [3, 4, 5, 2],
    [2, 3, 4, 5],
    [1, 2, 3, 4],
])
# Mapeo de ID de película a nombre de película
peliculas = {
    0: 'El Gran Escape',
    1: 'La Aventura Comienza',
    2: 'El Misterio del Pasado',
    3: 'Amanecer en la Ciudad',
}

# Creación del modelo de recomendación
num_users, num_movies = data.shape
embedding_size = 5  # Tamaño de los vectores de embedding

user_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
user_embedding = Flatten()(user_embedding)

movie_input = Input(shape=(1,))
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size)(movie_input)
movie_embedding = Flatten()(movie_embedding)

dot_product = Dot(axes=1)([user_embedding, movie_embedding])

dense_1 = Dense(10, activation='relu')(dot_product)
output = Dense(1)(dense_1)

model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# Creación de datos para entrenamiento
users = np.array([[i] * num_movies for i in range(num_users)]).flatten()
movies = np.array([i for i in range(num_movies)] * num_users)
ratings = data.flatten()

model.fit([users, movies], ratings, epochs=10, batch_size=1)

# Predicción para un usuario específico
usuario_a_recomendar = 3
movie_ids = np.array([0, 1, 2, 3])
predicted_ratings = model.predict([np.array([usuario_a_recomendar] * len(movie_ids)), movie_ids])

print(f"Las recomendaciones para el usuario {usuario_a_recomendar} son:")
for i, rating in enumerate(predicted_ratings):
    movie_name = peliculas[i]  # Obtener el nombre de la película desde el diccionario
    print(f"Película: {movie_name} - Rating predicho = {rating[0]:.2f}")
