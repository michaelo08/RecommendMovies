
from flask import Flask, render_template, request, jsonify,send_file
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Datos de ejemplo
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

# Creación del modelo de recomendación (simulado)
num_users, num_movies = data.shape
embedding_size = 5

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

# Ruta para el inicio
@app.route('/')
def index():
    return send_file('./templates/indes.html')

# Ruta para obtener recomendaciones
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    user_id = int(request.form['userId'])  # ID de usuario del frontend

    # Simulación de recomendaciones
    movie_ids = np.array([0, 1, 2, 3])
    predicted_ratings = model.predict([np.array([user_id] * len(movie_ids)), movie_ids])

    recommendations = [f"{peliculas[i]} - Rating: {rating[0]:.2f}" for i, rating in enumerate(predicted_ratings)]
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
