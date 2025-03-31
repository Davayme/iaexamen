import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ejemplo de descripciones de películas
peliculas = [
    "Un héroe lucha contra un villano para salvar al mundo",
    "Un grupo de amigos emprende una aventura épica",
    "Una comedia romántica llena de situaciones divertidas",
    "Un thriller lleno de giros inesperados"
]

# Vectorización y cálculo de similitud
vectorizador = TfidfVectorizer()
matriz_tfidf = vectorizador.fit_transform(peliculas)
similitud = cosine_similarity(matriz_tfidf)

# Mostrar matriz de similitud
print("=== Búsqueda por Similitud Semántica ===")
print("Matriz de similitud entre películas:")
print(similitud)

# Encontrar la película más similar a la primera
mas_similar = similitud[0].argsort()[-2]
print(f"\nLa película más similar a '{peliculas[0]}' es: '{peliculas[mas_similar]}'")