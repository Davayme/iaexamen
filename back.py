from flask import Flask, request, jsonify
from acrcloud.recognizer import ACRCloudRecognizer
import os

app = Flask(__name__)

# Configuración de ACRCloud
config = {
    'host': 'identify-us-west-2.acrcloud.com',  # Reemplaza con tu host de ACRCloud
    'access_key': 'ccdee699a8f78f3d4bb84848a59a50d9',  # Reemplaza con tu access_key
    'access_secret': 'kCYBGBxi342lA5FvcG85deUn0Y0py0OwLkTzCwF6',  # Reemplaza con tu access_secret
    'timeout': 10  # Tiempo de espera en segundos
}

# Inicializar el reconocedor de ACRCloud
recognizer = ACRCloudRecognizer(config)

@app.route('/identify-song', methods=['POST'])
def identify_song():
    # Verificar si se envió un archivo de audio
    if 'audio' not in request.files:
        return jsonify({'error': 'No se proporcionó un archivo de audio'}), 400

    audio_file = request.files['audio']

    # Guardar el archivo temporalmente
    temp_file_path = 'temp_audio.wav'
    audio_file.save(temp_file_path)

    # Identificar la canción usando ACRCloud
    try:
        result = recognizer.recognize_by_file(temp_file_path, 0, 3)  # Usar los primeros 3 segundos
        os.remove(temp_file_path)  # Eliminar el archivo temporal

        # Parsear el resultado
        if result and 'metadata' in result and 'music' in result['metadata']:
            song_info = result['metadata']['music'][0]
            return jsonify({
                'song': song_info.get('title', 'Desconocido'),
                'artist': song_info.get('artist', 'Desconocido')
            })
        else:
            return jsonify({'error': 'No se pudo identificar la canción'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)