
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

from fruitscan import analizar_fruta

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def archivo_permitido(nombre):
    return (
        "." in nombre
        and nombre.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/analizar", methods=["POST"])
def analizar():

    if "imagen" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    archivo = request.files["imagen"]

    if archivo.filename == "":
        return jsonify({"error": "No seleccionaste una imagen"}), 400

    if not archivo_permitido(archivo.filename):
        return jsonify({"error": "Formato de imagen no permitido"}), 400

    nombre = secure_filename(archivo.filename)
    ruta = os.path.join(app.config["UPLOAD_FOLDER"], nombre)

    archivo.save(ruta)

    try:
        resultado = analizar_fruta(ruta, mostrar=False)
        return jsonify(resultado)

    except Exception as error:
        return jsonify({"error": str(error)}), 500

    finally:
        if os.path.exists(ruta):
            os.remove(ruta)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
