
document.addEventListener("DOMContentLoaded", () => {

    // ELEMENTOS
    const inputImagen = document.getElementById("imagen");
    const botonAnalizar = document.getElementById("botonAnalizar");
    const uploadArea = document.getElementById("uploadArea");
    const uploadContent = document.getElementById("uploadContent");
    const previewImagen = document.getElementById("previewImagen");

    const resultadoVacio = document.getElementById("resultadoVacio");
    const resultadoContenido = document.getElementById("resultadoContenido");

    const resultadoFruta = document.getElementById("resultadoFruta");
    const confianzaFruta = document.getElementById("confianzaFruta");
    const resultadoEstado = document.getElementById("resultadoEstado");
    const confianzaEstado = document.getElementById("confianzaEstado");
    const estadoIndicador = document.getElementById("estadoIndicador");

    let imagenSeleccionada = null;

    // SELECCIONAR IMAGEN
    inputImagen.addEventListener("change", () => {
        if (inputImagen.files.length > 0) {
            procesarImagen(inputImagen.files[0]);
        }
    });

    function procesarImagen(archivo) {

        if (!archivo.type.startsWith("image/")) {
            alert("Selecciona un archivo de imagen válido.");
            return;
        }

        imagenSeleccionada = archivo;

        const lector = new FileReader();

        lector.onload = (evento) => {
            previewImagen.src = evento.target.result;
            previewImagen.hidden = false;
            uploadContent.hidden = true;
        };

        lector.readAsDataURL(archivo);

        resultadoContenido.hidden = true;
        resultadoVacio.hidden = false;
    }

    // ARRASTRAR Y SOLTAR
    uploadArea.addEventListener("dragover", (evento) => {
        evento.preventDefault();
        uploadArea.classList.add("drag-active");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("drag-active");
    });

    uploadArea.addEventListener("drop", (evento) => {
        evento.preventDefault();
        uploadArea.classList.remove("drag-active");

        const archivo = evento.dataTransfer.files[0];

        if (archivo) {
            procesarImagen(archivo);
        }
    });

    // ANALIZAR IMAGEN
    botonAnalizar.addEventListener("click", async () => {

        if (!imagenSeleccionada) {
            alert("Primero selecciona una fotografía.");
            return;
        }

        const datos = new FormData();
        datos.append("imagen", imagenSeleccionada);

        botonAnalizar.disabled = true;
        botonAnalizar.innerHTML = "<span>Analizando imagen...</span><span>⏳</span>";

        try {

            const respuesta = await fetch("/analizar", {
                method: "POST",
                body: datos
            });

            const resultado = await respuesta.json();

            if (!respuesta.ok) {
                throw new Error(resultado.error || "No se pudo analizar la imagen.");
            }

            // MOSTRAR IDENTIFICACIÓN
            resultadoFruta.textContent = resultado.fruta;
            confianzaFruta.textContent =
                "Confianza del modelo: " + resultado.confianza_fruta + "%";

            // MOSTRAR ESTADO
            resultadoEstado.textContent = resultado.estado;
            confianzaEstado.textContent =
                "Confianza del modelo: " + resultado.confianza_estado + "%";

            // COLOR SEGÚN ESTADO
            const estado = resultado.estado.toLowerCase();

            if (estado.includes("recien")) {
                estadoIndicador.style.backgroundColor = "#36a66b";
            } else if (estado.includes("leve")) {
                estadoIndicador.style.backgroundColor = "#e5ad36";
            } else if (estado.includes("podrido")) {
                estadoIndicador.style.backgroundColor = "#dc5555";
            } else {
                estadoIndicador.style.backgroundColor = "#89938b";
            }

            resultadoVacio.hidden = true;
            resultadoContenido.hidden = false;

        } catch (error) {

            alert("Error al analizar: " + error.message);

        } finally {

            botonAnalizar.disabled = false;
            botonAnalizar.innerHTML = "<span>Analizar imagen</span><span>→</span>";

        }
    });

});
