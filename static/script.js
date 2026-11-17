document.addEventListener("DOMContentLoaded", function () {

    const inputImagen = document.getElementById("imagen");
    const botonAnalizar = document.getElementById("botonAnalizar");

    console.log("FruitScan AI: JavaScript conectado");

    if (!inputImagen || !botonAnalizar) {
        console.error("No se encontró el input o el botón.");
        return;
    }

    botonAnalizar.addEventListener("click", async function () {

        const archivo = inputImagen.files[0];

        if (!archivo) {
            alert("Primero selecciona una imagen.");
            return;
        }

        const datos = new FormData();
        datos.append("imagen", archivo);

        botonAnalizar.textContent = "Analizando...";
        botonAnalizar.disabled = true;

        try {
            const respuesta = await fetch("/analizar", {
                method: "POST",
                body: datos
            });

            const resultado = await respuesta.json();

            if (!respuesta.ok) {
                throw new Error(resultado.error || "Error al analizar");
            }

            alert(
                "FRUITSCAN AI\n\n" +
                "Alimento: " + resultado.fruta + "\n" +
                "Estado visual: " + resultado.estado
            );

        } catch (error) {
            alert("Error: " + error.message);

        } finally {
            botonAnalizar.textContent = "Analizar imagen";
            botonAnalizar.disabled = false;
        }
    });
});