import random
from typing import List, Dict

import streamlit as st


def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def transmitir_bits(bits: str, prob_error: float = 0.1, semilla: int | None = None) -> tuple[str, int]:
    """
    Simula la transmisión de una secuencia binaria.
    Cada bit puede invertirse con una probabilidad dada.
    """
    rng = random.Random(semilla)

    recibido = []
    errores = 0

    for bit in bits:
        if rng.random() < prob_error:
            recibido.append("1" if bit == "0" else "0")
            errores += 1
        else:
            recibido.append(bit)

    return "".join(recibido), errores


def calcular_ber(original: str, recibido: str) -> tuple[int, float]:
    if len(original) != len(recibido):
        raise ValueError("Las secuencias deben tener la misma longitud.")

    errores = sum(1 for a, b in zip(original, recibido) if a != b)
    ber = errores / len(original) if original else 0.0
    return errores, ber


def comparar_bits(original: str, recibido: str) -> List[Dict[str, str]]:
    filas = []
    for i, (tx, rx) in enumerate(zip(original, recibido), start=1):
        filas.append(
            {
                "Posición": i,
                "Tx": tx,
                "Rx": rx,
                "Estado": "OK" if tx == rx else "ERROR",
            }
        )
    return filas


st.set_page_config(page_title="Stark Telecom Lab - Guía 1", layout="wide")

st.title("Stark Telecom Lab")
st.subheader("Guía 1: Fundamentos de transmisión digital y errores en el canal")

st.markdown(
    """
Esta interfaz permite simular una transmisión binaria básica, introducir errores en el canal
y medir el desempeño mediante la tasa de error de bit (BER).

Modelo conceptual:
- Transmisor: genera la secuencia binaria
- Canal: introduce errores con una probabilidad definida
- Receptor: compara la secuencia recibida con la original
"""
)

with st.sidebar:
    st.header("Parámetros de simulación")

    mensaje = st.text_input(
        "Secuencia binaria transmitida",
        value="1011001",
        help="Ingrese únicamente ceros y unos.",
    ).strip()

    prob_error = st.slider(
        "Probabilidad de error del canal",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Probabilidad de que un bit cambie durante la transmisión.",
    )

    usar_semilla = st.checkbox("Usar semilla fija", value=True)

    semilla = None
    if usar_semilla:
        semilla = st.number_input(
            "Semilla",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            help="Permite repetir exactamente el mismo experimento.",
        )

    ejecutar = st.button("Ejecutar simulación", use_container_width=True)

if not validar_bits(mensaje):
    st.error("La secuencia ingresada no es válida. Use únicamente caracteres 0 y 1.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Información de entrada")
    st.write(f"**Mensaje transmitido:** `{mensaje}`")
    st.write(f"**Longitud:** {len(mensaje)} bits")
    st.write(f"**Probabilidad de error:** {prob_error:.2f}")
    if usar_semilla:
        st.write(f"**Semilla:** {semilla}")

with col2:
    st.markdown("### Objetivo didáctico")
    st.info(
        "Observe cómo cambia la secuencia recibida al variar la probabilidad de error "
        "y cómo el BER cuantifica el efecto del canal."
    )

if ejecutar:
    recibido, errores_canal = transmitir_bits(mensaje, prob_error=prob_error, semilla=semilla)
    errores, ber = calcular_ber(mensaje, recibido)
    tabla = comparar_bits(mensaje, recibido)

    st.markdown("---")
    st.markdown("## Resultados de la simulación")

    r1, r2, r3 = st.columns(3)
    r1.metric("Bits transmitidos", len(mensaje))
    r2.metric("Bits erróneos", errores)
    r3.metric("BER", f"{ber:.4f}")

    st.markdown("### Secuencias")
    st.code(
        f"Transmitido: {mensaje}\n"
        f"Recibido:    {recibido}",
        language="text",
    )

    st.markdown("### Comparación bit a bit")
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.markdown("### Interpretación")
    if errores == 0:
        st.success(
            "No se observaron errores en esta transmisión. "
            "En este caso, el BER es igual a 0."
        )
    else:
        st.warning(
            f"Se detectaron {errores} errores en la secuencia recibida. "
            "Esto muestra que el canal puede alterar la información transmitida."
        )

    st.markdown(
        f"""
**Cálculo del BER**

BER = errores / bits transmitidos = **{errores} / {len(mensaje)} = {ber:.4f}**
"""
    )

    with st.expander("Preguntas de análisis sugeridas"):
        st.markdown(
            """
1. ¿Qué ocurre con el BER cuando aumenta la probabilidad de error?
2. ¿Puede el receptor recuperar el mensaje original si solo observa la secuencia recibida?
3. ¿Por qué esta situación justifica el uso de técnicas de detección y corrección de errores?
"""
        )
else:
    st.markdown("---")
    st.markdown("## Instrucciones")
    st.markdown(
        """
1. Ingrese una secuencia binaria.
2. Ajuste la probabilidad de error del canal.
3. Presione **Ejecutar simulación**.
4. Analice la secuencia recibida y el valor del BER.
"""
    )