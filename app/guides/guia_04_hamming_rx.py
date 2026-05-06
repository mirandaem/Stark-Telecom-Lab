import numpy as np
import pandas as pd
import streamlit as st


def validar_codigo_hamming(bits: str) -> bool:
    """
    Valida que la palabra recibida tenga exactamente 7 bits.
    """
    return len(bits) == 7 and all(bit in "01" for bit in bits)


def invertir_bit(bits: str, posicion: int) -> str:
    """
    Invierte el bit de una posición dada. La posición inicia en 1.
    """
    lista = list(bits)
    indice = posicion - 1
    lista[indice] = "1" if lista[indice] == "0" else "0"
    return "".join(lista)


def calcular_sindrome(bits: str) -> dict:
    """
    Calcula el síndrome para Hamming (7,4) con paridad par.

    Estructura:
    Posición: 1  2  3  4  5  6  7
    Tipo:     P1 P2 D1 P4 D2 D3 D4

    Verificaciones:
    s1 revisa posiciones 1, 3, 5, 7
    s2 revisa posiciones 2, 3, 6, 7
    s4 revisa posiciones 4, 5, 6, 7
    """
    b = [int(bit) for bit in bits]

    # Ajuste para trabajar con posiciones humanas:
    # b[0] -> posición 1
    # b[1] -> posición 2
    # ...
    # b[6] -> posición 7
    s1 = b[0] ^ b[2] ^ b[4] ^ b[6]
    s2 = b[1] ^ b[2] ^ b[5] ^ b[6]
    s4 = b[3] ^ b[4] ^ b[5] ^ b[6]

    posicion_error = s1 + (2 * s2) + (4 * s4)

    return {
        "s1": s1,
        "s2": s2,
        "s4": s4,
        "sindrome_binario": f"{s4}{s2}{s1}",
        "posicion_error": posicion_error,
    }


def corregir_hamming(bits: str) -> dict:
    """
    Corrige una palabra Hamming (7,4) si el síndrome indica error de un bit.
    """
    sindrome = calcular_sindrome(bits)
    posicion_error = sindrome["posicion_error"]

    if posicion_error == 0:
        corregido = bits
        estado = "Sin error detectable"
    else:
        corregido = invertir_bit(bits, posicion_error)
        estado = f"Error corregido en la posición {posicion_error}"

    datos_extraidos = corregido[2] + corregido[4] + corregido[5] + corregido[6]

    return {
        "recibido": bits,
        "sindrome": sindrome,
        "corregido": corregido,
        "datos_extraidos": datos_extraidos,
        "estado": estado,
    }


def construir_tabla_sindrome(bits: str, sindrome: dict) -> pd.DataFrame:
    """
    Construye tabla de verificaciones de paridad.
    """
    b = [int(bit) for bit in bits]

    filas = [
        {
            "Comprobación": "s1",
            "Posiciones revisadas": "1, 3, 5, 7",
            "Bits revisados": f"{b[0]}, {b[2]}, {b[4]}, {b[6]}",
            "Operación": f"{b[0]} ⊕ {b[2]} ⊕ {b[4]} ⊕ {b[6]}",
            "Resultado": sindrome["s1"],
        },
        {
            "Comprobación": "s2",
            "Posiciones revisadas": "2, 3, 6, 7",
            "Bits revisados": f"{b[1]}, {b[2]}, {b[5]}, {b[6]}",
            "Operación": f"{b[1]} ⊕ {b[2]} ⊕ {b[5]} ⊕ {b[6]}",
            "Resultado": sindrome["s2"],
        },
        {
            "Comprobación": "s4",
            "Posiciones revisadas": "4, 5, 6, 7",
            "Bits revisados": f"{b[3]}, {b[4]}, {b[5]}, {b[6]}",
            "Operación": f"{b[3]} ⊕ {b[4]} ⊕ {b[5]} ⊕ {b[6]}",
            "Resultado": sindrome["s4"],
        },
    ]

    return pd.DataFrame(filas)


def construir_tabla_comparacion(original: str, recibido: str, corregido: str) -> pd.DataFrame:
    """
    Construye tabla comparativa entre palabra original, recibida y corregida.
    """
    filas = []

    for i, (tx, rx, corr) in enumerate(zip(original, recibido, corregido), start=1):
        filas.append(
            {
                "Posición": i,
                "Transmitido": tx,
                "Recibido": rx,
                "Corregido": corr,
                "Cambio en canal": "Sí" if tx != rx else "No",
                "Cambio por corrección": "Sí" if rx != corr else "No",
            }
        )

    return pd.DataFrame(filas)


def extraer_datos_hamming(bits: str) -> str:
    """
    Extrae los datos D1 D2 D3 D4 de una palabra Hamming (7,4).
    """
    return bits[2] + bits[4] + bits[5] + bits[6]


def render_guia_04() -> None:
    st.title("Guía 4: Decodificación Hamming y corrección mediante síndrome")

    st.markdown(
        """
Esta guía estudia el proceso realizado por el receptor en un sistema que utiliza código
Hamming (7,4). El estudiante podrá inyectar errores, calcular el síndrome, localizar
la posición afectada y observar la corrección de errores de un solo bit.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Corrección de 1 bit",
            "Errores múltiples",
            "Análisis",
            "Dinámica",
            "Conclusiones",
            "Referencias",
        ]
    )

    with tabs[0]:
        st.header("Objetivos")

        st.markdown(
            """
**Objetivo general**

Comprender cómo el código Hamming (7,4) permite detectar y corregir errores de un solo
bit mediante el cálculo del síndrome en el receptor.

**Objetivos específicos**

1. Identificar la estructura de una palabra Hamming (7,4).
2. Calcular las comprobaciones de paridad en el receptor.
3. Construir el síndrome a partir de los bits recibidos.
4. Interpretar el síndrome como la posición del error.
5. Corregir un error de un solo bit.
6. Extraer los datos originales después de la corrección.
7. Analizar las limitaciones del código Hamming frente a errores múltiples.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En la Guía 3 se estudió cómo el transmisor agrega redundancia a un mensaje de 4 bits
para formar una palabra Hamming de 7 bits. En esta guía se analiza el proceso inverso:
el receptor toma la palabra recibida, revisa las paridades y determina si existe un error.

La estructura utilizada es:

| Posición | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Tipo | P1 | P2 | D1 | P4 | D2 | D3 | D4 |

El receptor calcula tres comprobaciones de paridad:

$$
s_1 = b_1 \\oplus b_3 \\oplus b_5 \\oplus b_7
$$

$$
s_2 = b_2 \\oplus b_3 \\oplus b_6 \\oplus b_7
$$

$$
s_4 = b_4 \\oplus b_5 \\oplus b_6 \\oplus b_7
$$

El síndrome se forma como:

$$
S = (s_4s_2s_1)_2
$$

Si el síndrome es cero, no se detecta error. Si el síndrome es distinto de cero, su valor
decimal indica la posición del bit que debe corregirse.

Por ejemplo:

$$
S = 101_2 = 5
$$

indica un error en la posición 5.

El código Hamming (7,4) permite corregir errores de un solo bit. Sin embargo, cuando
ocurren dos o más errores dentro de la misma palabra, el síndrome puede apuntar a una
posición incorrecta. Esta limitación justifica el uso posterior de CRC como mecanismo
de detección de errores remanentes.
"""
        )

    with tabs[2]:
        st.header("Corrección de un error de un solo bit")

        st.markdown(
            """
Ingrese una palabra Hamming válida de 7 bits. Luego seleccione una posición para
inyectar un error. La aplicación calculará el síndrome y realizará la corrección.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            palabra_tx = st.text_input(
                "Palabra Hamming transmitida",
                value="0110011",
                max_chars=7,
                help="Debe ingresar una palabra de 7 bits.",
                key="g4_palabra_tx",
            ).strip()

            posicion_error = st.number_input(
                "Posición del error a inyectar",
                min_value=1,
                max_value=7,
                value=5,
                step=1,
                key="g4_pos_error",
            )

            ejecutar = st.button("Inyectar error y corregir", use_container_width=True)

        with col_info:
            st.info(
                """
El síndrome permite localizar el error. Si ocurre un solo error, Hamming (7,4)
puede corregirlo invirtiendo el bit indicado por el síndrome.
"""
            )

        if not validar_codigo_hamming(palabra_tx):
            st.error("Debe ingresar exactamente 7 bits. Ejemplo válido: 0110011.")
        elif ejecutar:
            palabra_rx = invertir_bit(palabra_tx, posicion_error)
            resultado = corregir_hamming(palabra_rx)
            sindrome = resultado["sindrome"]

            tabla_sindrome = construir_tabla_sindrome(palabra_rx, sindrome)
            tabla_comparacion = construir_tabla_comparacion(
                palabra_tx,
                palabra_rx,
                resultado["corregido"],
            )

            st.subheader("Secuencias")

            st.code(
                f"Palabra transmitida: {palabra_tx}\n"
                f"Palabra recibida:    {palabra_rx}\n"
                f"Palabra corregida:   {resultado['corregido']}\n"
                f"Datos recuperados:   {resultado['datos_extraidos']}",
                language="text",
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Síndrome", sindrome["sindrome_binario"])
            c2.metric("Posición detectada", sindrome["posicion_error"])
            c3.metric("Estado", resultado["estado"])

            st.subheader("Tabla de comprobaciones de paridad")

            st.dataframe(tabla_sindrome, use_container_width=True, hide_index=True)

            st.subheader("Comparación transmitido, recibido y corregido")

            st.dataframe(tabla_comparacion, use_container_width=True, hide_index=True)

            if resultado["corregido"] == palabra_tx:
                st.success(
                    "La corrección fue exitosa. La palabra corregida coincide con la palabra transmitida."
                )
            else:
                st.error(
                    "La corrección no coincide con la palabra transmitida. Revise el procedimiento."
                )

            st.markdown(
                f"""
**Interpretación**

El síndrome calculado fue:

$$
S = ({sindrome["s4"]}{sindrome["s2"]}{sindrome["s1"]})_2 = {sindrome["posicion_error"]}
$$

Por tanto, el receptor identifica error en la posición {sindrome["posicion_error"]}.
"""
            )

            st.session_state["guia_04_ultimo_resultado"] = {
                "tx": palabra_tx,
                "rx": palabra_rx,
                "corregido": resultado["corregido"],
                "sindrome": sindrome["sindrome_binario"],
                "posicion": sindrome["posicion_error"],
                "datos": resultado["datos_extraidos"],
            }

    with tabs[3]:
        st.header("Errores múltiples y límite de Hamming")

        st.markdown(
            """
En esta sección se inyectan dos errores dentro de la misma palabra Hamming. El objetivo
es observar que el código Hamming (7,4) no garantiza corrección adecuada cuando ocurren
errores múltiples.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            palabra_multi = st.text_input(
                "Palabra Hamming transmitida",
                value="0110011",
                max_chars=7,
                key="g4_multi_palabra",
            ).strip()

            pos_1 = st.number_input(
                "Primera posición con error",
                min_value=1,
                max_value=7,
                value=2,
                step=1,
                key="g4_multi_pos1",
            )

            pos_2 = st.number_input(
                "Segunda posición con error",
                min_value=1,
                max_value=7,
                value=5,
                step=1,
                key="g4_multi_pos2",
            )

            ejecutar_multi = st.button("Inyectar dos errores", use_container_width=True)

        with col_info:
            st.warning(
                """
Hamming (7,4) está diseñado para corregir un solo error por palabra. Si ocurren dos
errores, el síndrome puede señalar una posición incorrecta y provocar una corrección
equivocada.
"""
            )

        if not validar_codigo_hamming(palabra_multi):
            st.error("Debe ingresar exactamente 7 bits.")
        elif pos_1 == pos_2:
            st.error("Seleccione dos posiciones distintas.")
        elif ejecutar_multi:
            palabra_rx = invertir_bit(palabra_multi, pos_1)
            palabra_rx = invertir_bit(palabra_rx, pos_2)

            resultado = corregir_hamming(palabra_rx)
            sindrome = resultado["sindrome"]

            tabla_sindrome = construir_tabla_sindrome(palabra_rx, sindrome)
            tabla_comparacion = construir_tabla_comparacion(
                palabra_multi,
                palabra_rx,
                resultado["corregido"],
            )

            st.subheader("Secuencias")

            st.code(
                f"Palabra transmitida: {palabra_multi}\n"
                f"Palabra recibida:    {palabra_rx}\n"
                f"Palabra corregida:   {resultado['corregido']}",
                language="text",
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Síndrome", sindrome["sindrome_binario"])
            c2.metric("Posición sugerida", sindrome["posicion_error"])
            c3.metric("Coincide con original", "Sí" if resultado["corregido"] == palabra_multi else "No")

            st.subheader("Tabla de comprobaciones")

            st.dataframe(tabla_sindrome, use_container_width=True, hide_index=True)

            st.subheader("Comparación")

            st.dataframe(tabla_comparacion, use_container_width=True, hide_index=True)

            if resultado["corregido"] == palabra_multi:
                st.success(
                    "En este caso particular la palabra coincide, pero no debe asumirse como garantía general."
                )
            else:
                st.error(
                    "La corrección no recuperó la palabra original. Esto demuestra el límite de Hamming ante errores múltiples."
                )

            st.markdown(
                """
**Conclusión del caso**

Cuando ocurren dos errores, el síndrome ya no representa necesariamente la posición real
de un único bit alterado. El receptor puede invertir un tercer bit y producir una palabra
incorrecta. Por esta razón, en sistemas más robustos se agrega un mecanismo de detección
adicional, como CRC.
"""
            )

    with tabs[4]:
        st.header("Análisis de resultados")

        st.markdown(
            """
Analice los resultados obtenidos:

1. ¿Qué representa el síndrome?
2. ¿Por qué el síndrome puede interpretarse como una posición?
3. ¿Qué ocurre cuando el síndrome es cero?
4. ¿Qué ocurre cuando se inyecta un solo error?
5. ¿Qué ocurre cuando se inyectan dos errores?
6. ¿Por qué este límite justifica el uso de CRC en una etapa posterior?
"""
        )

        if "guia_04_ultimo_resultado" in st.session_state:
            ultimo = st.session_state["guia_04_ultimo_resultado"]

            st.subheader("Último caso de corrección de un bit")

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Transmitido": ultimo["tx"],
                            "Recibido": ultimo["rx"],
                            "Corregido": ultimo["corregido"],
                            "Síndrome": ultimo["sindrome"],
                            "Posición detectada": ultimo["posicion"],
                            "Datos recuperados": ultimo["datos"],
                        }
                    ]
                )
            )
        else:
            st.info("Ejecute primero una corrección de un error en la pestaña correspondiente.")

    with tabs[5]:
        st.header("Dinámica de aprendizaje")

        st.markdown(
            """
Realice las siguientes actividades:

1. Use la palabra Hamming `0110011`.
2. Inyecte error en la posición 1 y registre el síndrome.
3. Repita para las posiciones 2, 3, 4, 5, 6 y 7.
4. Verifique que el síndrome coincide con la posición del error.
5. Inyecte dos errores y observe si el sistema logra recuperar la palabra original.
6. Explique por qué Hamming requiere apoyo de un detector más robusto para errores múltiples.
"""
        )

        pregunta_1 = st.radio(
            "Pregunta 1: ¿Qué indica un síndrome distinto de cero?",
            [
                "Que no existe error detectable.",
                "La posición del error de un solo bit.",
                "La cantidad total de bits de datos.",
                "La longitud del mensaje original.",
            ],
            index=None,
            key="g4_pregunta_1",
        )

        if pregunta_1:
            if pregunta_1 == "La posición del error de un solo bit.":
                st.success("Correcto. En Hamming (7,4), el síndrome indica la posición del error de un bit.")
            else:
                st.error("Revise la interpretación del síndrome.")

        pregunta_2 = st.radio(
            "Pregunta 2: ¿Qué ocurre si el síndrome es 000?",
            [
                "Se detecta error en la posición 7.",
                "No se detecta error.",
                "Siempre hay dos errores.",
                "Debe invertirse el primer bit.",
            ],
            index=None,
            key="g4_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "No se detecta error.":
                st.success("Correcto. Un síndrome cero indica que no se detecta error bajo las verificaciones de paridad.")
            else:
                st.error("Revise el significado del síndrome cero.")

        pregunta_3 = st.radio(
            "Pregunta 3: ¿Cuál es la principal limitación de Hamming (7,4)?",
            [
                "No puede transmitir bits.",
                "No puede corregir de forma confiable errores múltiples.",
                "No utiliza bits de paridad.",
                "No permite construir palabras código.",
            ],
            index=None,
            key="g4_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "No puede corregir de forma confiable errores múltiples.":
                st.success("Correcto. Hamming (7,4) corrige un error de un bit, pero falla ante errores múltiples.")
            else:
                st.error("Revise la capacidad de corrección del código Hamming.")

    with tabs[6]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- el receptor calcula un síndrome a partir de verificaciones de paridad;
- el síndrome permite localizar errores de un solo bit;
- la corrección se realiza invirtiendo el bit indicado por el síndrome;
- Hamming (7,4) permite recuperar los 4 bits de datos si ocurre un solo error;
- cuando ocurren errores múltiples, el síndrome puede inducir una corrección incorrecta;
- esta limitación justifica la integración posterior de CRC como detector de errores remanentes.
"""
        )

    with tabs[7]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2012). *Data communications and networking* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Lin, S., & Costello, D. J. (1983). *Error control coding: Fundamentals and applications*. Prentice-Hall.
"""
        )