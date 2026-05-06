import numpy as np
import pandas as pd
import streamlit as st


def validar_bits_4(bits: str) -> bool:
    """
    Valida que la entrada sea una secuencia binaria de exactamente 4 bits.
    """
    return len(bits) == 4 and all(bit in "01" for bit in bits)


def calcular_paridades_hamming_7_4(bits: str) -> dict:
    """
    Calcula los bits de paridad para Hamming (7,4) con paridad par.

    Estructura usada:
    Posición: 1  2  3  4  5  6  7
    Tipo:     P1 P2 D1 P4 D2 D3 D4

    Los bits de datos se colocan así:
    D1 -> posición 3
    D2 -> posición 5
    D3 -> posición 6
    D4 -> posición 7
    """
    d1, d2, d3, d4 = [int(bit) for bit in bits]

    # Paridad par:
    # P1 cubre posiciones 1, 3, 5, 7
    # P2 cubre posiciones 2, 3, 6, 7
    # P4 cubre posiciones 4, 5, 6, 7
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4

    codigo = [p1, p2, d1, p4, d2, d3, d4]

    return {
        "P1": p1,
        "P2": p2,
        "P4": p4,
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "D4": d4,
        "codigo": codigo,
        "codigo_str": "".join(str(bit) for bit in codigo),
    }


def construir_tabla_posiciones(resultado: dict) -> pd.DataFrame:
    """
    Construye una tabla con las posiciones del código Hamming (7,4).
    """
    return pd.DataFrame(
        {
            "Posición": [1, 2, 3, 4, 5, 6, 7],
            "Tipo": ["P1", "P2", "D1", "P4", "D2", "D3", "D4"],
            "Valor": [
                resultado["P1"],
                resultado["P2"],
                resultado["D1"],
                resultado["P4"],
                resultado["D2"],
                resultado["D3"],
                resultado["D4"],
            ],
            "Descripción": [
                "Paridad que cubre posiciones 1, 3, 5 y 7",
                "Paridad que cubre posiciones 2, 3, 6 y 7",
                "Bit de dato 1",
                "Paridad que cubre posiciones 4, 5, 6 y 7",
                "Bit de dato 2",
                "Bit de dato 3",
                "Bit de dato 4",
            ],
        }
    )


def construir_tabla_cobertura(resultado: dict) -> pd.DataFrame:
    """
    Muestra qué posiciones participan en el cálculo de cada paridad.
    """
    return pd.DataFrame(
        {
            "Bit de paridad": ["P1", "P2", "P4"],
            "Posiciones cubiertas": ["1, 3, 5, 7", "2, 3, 6, 7", "4, 5, 6, 7"],
            "Bits usados para calcular": ["D1, D2, D4", "D1, D3, D4", "D2, D3, D4"],
            "Operación XOR": [
                f"{resultado['D1']} ⊕ {resultado['D2']} ⊕ {resultado['D4']}",
                f"{resultado['D1']} ⊕ {resultado['D3']} ⊕ {resultado['D4']}",
                f"{resultado['D2']} ⊕ {resultado['D3']} ⊕ {resultado['D4']}",
            ],
            "Resultado": [resultado["P1"], resultado["P2"], resultado["P4"]],
        }
    )


def matriz_generadora_hamming_7_4() -> np.ndarray:
    """
    Matriz generadora correspondiente a la estructura:
    [P1 P2 D1 P4 D2 D3 D4]

    Para m = [D1 D2 D3 D4], la palabra código es:
    c = m · G mod 2
    """
    return np.array(
        [
            [1, 1, 1, 0, 0, 0, 0],  # D1 contribuye a P1, P2 y posición D1
            [1, 0, 0, 1, 1, 0, 0],  # D2 contribuye a P1, P4 y posición D2
            [0, 1, 0, 1, 0, 1, 0],  # D3 contribuye a P2, P4 y posición D3
            [1, 1, 0, 1, 0, 0, 1],  # D4 contribuye a P1, P2, P4 y posición D4
        ],
        dtype=int,
    )


def codificar_con_matriz(bits: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Codifica el mensaje usando matriz generadora.
    """
    m = np.array([int(bit) for bit in bits], dtype=int)
    G = matriz_generadora_hamming_7_4()
    c = (m @ G) % 2
    return m, G, c


def dataframe_matriz(G: np.ndarray) -> pd.DataFrame:
    """
    Convierte la matriz generadora en DataFrame para mostrarla.
    """
    return pd.DataFrame(
        G,
        index=["D1", "D2", "D3", "D4"],
        columns=["P1", "P2", "D1", "P4", "D2", "D3", "D4"],
    )


def render_guia_03() -> None:
    st.title("Guía 3: Codificación Hamming (7,4) en el transmisor")

    st.markdown(
        """
Esta guía introduce el código Hamming (7,4) como una técnica de corrección de errores
hacia adelante. El objetivo es comprender cómo el transmisor agrega bits de paridad a
un mensaje de 4 bits para formar una palabra código de 7 bits.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Codificación por posiciones",
            "Método matricial",
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

Comprender cómo el código Hamming (7,4) agrega redundancia estructurada en el transmisor
para proteger una secuencia binaria ante errores de transmisión.

**Objetivos específicos**

1. Identificar la relación entre bits de datos, bits de paridad y palabra código.
2. Reconocer la estructura de posiciones del código Hamming (7,4).
3. Calcular los bits de paridad mediante operaciones XOR.
4. Construir la palabra código de 7 bits.
5. Comparar el método por posiciones con el método matricial.
6. Relacionar la redundancia agregada con la futura capacidad de corrección en el receptor.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En las guías anteriores se observó que el ruido puede alterar la señal recibida y producir
errores de bit. Para enfrentar este problema, los sistemas de comunicación incorporan
redundancia, es decir, información adicional calculada a partir de los datos originales.

El código Hamming es una técnica de corrección de errores hacia adelante. Esto significa
que el transmisor agrega bits de control antes de enviar la información, de modo que el
receptor pueda detectar y corregir ciertos errores sin solicitar retransmisión.

En el código Hamming (7,4):

- se tienen $k = 4$ bits de datos;
- se agregan $r = 3$ bits de paridad;
- se obtiene una palabra código de $n = 7$ bits.

La relación general entre datos, paridad y longitud total es:

$$
n = k + r
$$

Para determinar cuántos bits de paridad se necesitan, se utiliza la condición:

$$
2^r \\geq k + r + 1
$$

Para $k = 4$:

$$
2^3 = 8 \\geq 4 + 3 + 1 = 8
$$

Por tanto, se requieren tres bits de paridad.

En Hamming (7,4), los bits de paridad se colocan en posiciones que son potencias de dos:

$$
1, 2, 4
$$

Los bits de datos se colocan en las posiciones restantes:

$$
3, 5, 6, 7
$$

La estructura utilizada en esta guía es:

| Posición | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Tipo | P1 | P2 | D1 | P4 | D2 | D3 | D4 |

El código generado en esta guía usa paridad par.
"""
        )

    with tabs[2]:
        st.header("Codificación por posiciones")

        st.markdown(
            """
Ingrese un mensaje de 4 bits. La aplicación colocará los bits de datos en las posiciones
correspondientes y calculará los bits de paridad.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            bits = st.text_input(
                "Mensaje de 4 bits",
                value="1011",
                max_chars=4,
                help="Ingrese exactamente 4 bits.",
                key="g3_bits_posiciones",
            ).strip()

            ejecutar = st.button("Codificar con Hamming (7,4)", use_container_width=True)

        with col_info:
            st.info(
                """
La estructura usada es:

Posición 1: P1  
Posición 2: P2  
Posición 3: D1  
Posición 4: P4  
Posición 5: D2  
Posición 6: D3  
Posición 7: D4
"""
            )

        if not validar_bits_4(bits):
            st.error("Debe ingresar exactamente 4 bits. Ejemplo válido: 1011.")
        elif ejecutar:
            resultado = calcular_paridades_hamming_7_4(bits)
            tabla_posiciones = construir_tabla_posiciones(resultado)
            tabla_cobertura = construir_tabla_cobertura(resultado)

            st.subheader("Resultado de la codificación")

            c1, c2, c3 = st.columns(3)
            c1.metric("Bits de datos k", "4")
            c2.metric("Bits de paridad r", "3")
            c3.metric("Longitud total n", "7")

            st.code(
                f"Mensaje original: {bits}\n"
                f"Palabra Hamming:  {resultado['codigo_str']}",
                language="text",
            )

            st.subheader("Tabla de posiciones")

            st.dataframe(tabla_posiciones, use_container_width=True, hide_index=True)

            st.subheader("Cálculo de paridades")

            st.dataframe(tabla_cobertura, use_container_width=True, hide_index=True)

            st.markdown(
                f"""
**Cálculos realizados con paridad par**

$$
P1 = D1 \\oplus D2 \\oplus D4 = {resultado['D1']} \\oplus {resultado['D2']} \\oplus {resultado['D4']} = {resultado['P1']}
$$

$$
P2 = D1 \\oplus D3 \\oplus D4 = {resultado['D1']} \\oplus {resultado['D3']} \\oplus {resultado['D4']} = {resultado['P2']}
$$

$$
P4 = D2 \\oplus D3 \\oplus D4 = {resultado['D2']} \\oplus {resultado['D3']} \\oplus {resultado['D4']} = {resultado['P4']}
$$
"""
            )

            st.session_state["guia_03_bits"] = bits
            st.session_state["guia_03_codigo"] = resultado["codigo_str"]

    with tabs[3]:
        st.header("Método matricial")

        st.markdown(
            """
El mismo proceso de codificación puede expresarse mediante una matriz generadora. Si el
mensaje se representa como:

$$
m = [D1 \\quad D2 \\quad D3 \\quad D4]
$$

entonces la palabra código se obtiene como:

$$
c = mG \\mod 2
$$

donde $G$ es la matriz generadora del código Hamming (7,4).
"""
        )

        bits_matriz = st.text_input(
            "Mensaje de 4 bits para método matricial",
            value="1011",
            max_chars=4,
            key="g3_bits_matriz",
        ).strip()

        if not validar_bits_4(bits_matriz):
            st.error("Debe ingresar exactamente 4 bits.")
        else:
            m, G, c = codificar_con_matriz(bits_matriz)
            df_G = dataframe_matriz(G)

            st.subheader("Vector de datos")

            st.code(
                f"m = {m.tolist()}",
                language="text",
            )

            st.subheader("Matriz generadora G")

            st.dataframe(df_G, use_container_width=True)

            st.subheader("Resultado de la multiplicación módulo 2")

            st.code(
                f"c = m · G mod 2 = {c.tolist()}\n"
                f"Palabra código = {''.join(str(bit) for bit in c)}",
                language="text",
            )

            resultado_posiciones = calcular_paridades_hamming_7_4(bits_matriz)["codigo_str"]
            resultado_matriz = "".join(str(bit) for bit in c)

            if resultado_posiciones == resultado_matriz:
                st.success(
                    "El resultado por matriz coincide con el resultado por posiciones."
                )
            else:
                st.error(
                    "El resultado no coincide. Revise la matriz generadora o la estructura de posiciones."
                )

    with tabs[4]:
        st.header("Dinámica de aprendizaje")

        st.markdown(
            """
Realice las siguientes actividades:

1. Codifique manualmente el mensaje `1011`.
2. Compare su resultado con la app.
3. Repita el procedimiento para `0001`, `0110` y `1111`.
4. Identifique qué bits corresponden a datos y cuáles a paridad.
5. Compare el método por posiciones con el método matricial.
"""
        )

        pregunta_1 = st.radio(
            "Pregunta 1: En Hamming (7,4), ¿cuántos bits de datos se codifican?",
            [
                "3 bits",
                "4 bits",
                "7 bits",
                "8 bits",
            ],
            index=None,
            key="g3_pregunta_1",
        )

        if pregunta_1:
            if pregunta_1 == "4 bits":
                st.success("Correcto. Hamming (7,4) codifica 4 bits de datos.")
            else:
                st.error("Revise el significado de la notación Hamming (7,4).")

        pregunta_2 = st.radio(
            "Pregunta 2: ¿En qué posiciones se colocan los bits de paridad?",
            [
                "1, 2 y 4",
                "3, 5 y 7",
                "1, 3 y 5",
                "5, 6 y 7",
            ],
            index=None,
            key="g3_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "1, 2 y 4":
                st.success("Correcto. Los bits de paridad se colocan en posiciones potencia de dos.")
            else:
                st.error("Recuerde que las posiciones de paridad son potencias de dos.")

        pregunta_3 = st.radio(
            "Pregunta 3: ¿Cuál es la finalidad de agregar bits de paridad en Hamming?",
            [
                "Reducir el tamaño del mensaje.",
                "Eliminar completamente el ruido del canal.",
                "Agregar redundancia para permitir detección y corrección en el receptor.",
                "Convertir la señal analógica en digital.",
            ],
            index=None,
            key="g3_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "Agregar redundancia para permitir detección y corrección en el receptor.":
                st.success("Correcto. La redundancia permite que el receptor localice errores simples.")
            else:
                st.error("Revise el propósito del control de errores.")

    with tabs[5]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- Hamming (7,4) transforma 4 bits de datos en una palabra código de 7 bits.
- Los bits adicionales son bits de paridad.
- Las posiciones de paridad corresponden a potencias de dos.
- Los bits de paridad se calculan mediante operaciones XOR.
- La redundancia agregada por el transmisor permite que el receptor pueda detectar y corregir errores de un bit.
- La codificación puede realizarse mediante posiciones o mediante una matriz generadora.
"""
        )

    with tabs[6]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2012). *Data communications and networking* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Lin, S., & Costello, D. J. (1983). *Error control coding: Fundamentals and applications*. Prentice-Hall.
"""
        )