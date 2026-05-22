import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Funciones base Hamming (7,4)
# ============================================================

def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_bits_4(bits: str) -> bool:
    """
    Valida que la entrada sea una secuencia binaria de exactamente 4 bits.
    """
    return len(bits) == 4 and all(bit in "01" for bit in bits)


def rellenar_a_multiplo(bits: str, multiplo: int) -> tuple[str, int]:
    """
    Rellena con ceros hasta que la longitud sea múltiplo del tamaño indicado.
    Devuelve la secuencia rellenada y la cantidad de bits agregados.
    """
    residuo = len(bits) % multiplo

    if residuo == 0:
        return bits, 0

    padding = multiplo - residuo
    return bits + ("0" * padding), padding


def dividir_en_bloques(bits: str, tamano_bloque: int) -> list[str]:
    """
    Divide una secuencia binaria en bloques de tamaño fijo.
    """
    return [bits[i:i + tamano_bloque] for i in range(0, len(bits), tamano_bloque)]


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

    # Paridad par
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
            [1, 1, 1, 0, 0, 0, 0],  # D1
            [1, 0, 0, 1, 1, 0, 0],  # D2
            [0, 1, 0, 1, 0, 1, 0],  # D3
            [1, 1, 0, 1, 0, 0, 1],  # D4
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


def codificar_mensaje_por_bloques(bits: str) -> tuple[str, int, pd.DataFrame]:
    """
    Codifica una secuencia de cualquier longitud usando Hamming (7,4).
    Si la longitud no es múltiplo de 4, se agregan ceros de relleno.
    """
    bits_rellenados, padding = rellenar_a_multiplo(bits, 4)
    bloques = dividir_en_bloques(bits_rellenados, 4)

    filas = []
    codigos = []

    for i, bloque in enumerate(bloques, start=1):
        resultado = calcular_paridades_hamming_7_4(bloque)
        codigo = resultado["codigo_str"]
        codigos.append(codigo)

        filas.append(
            {
                "Bloque": i,
                "Datos originales del bloque": bloque,
                "P1": resultado["P1"],
                "P2": resultado["P2"],
                "P4": resultado["P4"],
                "Palabra Hamming (7 bits)": codigo,
            }
        )

    codigo_total = "".join(codigos)
    return codigo_total, padding, pd.DataFrame(filas)


def construir_tabla_estructura_vacia() -> pd.DataFrame:
    """
    Tabla didáctica que muestra la estructura de posiciones antes de colocar los datos.
    """
    return pd.DataFrame(
        {
            "Posición": [1, 2, 3, 4, 5, 6, 7],
            "Tipo": ["P1", "P2", "D1", "P4", "D2", "D3", "D4"],
            "Rol": [
                "Paridad",
                "Paridad",
                "Dato",
                "Paridad",
                "Dato",
                "Dato",
                "Dato",
            ],
            "Potencia de 2": ["Sí", "Sí", "No", "Sí", "No", "No", "No"],
        }
    )


def calcular_eficiencia(k: int, n: int) -> float:
    return k / n


# ============================================================
# Interfaz Streamlit
# ============================================================

def render_guia_03() -> None:
    st.title("Guía 3: Codificación Hamming (7,4) en el transmisor")

    st.markdown(
        """
Esta guía introduce el código Hamming (7,4) como técnica de corrección de errores hacia
adelante. El objetivo es comprender cómo el transmisor agrega redundancia estructurada
a los datos antes de enviarlos por un canal que puede introducir errores.

A diferencia de las guías de ruido y BER, esta guía no requiere gráficas continuas. El
proceso se representa mediante tablas discretas, posiciones, bloques y operaciones XOR.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Codificación de un bloque",
            "Codificación por bloques",
            "Método matricial",
            "Análisis y dinámica",
            "Conclusiones",
            "Referencias",
        ]
    )

    # ========================================================
    # Objetivos
    # ========================================================

    with tabs[0]:
        st.header("Objetivos")

        st.markdown(
            """
**Objetivo general**

Comprender cómo el código Hamming (7,4) agrega redundancia en el transmisor para
proteger una secuencia binaria ante errores de transmisión.

**Objetivos específicos**

1. Identificar la relación entre bits de datos, bits de paridad y palabra código.
2. Reconocer la estructura de posiciones del código Hamming (7,4).
3. Calcular bits de paridad mediante operaciones XOR.
4. Construir una palabra código de 7 bits a partir de 4 bits de datos.
5. Codificar mensajes más largos dividiéndolos en bloques de 4 bits.
6. Calcular la eficiencia de codificación.
7. Comparar el método por posiciones con el método matricial.
8. Relacionar la redundancia agregada con la futura corrección en el receptor.
"""
        )

    # ========================================================
    # Teoría
    # ========================================================

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En las guías anteriores se observó que el ruido puede alterar la señal recibida y producir
errores de bit. Para enfrentar este problema, los sistemas de comunicación agregan
redundancia, es decir, bits adicionales calculados a partir de los datos originales.

El código Hamming es una técnica de corrección de errores hacia adelante. Esto significa
que el transmisor agrega bits de control antes de enviar la información, de modo que el
receptor pueda detectar y corregir ciertos errores sin solicitar retransmisión.

En Hamming (7,4):

- $k = 4$ bits de datos;
- $r = 3$ bits de paridad;
- $n = 7$ bits totales.

La relación entre estos valores es:

$$
n = k + r
$$

Para determinar cuántos bits de paridad se necesitan, se usa la condición:

$$
2^r \\geq k + r + 1
$$

Para $k = 4$:

$$
2^3 = 8 \\geq 4 + 3 + 1 = 8
$$

Por tanto, se requieren tres bits de paridad.

Los bits de paridad se colocan en posiciones que son potencias de dos:

$$
1, 2, 4
$$

Los bits de datos se colocan en las posiciones restantes:

$$
3, 5, 6, 7
$$

La estructura usada en esta guía es:

| Posición | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Tipo | P1 | P2 | D1 | P4 | D2 | D3 | D4 |

El código generado usa paridad par. Esto significa que cada grupo verificado por una
paridad debe tener una cantidad par de unos.

La eficiencia de codificación se define como:

$$
\\eta = \\frac{k}{n}
$$

Para Hamming (7,4):

$$
\\eta = \\frac{4}{7} \\approx 0.5714
$$

Esto significa que aproximadamente el 57.14% de los bits transmitidos corresponden a
información útil y el resto corresponde a redundancia.
"""
        )

        st.subheader("Estructura discreta de posiciones")

        st.dataframe(
            construir_tabla_estructura_vacia(),
            width="stretch",
            hide_index=True,
        )

        eficiencia = calcular_eficiencia(4, 7)

        c1, c2, c3 = st.columns(3)
        c1.metric("Bits de datos k", "4")
        c2.metric("Bits totales n", "7")
        c3.metric("Eficiencia η = k/n", f"{eficiencia:.4f}")

    # ========================================================
    # Codificación de un bloque
    # ========================================================

    with tabs[2]:
        st.header("Codificación de un bloque de 4 bits")

        st.markdown(
            """
En esta sección se codifica un único bloque de 4 bits. La aplicación coloca los bits
de datos en sus posiciones correspondientes y calcula los bits de paridad.
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

            ejecutar = st.button("Codificar bloque con Hamming (7,4)", width="stretch")

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

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits de datos k", "4")
            c2.metric("Bits de paridad r", "3")
            c3.metric("Longitud total n", "7")
            c4.metric("Eficiencia η", f"{calcular_eficiencia(4, 7):.4f}")

            st.code(
                f"Mensaje original: {bits}\n"
                f"Palabra Hamming:  {resultado['codigo_str']}",
                language="text",
            )

            st.subheader("Tabla de posiciones")

            st.dataframe(tabla_posiciones, width="stretch", hide_index=True)

            st.subheader("Cálculo de paridades")

            st.dataframe(tabla_cobertura, width="stretch", hide_index=True)

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

            st.session_state["guia_03_bloque_bits"] = bits
            st.session_state["guia_03_bloque_codigo"] = resultado["codigo_str"]

    # ========================================================
    # Codificación por bloques
    # ========================================================

    with tabs[3]:
        st.header("Codificación por bloques")

        st.markdown(
            """
Hamming (7,4) trabaja sobre bloques de 4 bits de datos. Por tanto, si el mensaje tiene
más de 4 bits, se divide en bloques de 4 bits y cada bloque se codifica por separado.

Ejemplos:

- 8 bits de datos → 2 bloques de 4 bits → 14 bits codificados.
- 16 bits de datos → 4 bloques de 4 bits → 28 bits codificados.
- Si la longitud no es múltiplo de 4, se agregan ceros de relleno.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            modo = st.radio(
                "Modo de entrada",
                ["Mensaje manual", "Mensaje aleatorio"],
                key="g3_modo_bloques",
            )

            if modo == "Mensaje manual":
                bits_largos = st.text_input(
                    "Mensaje binario",
                    value="10110011",
                    help="Puede ingresar 4, 8, 16 o más bits.",
                    key="g3_bits_largos_manual",
                ).strip()
            else:
                cantidad_bits = st.selectbox(
                    "Cantidad de bits de datos",
                    [4, 8, 16, 32],
                    index=2,
                    key="g3_cantidad_bits_aleatorios",
                )

                semilla = st.number_input(
                    "Semilla",
                    min_value=0,
                    max_value=999999,
                    value=30,
                    step=1,
                    key="g3_semilla_bloques",
                )

                rng = np.random.default_rng(int(semilla))
                bits_array = rng.integers(0, 2, size=cantidad_bits)
                bits_largos = "".join(str(bit) for bit in bits_array)

                st.code(f"Mensaje generado: {bits_largos}", language="text")

            ejecutar_bloques = st.button("Codificar mensaje por bloques", width="stretch")

        with col_info:
            st.info(
                """
Cada bloque de 4 bits produce una palabra Hamming de 7 bits.  
Esto aumenta la cantidad de bits transmitidos, pero agrega redundancia útil para la
corrección posterior en el receptor.
"""
            )

        if not validar_bits(bits_largos):
            st.error("El mensaje debe contener únicamente 0 y 1.")
        elif ejecutar_bloques:
            codigo_total, padding, tabla_bloques = codificar_mensaje_por_bloques(bits_largos)

            bits_rellenados, _ = rellenar_a_multiplo(bits_largos, 4)
            cantidad_bloques = len(dividir_en_bloques(bits_rellenados, 4))
            bits_codificados = len(codigo_total)
            eficiencia_total = len(bits_rellenados) / bits_codificados if bits_codificados else 0

            st.subheader("Resumen de codificación por bloques")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits ingresados", len(bits_largos))
            c2.metric("Bits de relleno", padding)
            c3.metric("Bloques de 4 bits", cantidad_bloques)
            c4.metric("Bits codificados", bits_codificados)

            c5, c6, c7 = st.columns(3)
            c5.metric("Eficiencia por bloque", f"{calcular_eficiencia(4, 7):.4f}")
            c6.metric("Eficiencia total", f"{eficiencia_total:.4f}")
            c7.metric("Redundancia total", f"{1 - eficiencia_total:.4f}")

            st.code(
                f"Mensaje original:          {bits_largos}\n"
                f"Mensaje con relleno:       {bits_rellenados}\n"
                f"Palabra codificada total:  {codigo_total}",
                language="text",
            )

            st.subheader("Tabla de bloques codificados")

            st.dataframe(tabla_bloques, width="stretch", hide_index=True)

            st.session_state["guia_03_bloques_resultado"] = {
                "mensaje_original": bits_largos,
                "mensaje_rellenado": bits_rellenados,
                "codigo_total": codigo_total,
                "padding": padding,
                "bloques": cantidad_bloques,
                "eficiencia": eficiencia_total,
            }

    # ========================================================
    # Método matricial
    # ========================================================

    with tabs[4]:
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

            st.dataframe(df_G, width="stretch")

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

    # ========================================================
    # Análisis y dinámica
    # ========================================================

    with tabs[5]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección combina interpretación de resultados, ejercicios guiados y preguntas
conceptuales. El objetivo es que el estudiante comprenda no solo el resultado de la
codificación, sino también el costo de agregar redundancia.
"""
        )

        if "guia_03_bloque_bits" in st.session_state:
            st.subheader("Último bloque codificado")

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Bloque original": st.session_state["guia_03_bloque_bits"],
                            "Palabra Hamming": st.session_state["guia_03_bloque_codigo"],
                            "Eficiencia η": calcular_eficiencia(4, 7),
                        }
                    ]
                )
            )
        else:
            st.info("Codifique primero un bloque en la pestaña correspondiente.")

        if "guia_03_bloques_resultado" in st.session_state:
            st.subheader("Última codificación por bloques")

            resultado = st.session_state["guia_03_bloques_resultado"]

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Mensaje original": resultado["mensaje_original"],
                            "Mensaje con relleno": resultado["mensaje_rellenado"],
                            "Bits de relleno": resultado["padding"],
                            "Bloques": resultado["bloques"],
                            "Bits codificados": len(resultado["codigo_total"]),
                            "Eficiencia total": resultado["eficiencia"],
                        }
                    ]
                )
            )
        else:
            st.info("Ejecute una codificación por bloques para ver el resumen.")

        st.subheader("Actividades guiadas")

        st.markdown(
            """
Realice las siguientes actividades:

1. Codifique manualmente el bloque `1011`.
2. Compare su resultado con la app.
3. Repita el procedimiento para `0001`, `0110` y `1111`.
4. Identifique qué bits corresponden a datos y cuáles a paridad.
5. Ingrese un mensaje de 8 bits y observe que se divide en 2 bloques.
6. Ingrese un mensaje de 16 bits y observe que se divide en 4 bloques.
7. Calcule la eficiencia de Hamming (7,4).
8. Explique por qué agregar redundancia aumenta la cantidad de bits transmitidos.
"""
        )

        st.subheader("Preguntas de análisis")

        pregunta_1 = st.radio(
            "Pregunta 1: En Hamming (7,4), ¿cuántos bits de datos se codifican por bloque?",
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
                st.success("Correcto. Hamming (7,4) codifica 4 bits de datos por bloque.")
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

        pregunta_4 = st.radio(
            "Pregunta 4: ¿Cuál es la eficiencia de Hamming (7,4)?",
            [
                "3/7",
                "4/7",
                "7/4",
                "1/7",
            ],
            index=None,
            key="g3_pregunta_4",
        )

        if pregunta_4:
            if pregunta_4 == "4/7":
                st.success("Correcto. La eficiencia es η = k/n = 4/7.")
            else:
                st.error("Revise la definición de eficiencia η = k/n.")

        pregunta_5 = st.radio(
            "Pregunta 5: Si se codifican 16 bits de datos con Hamming (7,4), ¿cuántos bloques se forman?",
            [
                "2 bloques",
                "3 bloques",
                "4 bloques",
                "7 bloques",
            ],
            index=None,
            key="g3_pregunta_5",
        )

        if pregunta_5:
            if pregunta_5 == "4 bloques":
                st.success("Correcto. 16 bits / 4 bits por bloque = 4 bloques.")
            else:
                st.error("Divida la cantidad total de bits entre 4 bits por bloque.")

    # ========================================================
    # Conclusiones
    # ========================================================

    with tabs[6]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- Hamming (7,4) transforma 4 bits de datos en una palabra código de 7 bits.
- Los bits adicionales son bits de paridad.
- Las posiciones de paridad corresponden a potencias de dos.
- Los bits de paridad se calculan mediante operaciones XOR.
- Los mensajes largos se dividen en bloques de 4 bits.
- Cada bloque se codifica de forma independiente.
- La eficiencia de Hamming (7,4) es $\\eta = 4/7$.
- La redundancia aumenta los bits transmitidos, pero permite corrección posterior.
- La codificación puede realizarse mediante posiciones o mediante una matriz generadora.
"""
        )

    # ========================================================
    # Referencias
    # ========================================================

    with tabs[7]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2012). *Data communications and networking* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Lin, S., & Costello, D. J. (1983). *Error control coding: Fundamentals and applications*. Prentice-Hall.
"""
        )