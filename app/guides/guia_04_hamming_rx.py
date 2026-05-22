import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Funciones base Hamming (7,4)
# Estructura: [P1 P2 D1 P4 D2 D3 D4]
# ============================================================

def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_codigo_hamming(bits: str) -> bool:
    """
    Valida que la palabra recibida tenga exactamente 7 bits.
    """
    return len(bits) == 7 and all(bit in "01" for bit in bits)


def validar_longitud_multiple_7(bits: str) -> bool:
    """
    Valida que una secuencia tenga longitud múltiplo de 7.
    """
    return validar_bits(bits) and len(bits) % 7 == 0


def dividir_en_bloques(bits: str, tamano: int) -> list[str]:
    """
    Divide una secuencia binaria en bloques de tamaño fijo.
    """
    return [bits[i:i + tamano] for i in range(0, len(bits), tamano)]


def invertir_bit(bits: str, posicion: int) -> str:
    """
    Invierte el bit de una posición dada. La posición inicia en 1.
    """
    lista = list(bits)
    indice = posicion - 1
    lista[indice] = "1" if lista[indice] == "0" else "0"
    return "".join(lista)


def extraer_datos_hamming(bits: str) -> str:
    """
    Extrae los datos D1 D2 D3 D4 de una palabra Hamming (7,4).
    """
    return bits[2] + bits[4] + bits[5] + bits[6]


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
        corrigio = False
    else:
        corregido = invertir_bit(bits, posicion_error)
        estado = f"Error corregido en la posición {posicion_error}"
        corrigio = True

    datos_extraidos = extraer_datos_hamming(corregido)

    return {
        "recibido": bits,
        "sindrome": sindrome,
        "corregido": corregido,
        "datos_extraidos": datos_extraidos,
        "estado": estado,
        "corrigio": corrigio,
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


def construir_tabla_estructura_receptor() -> pd.DataFrame:
    """
    Tabla de estructura de posiciones para el receptor.
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
            "Participa en": [
                "s1",
                "s2",
                "s1, s2",
                "s4",
                "s1, s4",
                "s2, s4",
                "s1, s2, s4",
            ],
        }
    )


def decodificar_bloques_hamming(bits_codificados: str) -> tuple[str, str, pd.DataFrame]:
    """
    Decodifica una secuencia formada por bloques Hamming de 7 bits.
    Devuelve datos recuperados, bloques corregidos y tabla de resultados.
    """
    bloques = dividir_en_bloques(bits_codificados, 7)

    datos_recuperados = []
    bloques_corregidos = []
    filas = []

    for i, bloque in enumerate(bloques, start=1):
        resultado = corregir_hamming(bloque)
        sindrome = resultado["sindrome"]

        datos_recuperados.append(resultado["datos_extraidos"])
        bloques_corregidos.append(resultado["corregido"])

        filas.append(
            {
                "Bloque": i,
                "Recibido": bloque,
                "Síndrome": sindrome["sindrome_binario"],
                "Posición detectada": sindrome["posicion_error"],
                "Corrección aplicada": "Sí" if resultado["corrigio"] else "No",
                "Bloque corregido": resultado["corregido"],
                "Datos recuperados": resultado["datos_extraidos"],
                "Estado": resultado["estado"],
            }
        )

    return "".join(datos_recuperados), "".join(bloques_corregidos), pd.DataFrame(filas)


def inyectar_errores_en_bloques(
    bits_codificados: str,
    cantidad_errores_por_bloque: int,
    semilla: int | None = None,
) -> tuple[str, pd.DataFrame]:
    """
    Inyecta errores aleatorios por bloque de 7 bits.
    La cantidad de errores por bloque puede ser 0, 1 o 2.
    """
    rng = np.random.default_rng(semilla)
    bloques = dividir_en_bloques(bits_codificados, 7)

    bloques_rx = []
    filas = []

    for i, bloque in enumerate(bloques, start=1):
        bloque_rx = bloque
        posiciones = []

        if cantidad_errores_por_bloque > 0:
            posiciones = rng.choice(
                np.arange(1, 8),
                size=cantidad_errores_por_bloque,
                replace=False,
            ).tolist()

            for posicion in posiciones:
                bloque_rx = invertir_bit(bloque_rx, int(posicion))

        bloques_rx.append(bloque_rx)

        filas.append(
            {
                "Bloque": i,
                "Bloque original": bloque,
                "Posiciones alteradas": ", ".join(str(p) for p in posiciones) if posiciones else "Ninguna",
                "Bloque recibido": bloque_rx,
            }
        )

    return "".join(bloques_rx), pd.DataFrame(filas)


# ============================================================
# Interfaz Streamlit
# ============================================================

def render_guia_04() -> None:
    st.title("Guía 4: Decodificación Hamming y corrección mediante síndrome")

    st.markdown(
        """
Esta guía estudia el proceso realizado por el receptor cuando se utiliza código
Hamming (7,4). El estudiante podrá inyectar errores, calcular el síndrome, localizar
la posición afectada y observar la corrección de errores de un solo bit.

Siguiendo la estructura de la Guía 3, el análisis se mantiene discreto mediante tablas,
bloques y operaciones XOR. Hamming (7,4) corrige un error por bloque de 7 bits, pero no
garantiza corrección correcta cuando ocurren errores múltiples dentro del mismo bloque.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Corrección de un bloque",
            "Decodificación por bloques",
            "Errores múltiples",
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

Comprender cómo el código Hamming (7,4) permite detectar y corregir errores de un solo
bit mediante el cálculo del síndrome en el receptor.

**Objetivos específicos**

1. Identificar la estructura de una palabra Hamming (7,4).
2. Calcular las comprobaciones de paridad en el receptor.
3. Construir el síndrome a partir de los bits recibidos.
4. Interpretar el síndrome como la posición del error.
5. Corregir un error de un solo bit.
6. Extraer los datos originales después de la corrección.
7. Decodificar secuencias formadas por varios bloques Hamming.
8. Analizar las limitaciones de Hamming frente a errores múltiples.
"""
        )

    # ========================================================
    # Teoría
    # ========================================================

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

Si el síndrome es cero:

$$
S = 000
$$

no se detecta error. Si el síndrome es distinto de cero, su valor decimal indica la
posición del bit que debe corregirse.

Por ejemplo:

$$
S = 101_2 = 5
$$

indica un error en la posición 5.

El código Hamming (7,4) permite corregir un error de un solo bit por bloque. Si ocurren
dos o más errores dentro del mismo bloque, el síndrome puede apuntar a una posición
incorrecta y producir una corrección equivocada.
"""
        )

        st.subheader("Estructura del receptor")

        st.dataframe(
            construir_tabla_estructura_receptor(),
            width="stretch",
            hide_index=True,
        )

        st.info(
            """
Interpretación importante: el síndrome no es una cantidad de errores. En Hamming (7,4),
el síndrome se interpreta como la posición del error cuando se asume que hay un solo bit
alterado dentro del bloque.
"""
        )

    # ========================================================
    # Corrección de un bloque
    # ========================================================

    with tabs[2]:
        st.header("Corrección de un bloque de 7 bits")

        st.markdown(
            """
Ingrese una palabra Hamming válida de 7 bits. Luego seleccione una posición para inyectar
un error. La aplicación calculará el síndrome y aplicará la corrección correspondiente.
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

            ejecutar = st.button("Inyectar error y corregir", width="stretch")

        with col_info:
            st.info(
                """
Si ocurre un solo error dentro del bloque, el síndrome indica la posición alterada.
El receptor corrige invirtiendo el bit señalado por el síndrome.
"""
            )

        if not validar_codigo_hamming(palabra_tx):
            st.error("Debe ingresar exactamente 7 bits. Ejemplo válido: 0110011.")
        elif ejecutar:
            palabra_rx = invertir_bit(palabra_tx, int(posicion_error))
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

            st.dataframe(tabla_sindrome, width="stretch", hide_index=True)

            st.subheader("Comparación transmitido, recibido y corregido")

            st.dataframe(tabla_comparacion, width="stretch", hide_index=True)

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

Por tanto, bajo la hipótesis de un único error, el receptor identifica error en la
posición {sindrome["posicion_error"]}.
"""
            )

            st.session_state["guia_04_ultimo_bloque"] = {
                "tx": palabra_tx,
                "rx": palabra_rx,
                "corregido": resultado["corregido"],
                "sindrome": sindrome["sindrome_binario"],
                "posicion": sindrome["posicion_error"],
                "datos": resultado["datos_extraidos"],
                "estado": resultado["estado"],
            }

    # ========================================================
    # Decodificación por bloques
    # ========================================================

    with tabs[3]:
        st.header("Decodificación por bloques")

        st.markdown(
            """
En la Guía 3 se codificaron mensajes largos dividiéndolos en bloques de 4 bits, donde
cada bloque produjo una palabra Hamming de 7 bits. En esta sección se realiza el proceso
inverso: una secuencia codificada se divide en bloques de 7 bits, se calcula el síndrome
de cada bloque y se recuperan los datos.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            secuencia_codificada = st.text_area(
                "Secuencia Hamming codificada",
                value="01100111110000",
                help="Debe tener longitud múltiplo de 7.",
                key="g4_secuencia_bloques",
            ).strip().replace(" ", "").replace("\n", "")

            cantidad_errores = st.selectbox(
                "Errores aleatorios por bloque",
                [0, 1, 2],
                index=1,
                key="g4_errores_por_bloque",
            )

            semilla = st.number_input(
                "Semilla",
                min_value=0,
                max_value=999999,
                value=50,
                step=1,
                key="g4_semilla_bloques",
            )

            ejecutar_bloques = st.button("Inyectar errores y decodificar bloques", width="stretch")

        with col_info:
            st.info(
                """
Use 1 error por bloque para observar corrección exitosa.  
Use 2 errores por bloque para observar los límites de Hamming.
"""
            )

        if not validar_longitud_multiple_7(secuencia_codificada):
            st.error("La secuencia debe ser binaria y tener longitud múltiplo de 7.")
        elif ejecutar_bloques:
            secuencia_rx, tabla_inyeccion = inyectar_errores_en_bloques(
                bits_codificados=secuencia_codificada,
                cantidad_errores_por_bloque=int(cantidad_errores),
                semilla=int(semilla),
            )

            datos_recuperados, bloques_corregidos, tabla_decodificacion = decodificar_bloques_hamming(
                secuencia_rx
            )

            bloques_originales = dividir_en_bloques(secuencia_codificada, 7)
            bloques_finales = dividir_en_bloques(bloques_corregidos, 7)

            bloques_correctos = sum(
                1 for original, corregido in zip(bloques_originales, bloques_finales)
                if original == corregido
            )

            total_bloques = len(bloques_originales)

            st.subheader("Resumen por bloques")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bloques evaluados", total_bloques)
            c2.metric("Errores por bloque", cantidad_errores)
            c3.metric("Bloques recuperados", bloques_correctos)
            c4.metric("Datos recuperados", len(datos_recuperados))

            st.code(
                f"Secuencia transmitida: {secuencia_codificada}\n"
                f"Secuencia recibida:    {secuencia_rx}\n"
                f"Bloques corregidos:    {bloques_corregidos}\n"
                f"Datos recuperados:     {datos_recuperados}",
                language="text",
            )

            st.subheader("Errores inyectados por bloque")
            st.dataframe(tabla_inyeccion, width="stretch", hide_index=True)

            st.subheader("Decodificación por síndrome")
            st.dataframe(tabla_decodificacion, width="stretch", hide_index=True)

            if cantidad_errores == 0:
                st.success("No se inyectaron errores. Los bloques deberían recuperarse sin corrección.")
            elif cantidad_errores == 1:
                st.success(
                    "Se inyectó un error por bloque. Hamming debería corregir cada bloque de forma adecuada."
                )
            else:
                st.warning(
                    "Se inyectaron dos errores por bloque. Hamming puede aplicar una corrección incorrecta."
                )

            st.session_state["guia_04_bloques_resultado"] = {
                "tx": secuencia_codificada,
                "rx": secuencia_rx,
                "corregidos": bloques_corregidos,
                "datos": datos_recuperados,
                "errores_por_bloque": cantidad_errores,
                "bloques": total_bloques,
                "bloques_correctos": bloques_correctos,
            }

    # ========================================================
    # Errores múltiples
    # ========================================================

    with tabs[4]:
        st.header("Errores múltiples y límite de Hamming")

        st.markdown(
            """
En esta sección se inyectan dos errores dentro de un mismo bloque Hamming. El objetivo
es observar que Hamming (7,4) no garantiza corrección adecuada cuando ocurren errores
múltiples.
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

            ejecutar_multi = st.button("Inyectar dos errores", width="stretch")

        with col_info:
            st.warning(
                """
Hamming (7,4) corrige un solo error por bloque. Con dos errores, el síndrome puede
señalar una posición que no corresponde a un único error real.
"""
            )

        if not validar_codigo_hamming(palabra_multi):
            st.error("Debe ingresar exactamente 7 bits.")
        elif pos_1 == pos_2:
            st.error("Seleccione dos posiciones distintas.")
        elif ejecutar_multi:
            palabra_rx = invertir_bit(palabra_multi, int(pos_1))
            palabra_rx = invertir_bit(palabra_rx, int(pos_2))

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
                f"Palabra corregida:   {resultado['corregido']}\n"
                f"Datos extraídos:     {resultado['datos_extraidos']}",
                language="text",
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Síndrome", sindrome["sindrome_binario"])
            c2.metric("Posición sugerida", sindrome["posicion_error"])
            c3.metric(
                "Coincide con original",
                "Sí" if resultado["corregido"] == palabra_multi else "No",
            )

            st.subheader("Tabla de comprobaciones")
            st.dataframe(tabla_sindrome, width="stretch", hide_index=True)

            st.subheader("Comparación")
            st.dataframe(tabla_comparacion, width="stretch", hide_index=True)

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

            st.session_state["guia_04_multi_resultado"] = {
                "tx": palabra_multi,
                "rx": palabra_rx,
                "corregido": resultado["corregido"],
                "sindrome": sindrome["sindrome_binario"],
                "posicion": sindrome["posicion_error"],
                "coincide": resultado["corregido"] == palabra_multi,
            }

    # ========================================================
    # Análisis y dinámica
    # ========================================================

    with tabs[5]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección integra la interpretación de resultados con ejercicios guiados. El objetivo
es comprobar que el estudiante comprende el significado del síndrome y los límites del
código Hamming.
"""
        )

        if "guia_04_ultimo_bloque" in st.session_state:
            st.subheader("Última corrección de un bloque")

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Transmitido": st.session_state["guia_04_ultimo_bloque"]["tx"],
                            "Recibido": st.session_state["guia_04_ultimo_bloque"]["rx"],
                            "Corregido": st.session_state["guia_04_ultimo_bloque"]["corregido"],
                            "Síndrome": st.session_state["guia_04_ultimo_bloque"]["sindrome"],
                            "Posición detectada": st.session_state["guia_04_ultimo_bloque"]["posicion"],
                            "Datos recuperados": st.session_state["guia_04_ultimo_bloque"]["datos"],
                        }
                    ]
                )
            )
        else:
            st.info("Ejecute primero la corrección de un bloque.")

        if "guia_04_bloques_resultado" in st.session_state:
            st.subheader("Última decodificación por bloques")

            resultado_bloques = st.session_state["guia_04_bloques_resultado"]

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Bloques": resultado_bloques["bloques"],
                            "Errores por bloque": resultado_bloques["errores_por_bloque"],
                            "Bloques recuperados": resultado_bloques["bloques_correctos"],
                            "Datos recuperados": resultado_bloques["datos"],
                        }
                    ]
                )
            )
        else:
            st.info("Ejecute una decodificación por bloques para ver el resumen.")

        st.subheader("Actividades guiadas")

        st.markdown(
            """
Realice las siguientes actividades:

1. Use la palabra Hamming `0110011`.
2. Inyecte un error en la posición 1 y registre el síndrome.
3. Repita para las posiciones 2, 3, 4, 5, 6 y 7.
4. Verifique que el síndrome coincide con la posición del error.
5. Decodifique una secuencia de dos bloques de 7 bits.
6. Inyecte un error por bloque y observe la corrección.
7. Inyecte dos errores por bloque y observe el límite del código.
8. Explique por qué CRC será necesario en la siguiente etapa.
"""
        )

        st.subheader("Preguntas de análisis")

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
                "No puede corregir de forma confiable errores múltiples dentro del mismo bloque.",
                "No utiliza bits de paridad.",
                "No permite construir palabras código.",
            ],
            index=None,
            key="g4_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "No puede corregir de forma confiable errores múltiples dentro del mismo bloque.":
                st.success("Correcto. Hamming (7,4) corrige un error por bloque, pero falla ante errores múltiples.")
            else:
                st.error("Revise la capacidad de corrección del código Hamming.")

        pregunta_4 = st.radio(
            "Pregunta 4: Si una secuencia Hamming tiene 21 bits, ¿cuántos bloques de 7 bits contiene?",
            [
                "2 bloques",
                "3 bloques",
                "4 bloques",
                "7 bloques",
            ],
            index=None,
            key="g4_pregunta_4",
        )

        if pregunta_4:
            if pregunta_4 == "3 bloques":
                st.success("Correcto. 21 bits / 7 bits por bloque = 3 bloques.")
            else:
                st.error("Divida la longitud total entre 7 bits por bloque.")

        pregunta_5 = st.radio(
            "Pregunta 5: ¿Por qué CRC complementa a Hamming?",
            [
                "Porque CRC corrige todos los errores múltiples.",
                "Porque CRC permite detectar errores remanentes que Hamming puede no corregir.",
                "Porque CRC elimina la necesidad de bits de paridad.",
                "Porque CRC convierte los bits en señal analógica.",
            ],
            index=None,
            key="g4_pregunta_5",
        )

        if pregunta_5:
            if pregunta_5 == "Porque CRC permite detectar errores remanentes que Hamming puede no corregir.":
                st.success("Correcto. CRC detecta errores remanentes, especialmente cuando Hamming falla ante errores múltiples.")
            else:
                st.error("Revise la diferencia entre corrección con Hamming y detección con CRC.")

    # ========================================================
    # Conclusiones
    # ========================================================

    with tabs[6]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- el receptor calcula un síndrome a partir de verificaciones de paridad;
- el síndrome se interpreta como la posición del error bajo la hipótesis de un único error;
- la corrección se realiza invirtiendo el bit indicado por el síndrome;
- Hamming (7,4) permite recuperar los 4 bits de datos si ocurre un solo error en el bloque;
- las secuencias largas se decodifican dividiéndolas en bloques de 7 bits;
- Hamming corrige un error por bloque, no errores múltiples dentro del mismo bloque;
- cuando ocurren errores múltiples, el síndrome puede inducir una corrección incorrecta;
- esta limitación justifica la integración posterior de CRC como detector de errores remanentes.
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