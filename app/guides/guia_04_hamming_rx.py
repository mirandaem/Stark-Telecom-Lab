from typing import List

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Funciones base Hamming (7,4)
# Estructura usada: [P1 P2 D1 P4 D2 D3 D4]
# ============================================================

def validar_bits(bits: str) -> bool:
    """
    Valida que una cadena contenga únicamente bits 0 y 1.
    """
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_codigo_hamming(bits: str) -> bool:
    """
    Valida que una palabra Hamming (7,4) tenga exactamente 7 bits.
    """
    return len(bits) == 7 and all(bit in "01" for bit in bits)


def validar_longitud_multiple_7(bits: str) -> bool:
    """
    Valida que una secuencia codificada pueda dividirse exactamente
    en bloques Hamming de 7 bits.

    En recepción no se agregan bits de relleno automáticamente.
    Si la secuencia no es múltiplo de 7, se considera incompleta
    o mal formada para decodificación Hamming (7,4).
    """
    return validar_bits(bits) and len(bits) % 7 == 0


def dividir_en_bloques(bits: str, tamano: int) -> List[str]:
    """
    Divide una secuencia binaria en bloques de tamaño fijo.
    """
    return [bits[i:i + tamano] for i in range(0, len(bits), tamano)]


def invertir_bit(bits: str, posicion: int) -> str:
    """
    Invierte el bit de una posición dada.
    La posición se cuenta desde 1.
    """
    lista = list(bits)
    indice = posicion - 1
    lista[indice] = "1" if lista[indice] == "0" else "0"
    return "".join(lista)


def extraer_datos_hamming(bits: str) -> str:
    """
    Extrae los datos D1 D2 D3 D4 desde la estructura:

    [P1 P2 D1 P4 D2 D3 D4]

    Posiciones de datos:
    D1 -> posición 3
    D2 -> posición 5
    D3 -> posición 6
    D4 -> posición 7
    """
    return bits[2] + bits[4] + bits[5] + bits[6]


def calcular_sindrome(bits: str) -> dict:
    """
    Calcula el síndrome para Hamming (7,4) con paridad par.

    Estructura:
    Posición: 1  2  3  4  5  6  7
    Tipo:     P1 P2 D1 P4 D2 D3 D4

    Comprobaciones de paridad:
    s1 revisa posiciones 1, 3, 5, 7
    s2 revisa posiciones 2, 3, 6, 7
    s4 revisa posiciones 4, 5, 6, 7

    El síndrome se forma como:

    S = (s4 s2 s1)_2

    Si S = 000, no se detecta error.
    Si S != 000, su valor decimal indica la posición del error,
    bajo la hipótesis de que solo ocurrió un error en el bloque.
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
    Aplica corrección Hamming (7,4) a una palabra recibida.

    Si el síndrome es 000, no se detecta error.
    Si el síndrome es distinto de 000, se invierte el bit indicado
    por el valor decimal del síndrome.

    Esta corrección es válida bajo la hipótesis de un único error
    dentro del bloque.
    """
    sindrome = calcular_sindrome(bits)
    posicion_error = sindrome["posicion_error"]

    if posicion_error == 0:
        corregido = bits
        estado = "Sin error detectable"
        corrigio = False
    else:
        corregido = invertir_bit(bits, posicion_error)
        estado = f"Corrección aplicada en posición {posicion_error}"
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


# ============================================================
# Tablas didácticas
# ============================================================

def construir_tabla_estructura_receptor() -> pd.DataFrame:
    """
    Tabla que muestra la estructura de posiciones usada por el receptor.
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
            "Interpretación": [
                "Bit de control asociado a la comprobación s1",
                "Bit de control asociado a la comprobación s2",
                "Bit de dato revisado por s1 y s2",
                "Bit de control asociado a la comprobación s4",
                "Bit de dato revisado por s1 y s4",
                "Bit de dato revisado por s2 y s4",
                "Bit de dato revisado por s1, s2 y s4",
            ],
        }
    )


def construir_tabla_sindrome(bits: str, sindrome: dict) -> pd.DataFrame:
    """
    Construye una tabla con las comprobaciones de paridad realizadas
    por el receptor.
    """
    b = [int(bit) for bit in bits]

    return pd.DataFrame(
        [
            {
                "Comprobación": "s1",
                "Posiciones revisadas": "1, 3, 5, 7",
                "Bits revisados": f"{b[0]}, {b[2]}, {b[4]}, {b[6]}",
                "Operación XOR": f"{b[0]} ⊕ {b[2]} ⊕ {b[4]} ⊕ {b[6]}",
                "Resultado": sindrome["s1"],
                "Interpretación": "0 indica paridad correcta; 1 indica inconsistencia en este grupo",
            },
            {
                "Comprobación": "s2",
                "Posiciones revisadas": "2, 3, 6, 7",
                "Bits revisados": f"{b[1]}, {b[2]}, {b[5]}, {b[6]}",
                "Operación XOR": f"{b[1]} ⊕ {b[2]} ⊕ {b[5]} ⊕ {b[6]}",
                "Resultado": sindrome["s2"],
                "Interpretación": "0 indica paridad correcta; 1 indica inconsistencia en este grupo",
            },
            {
                "Comprobación": "s4",
                "Posiciones revisadas": "4, 5, 6, 7",
                "Bits revisados": f"{b[3]}, {b[4]}, {b[5]}, {b[6]}",
                "Operación XOR": f"{b[3]} ⊕ {b[4]} ⊕ {b[5]} ⊕ {b[6]}",
                "Resultado": sindrome["s4"],
                "Interpretación": "0 indica paridad correcta; 1 indica inconsistencia en este grupo",
            },
        ]
    )


def construir_tabla_comparacion_un_error(original: str, recibido: str, corregido: str) -> pd.DataFrame:
    """
    Compara palabra transmitida, recibida y corregida para el caso
    de un único error.
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
                "Cambio por Hamming": "Sí" if rx != corr else "No",
                "Coincide al final": "Sí" if tx == corr else "No",
            }
        )

    return pd.DataFrame(filas)


def construir_tabla_errores_multiples(
    original: str,
    recibido: str,
    corregido: str,
    posiciones_reales: List[int],
    posicion_sindrome: int,
) -> pd.DataFrame:
    """
    Tabla específica para explicar errores múltiples.

    Muestra:
    - qué posiciones fueron alteradas realmente;
    - qué posición señaló el síndrome;
    - qué posición modificó Hamming;
    - si la palabra final coincide con la original.
    """
    filas = []

    for i, (tx, rx, corr) in enumerate(zip(original, recibido, corregido), start=1):
        error_real = i in posiciones_reales
        corregido_por_hamming = i == posicion_sindrome and posicion_sindrome != 0

        if error_real and corregido_por_hamming:
            interpretacion = "Era una posición alterada y Hamming modificó esta posición"
        elif error_real and not corregido_por_hamming:
            interpretacion = "Error real que permaneció después de la corrección"
        elif not error_real and corregido_por_hamming:
            interpretacion = "Hamming modificó una posición que no era error real"
        else:
            interpretacion = "Sin cambio relevante"

        filas.append(
            {
                "Posición": i,
                "Transmitido": tx,
                "Recibido": rx,
                "Corregido": corr,
                "Error real inyectado": "Sí" if error_real else "No",
                "Posición indicada por síndrome": "Sí" if corregido_por_hamming else "No",
                "Coincide al final": "Sí" if tx == corr else "No",
                "Interpretación": interpretacion,
            }
        )

    return pd.DataFrame(filas)


def construir_tabla_bloques_visual(tx: str, rx: str, corregidos: str) -> pd.DataFrame:
    """
    Construye una tabla para visualizar bloques transmitidos, recibidos
    y corregidos.
    """
    bloques_tx = dividir_en_bloques(tx, 7)
    bloques_rx = dividir_en_bloques(rx, 7)
    bloques_corregidos = dividir_en_bloques(corregidos, 7)

    filas = []

    for i, (b_tx, b_rx, b_corr) in enumerate(
        zip(bloques_tx, bloques_rx, bloques_corregidos),
        start=1,
    ):
        filas.append(
            {
                "Bloque": i,
                "Bloque transmitido": b_tx,
                "Bloque recibido": b_rx,
                "Bloque corregido": b_corr,
                "Coincide con transmitido": "Sí" if b_tx == b_corr else "No",
                "Datos recuperados": extraer_datos_hamming(b_corr),
            }
        )

    return pd.DataFrame(filas)


def construir_tabla_capacidad_hamming() -> pd.DataFrame:
    """
    Tabla resumen de capacidades y limitaciones de Hamming (7,4).
    """
    return pd.DataFrame(
        {
            "Caso": [
                "Sin error en el bloque",
                "Un error en el bloque",
                "Dos errores en el mismo bloque",
                "Más de dos errores en el mismo bloque",
            ],
            "Síndrome": [
                "000",
                "Distinto de 000",
                "Puede ser distinto de 000",
                "Puede ser distinto de 000",
            ],
            "Interpretación": [
                "No se detecta error",
                "El síndrome indica la posición del error",
                "El síndrome puede apuntar a una posición incorrecta",
                "El resultado no es confiable",
            ],
            "Resultado esperado": [
                "La palabra se deja igual",
                "Se corrige invirtiendo el bit indicado",
                "La corrección puede ser incorrecta",
                "La corrección puede ser incorrecta",
            ],
        }
    )


# ============================================================
# Decodificación por bloques
# ============================================================

def decodificar_bloques_hamming(bits_codificados: str) -> tuple[str, str, pd.DataFrame]:
    """
    Decodifica una secuencia formada por bloques Hamming de 7 bits.

    Devuelve:
    - datos recuperados;
    - secuencia de bloques corregidos;
    - tabla de resultados por bloque.
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

    cantidad_errores_por_bloque:
    - 0: no altera el bloque;
    - 1: caso corregible por Hamming (7,4);
    - 2: caso que evidencia el límite del código.
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
                "Bloque transmitido": bloque,
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
Esta guía estudia el proceso realizado por el receptor cuando se utiliza el código
Hamming (7,4). En la Guía 3 se explicó cómo el transmisor agrega bits de paridad
para formar una palabra codificada de 7 bits. Ahora se analiza cómo el receptor usa
esa redundancia para detectar y corregir errores.

El término correcto en español es **síndrome**. En este contexto, el síndrome es el
patrón formado por las comprobaciones de paridad del receptor. Cuando se asume que
solo ocurrió un error dentro del bloque, el valor decimal del síndrome indica la
posición que debe corregirse.

El código Hamming fue propuesto originalmente como un método para detectar y corregir
errores en información binaria, y forma parte de la teoría de códigos de control de
errores usada en comunicaciones digitales (Hamming, 1950; Lin & Costello, 2004).
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

Comprender cómo el código Hamming (7,4) permite detectar y corregir errores de un
solo bit mediante el cálculo del síndrome en el receptor.

**Objetivos específicos**

1. Identificar la estructura de una palabra Hamming (7,4).
2. Calcular las comprobaciones de paridad en el receptor.
3. Construir el síndrome a partir de los bits recibidos.
4. Interpretar el síndrome como la posición del error bajo la hipótesis de un solo error.
5. Corregir un error de un solo bit.
6. Extraer los datos originales después de la corrección.
7. Decodificar secuencias formadas por varios bloques Hamming.
8. Validar que la secuencia recibida tenga longitud múltiplo de 7.
9. Analizar las limitaciones de Hamming frente a errores múltiples.
10. Relacionar la limitación de Hamming con la necesidad de usar CRC en etapas posteriores.
"""
        )

    # ========================================================
    # Teoría
    # ========================================================

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En la Guía 3 se estudió cómo el transmisor agrega redundancia a un bloque de 4 bits
para formar una palabra Hamming de 7 bits. En esta guía se analiza el proceso inverso:
el receptor toma la palabra recibida, revisa las paridades y determina si existe un error.

El código Hamming (7,4) pertenece a los códigos de bloque lineales. Esto significa que
opera sobre bloques de longitud fija: 4 bits de datos se transforman en 7 bits codificados.
En recepción, cada bloque de 7 bits debe analizarse de forma independiente (Hamming, 1950;
Lin & Costello, 2004).

La estructura utilizada es:

| Posición | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Tipo | P1 | P2 | D1 | P4 | D2 | D3 | D4 |

Los bits de paridad se ubican en las posiciones 1, 2 y 4. Los bits de datos se ubican
en las posiciones 3, 5, 6 y 7. Esta organización permite que cada posición del bloque
participe en una combinación particular de comprobaciones de paridad (Lin & Costello, 2004).
"""
        )

        st.subheader("Estructura de posiciones en el receptor")

        st.dataframe(
            construir_tabla_estructura_receptor(),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Cálculo del síndrome")

        st.markdown(
            """
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

Cada comprobación produce un bit del síndrome. El síndrome se forma como:

$$
S = (s_4s_2s_1)_2
$$

Si el síndrome es:

$$
S = 000
$$

entonces no se detecta error bajo las comprobaciones de paridad.

Si el síndrome es distinto de cero, su valor decimal indica la posición del bit que debe
invertirse, siempre que se cumpla la hipótesis de que solo ocurrió un error en el bloque.
Por ejemplo:

$$
S = 101_2 = 5
$$

indica error en la posición 5.

Este mecanismo funciona porque cada posición de la palabra Hamming tiene una combinación
única de participación en las comprobaciones de paridad (Hamming, 1950; Forouzan, 2013).
"""
        )

        st.subheader("Capacidad y límite de Hamming (7,4)")

        st.markdown(
            """
Hamming (7,4) permite corregir **un error por bloque**. Esta frase es muy importante:
no significa que Hamming pueda corregir cualquier cantidad de errores, sino que su
corrección es confiable cuando ocurre un único error dentro de cada palabra de 7 bits.

Cuando ocurren dos o más errores dentro del mismo bloque, el síndrome puede tomar un
valor distinto de cero y señalar una posición incorrecta. En ese caso, el receptor puede
invertir un bit que no era el único problema y producir una palabra final incorrecta.

Por esta razón, Hamming se complementa frecuentemente con mecanismos de detección,
como CRC, para verificar si después de la corrección todavía quedan errores remanentes
(Lin & Costello, 2004; Stallings, 2015).
"""
        )

        st.dataframe(
            construir_tabla_capacidad_hamming(),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            """
En esta guía no se agregan bits de relleno en recepción. La secuencia recibida debe
tener longitud múltiplo de 7 porque cada palabra Hamming (7,4) contiene exactamente
7 bits. Si falta o sobra un bit, la secuencia no puede dividirse correctamente en
bloques Hamming.
"""
        )

    # ========================================================
    # Corrección de un bloque
    # ========================================================

    with tabs[2]:
        st.header("Corrección de un bloque de 7 bits")

        st.markdown(
            """
En esta sección se analiza un único bloque Hamming de 7 bits. El estudiante ingresa una
palabra transmitida y selecciona una posición donde se inyectará un error.

La app realiza el proceso completo del receptor:

1. Recibe la palabra alterada.
2. Calcula las comprobaciones de paridad.
3. Forma el síndrome.
4. Convierte el síndrome a posición decimal.
5. Invierte el bit indicado.
6. Extrae los datos recuperados.

Este proceso representa la corrección de un único error por bloque (Hamming, 1950;
Lin & Costello, 2004).
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

            ejecutar = st.button("Inyectar error y corregir")

        with col_info:
            st.info(
                """
Si ocurre un solo error dentro del bloque, el síndrome debe coincidir con la posición
alterada.

Ejemplo:

Si el síndrome es 101, su valor decimal es 5. Entonces el receptor invierte el bit
de la posición 5.
"""
            )

        if not validar_codigo_hamming(palabra_tx):
            st.error("Debe ingresar exactamente 7 bits. Ejemplo válido: 0110011.")
        elif ejecutar:
            palabra_rx = invertir_bit(palabra_tx, int(posicion_error))
            resultado = corregir_hamming(palabra_rx)
            sindrome = resultado["sindrome"]

            tabla_sindrome = construir_tabla_sindrome(palabra_rx, sindrome)
            tabla_comparacion = construir_tabla_comparacion_un_error(
                palabra_tx,
                palabra_rx,
                resultado["corregido"],
            )

            st.subheader("Secuencias del proceso")

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

            st.subheader("1. Tabla de comprobaciones de paridad")

            st.markdown(
                """
Cada fila muestra una comprobación de paridad. Si el resultado es 1, significa que
ese grupo presenta una inconsistencia. La combinación de esas inconsistencias forma
el síndrome.
"""
            )

            st.dataframe(tabla_sindrome, use_container_width=True, hide_index=True)

            st.subheader("2. Comparación transmitido, recibido y corregido")

            st.markdown(
                """
Esta tabla permite ver qué bit cambió por el canal y qué bit modificó Hamming.
Para el caso de un único error, la palabra corregida debe coincidir con la transmitida.
"""
            )

            st.dataframe(tabla_comparacion, use_container_width=True, hide_index=True)

            if resultado["corregido"] == palabra_tx:
                st.success("La corrección fue exitosa. La palabra corregida coincide con la palabra transmitida.")
            else:
                st.error("La corrección no coincide con la palabra transmitida. Revise el procedimiento.")

            st.markdown(
                f"""
**Interpretación**

El síndrome calculado fue:

$$
S = ({sindrome["s4"]}{sindrome["s2"]}{sindrome["s1"]})_2 = {sindrome["posicion_error"]}
$$

Por tanto, bajo la hipótesis de un único error, el receptor identifica una alteración
en la posición {sindrome["posicion_error"]}.

Después de invertir esa posición, se extraen los datos desde las posiciones 3, 5, 6 y 7.
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
Una secuencia codificada con Hamming (7,4) debe dividirse en bloques de 7 bits.
Por esta razón, la longitud de la secuencia recibida debe ser múltiplo de 7.

En esta sección no se agregan ceros de relleno automáticamente. Si la longitud no es
múltiplo de 7, se considera una secuencia inválida para decodificación Hamming.
Esto es importante porque el relleno, si existió, fue agregado antes de codificar en
el transmisor. En recepción, el sistema debe recibir palabras completas de 7 bits.

El procesamiento por bloques permite aplicar el mismo procedimiento de síndrome a cada
palabra Hamming recibida (Lin & Costello, 2004).
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            secuencia_codificada = st.text_area(
                "Secuencia Hamming codificada",
                value="01100111110000",
                help="Debe contener únicamente 0 y 1, y tener longitud múltiplo de 7.",
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

            ejecutar_bloques = st.button("Inyectar errores y decodificar bloques")

        with col_info:
            st.info(
                """
Use 1 error por bloque para observar corrección exitosa.

Use 2 errores por bloque para observar el límite del código Hamming (7,4).
"""
            )

            if validar_bits(secuencia_codificada):
                st.metric("Longitud ingresada", len(secuencia_codificada))
                st.metric("¿Múltiplo de 7?", "Sí" if len(secuencia_codificada) % 7 == 0 else "No")

        if not validar_bits(secuencia_codificada):
            st.error("La secuencia debe contener únicamente 0 y 1.")
        elif len(secuencia_codificada) % 7 != 0:
            st.error(
                "La secuencia codificada debe tener longitud múltiplo de 7. "
                "Cada bloque Hamming (7,4) recibido contiene exactamente 7 bits."
            )
        elif ejecutar_bloques:
            secuencia_rx, tabla_inyeccion = inyectar_errores_en_bloques(
                bits_codificados=secuencia_codificada,
                cantidad_errores_por_bloque=int(cantidad_errores),
                semilla=int(semilla),
            )

            datos_recuperados, bloques_corregidos, tabla_decodificacion = decodificar_bloques_hamming(
                secuencia_rx
            )

            tabla_bloques = construir_tabla_bloques_visual(
                secuencia_codificada,
                secuencia_rx,
                bloques_corregidos,
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

            st.subheader("1. Visualización por bloques")

            st.markdown(
                """
Esta tabla muestra de forma separada cada bloque transmitido, recibido y corregido.
Permite comprobar si Hamming recuperó correctamente cada palabra de 7 bits.
"""
            )

            st.dataframe(tabla_bloques, use_container_width=True, hide_index=True)

            st.subheader("2. Errores inyectados por bloque")

            st.markdown(
                """
Esta tabla muestra qué posiciones fueron alteradas artificialmente dentro de cada bloque.
Con un error por bloque, Hamming debería corregir. Con dos errores por bloque, se observa
la limitación del código.
"""
            )

            st.dataframe(tabla_inyeccion, use_container_width=True, hide_index=True)

            st.subheader("3. Decodificación por síndrome")

            st.markdown(
                """
Esta tabla muestra el síndrome calculado para cada bloque recibido y la corrección
aplicada por el receptor.
"""
            )

            st.dataframe(tabla_decodificacion, use_container_width=True, hide_index=True)

            if cantidad_errores == 0:
                st.success("No se inyectaron errores. Los bloques deberían recuperarse sin corrección.")
            elif cantidad_errores == 1:
                st.success("Se inyectó un error por bloque. Hamming debería corregir cada bloque.")
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
                "tabla_bloques": tabla_bloques,
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
múltiples en el mismo bloque.

Esto no contradice el funcionamiento del código. Simplemente muestra su límite: Hamming
(7,4) está diseñado para corregir un error por bloque. Cuando hay dos errores, el síndrome
puede señalar una posición que no corresponde a un único error real. Por eso, después de
la corrección Hamming, puede ser necesario aplicar un mecanismo de detección adicional,
como CRC (Lin & Costello, 2004; Stallings, 2015).
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

            ejecutar_multi = st.button("Inyectar dos errores")

        with col_info:
            st.warning(
                """
Hamming (7,4) corrige un solo error por bloque.

Con dos errores, el síndrome puede señalar una posición que no corresponde al patrón real
de errores. En ese caso, la corrección puede empeorar la palabra recibida.
"""
            )

        if not validar_codigo_hamming(palabra_multi):
            st.error("Debe ingresar exactamente 7 bits.")
        elif pos_1 == pos_2:
            st.error("Seleccione dos posiciones distintas.")
        elif ejecutar_multi:
            posiciones_reales = [int(pos_1), int(pos_2)]

            palabra_rx = invertir_bit(palabra_multi, int(pos_1))
            palabra_rx = invertir_bit(palabra_rx, int(pos_2))

            resultado = corregir_hamming(palabra_rx)
            sindrome = resultado["sindrome"]

            tabla_sindrome = construir_tabla_sindrome(palabra_rx, sindrome)
            tabla_multiple = construir_tabla_errores_multiples(
                original=palabra_multi,
                recibido=palabra_rx,
                corregido=resultado["corregido"],
                posiciones_reales=posiciones_reales,
                posicion_sindrome=sindrome["posicion_error"],
            )

            st.subheader("Secuencias del caso")

            st.code(
                f"Palabra transmitida: {palabra_multi}\n"
                f"Errores reales en:   {posiciones_reales}\n"
                f"Palabra recibida:    {palabra_rx}\n"
                f"Síndrome:            {sindrome['sindrome_binario']}\n"
                f"Posición sugerida:   {sindrome['posicion_error']}\n"
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

            st.subheader("1. Tabla de comprobaciones")

            st.markdown(
                """
Esta tabla muestra cómo se formó el síndrome. Aunque el síndrome se calcula correctamente,
su interpretación como posición de error solo es confiable cuando hay un único error.
"""
            )

            st.dataframe(tabla_sindrome, use_container_width=True, hide_index=True)

            st.subheader("2. Tabla de errores múltiples")

            st.markdown(
                """
Esta tabla permite ver claramente:

- qué posiciones fueron alteradas realmente;
- qué posición indicó el síndrome;
- qué bit modificó Hamming;
- si la palabra final coincide o no con la transmitida.

Esta tabla corrige la interpretación anterior, porque ahora no solo se muestra la palabra
recibida y corregida, sino también la relación entre los errores reales y la posición
sugerida por el síndrome.
"""
            )

            st.dataframe(tabla_multiple, use_container_width=True, hide_index=True)

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

Cuando ocurren dos errores dentro del mismo bloque, el síndrome deja de representar
de forma confiable una posición real de error único. El receptor puede invertir una
posición incorrecta y producir una palabra final diferente a la transmitida.

Esta limitación justifica la integración posterior de CRC como mecanismo de detección
de errores remanentes.
"""
            )

            st.session_state["guia_04_multi_resultado"] = {
                "tx": palabra_multi,
                "rx": palabra_rx,
                "corregido": resultado["corregido"],
                "sindrome": sindrome["sindrome_binario"],
                "posicion": sindrome["posicion_error"],
                "posiciones_reales": posiciones_reales,
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
es comprobar que el estudiante comprende el significado del síndrome, la decodificación
por bloques y los límites del código Hamming.
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
                            "Bloques recuperados correctamente": resultado_bloques["bloques_correctos"],
                            "Datos recuperados": resultado_bloques["datos"],
                        }
                    ]
                )
            )

            st.subheader("Bloques transmitidos, recibidos y corregidos")

            st.markdown(
                """
Esta tabla responde a la observación de mostrar los bloques dentro de la sección de
análisis. Aquí se visualiza cada bloque de 7 bits antes del canal, después del canal
y después de la corrección.
"""
            )

            st.dataframe(
                resultado_bloques["tabla_bloques"],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Ejecute una decodificación por bloques para ver el resumen.")

        if "guia_04_multi_resultado" in st.session_state:
            st.subheader("Último caso de errores múltiples")

            r = st.session_state["guia_04_multi_resultado"]

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Transmitido": r["tx"],
                            "Recibido": r["rx"],
                            "Corregido": r["corregido"],
                            "Errores reales": r["posiciones_reales"],
                            "Síndrome": r["sindrome"],
                            "Posición sugerida": r["posicion"],
                            "Coincide con original": r["coincide"],
                        }
                    ]
                )
            )
        else:
            st.info("Ejecute un caso de errores múltiples para analizar el límite del código.")

        st.subheader("Actividades guiadas")

        st.markdown(
            """
Realice las siguientes actividades:

1. Use la palabra Hamming `0110011`.
2. Inyecte un error en la posición 1 y registre el síndrome.
3. Repita para las posiciones 2, 3, 4, 5, 6 y 7.
4. Verifique que el síndrome coincide con la posición del error cuando solo hay un error.
5. Decodifique una secuencia de dos bloques de 7 bits.
6. Inyecte un error por bloque y observe la corrección.
7. Inyecte dos errores por bloque y observe el límite del código.
8. Explique por qué en recepción la secuencia debe ser múltiplo de 7.
9. Explique por qué CRC será necesario en la siguiente etapa.
"""
        )

        st.subheader("Preguntas de análisis")

        pregunta_1 = st.radio(
            "Pregunta 1: ¿Qué indica un síndrome distinto de cero en Hamming (7,4)?",
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
                st.success("Correcto. El síndrome indica la posición del error cuando se asume un único error.")
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
                st.success("Correcto. Un síndrome cero indica que no se detecta error bajo las comprobaciones de paridad.")
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

        pregunta_6 = st.radio(
            "Pregunta 6: ¿Por qué la secuencia recibida debe ser múltiplo de 7?",
            [
                "Porque cada palabra Hamming (7,4) recibida contiene exactamente 7 bits.",
                "Porque CRC siempre tiene 7 bits.",
                "Porque todos los mensajes digitales tienen 7 bits.",
                "Porque la semilla siempre genera bloques de 7.",
            ],
            index=None,
            key="g4_pregunta_6",
        )

        if pregunta_6:
            if pregunta_6 == "Porque cada palabra Hamming (7,4) recibida contiene exactamente 7 bits.":
                st.success("Correcto. En recepción se decodifican palabras Hamming completas de 7 bits.")
            else:
                st.error("Revise la estructura de Hamming (7,4).")

    # ========================================================
    # Conclusiones
    # ========================================================

    with tabs[6]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- El receptor calcula un síndrome a partir de verificaciones de paridad;
- El término técnico correcto en español es síndrome;
- El síndrome se interpreta como la posición del error bajo la hipótesis de un único error;
- La corrección se realiza invirtiendo el bit indicado por el síndrome;
- Hamming (7,4) permite recuperar los 4 bits de datos si ocurre un solo error en el bloque;
- Las secuencias largas se decodifican dividiéndolas en bloques de 7 bits;
- En recepción no se agregan bits de relleno automáticamente;
- Si la secuencia no es múltiplo de 7, no puede dividirse correctamente en palabras Hamming;
- Hamming corrige un error por bloque, no errores múltiples dentro del mismo bloque;
- Cuando ocurren errores múltiples, el síndrome puede inducir una corrección incorrecta;
- Esta limitación justifica la integración posterior de CRC como detector de errores remanentes.

La teoría aplicada en esta guía se fundamenta en el código Hamming original, en la teoría
de códigos de control de errores y en los principios de detección y corrección de errores
en comunicaciones digitales (Hamming, 1950; Lin & Costello, 2004; Forouzan, 2013; Stallings, 2015).
"""
        )

    # ========================================================
    # Referencias
    # ========================================================

    with tabs[7]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2013). *Data communications and networking* (5th ed.). McGraw-Hill Education.

Hamming, R. W. (1950). Error detecting and error correcting codes. *The Bell System Technical Journal, 29*(2), 147–160. https://doi.org/10.1002/j.1538-7305.1950.tb00463.x

Lin, S., & Costello, D. J. (2004). *Error control coding* (2nd ed.). Pearson.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.
"""
        )