import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Funciones base Hamming (7,4)
# ============================================================

def validar_bits(bits: str) -> bool:
    """
    Valida que una cadena contenga únicamente bits 0 y 1.
    """
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_bits_4(bits: str) -> bool:
    """
    Valida que una cadena sea una secuencia binaria de exactamente 4 bits.
    """
    return len(bits) == 4 and all(bit in "01" for bit in bits)


def rellenar_a_multiplo(bits: str, multiplo: int) -> tuple[str, int]:
    """
    Rellena una secuencia binaria con ceros hasta que su longitud sea múltiplo
    del tamaño indicado.

    Devuelve:
    - secuencia con relleno;
    - cantidad de bits de relleno agregados.

    Importante:
    Los bits de relleno no son bits de paridad. Los bits de relleno solo se agregan
    cuando el último bloque no tiene la longitud necesaria para aplicar Hamming (7,4).
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
    Calcula los bits de paridad para el código Hamming (7,4) con paridad par.

    Estructura usada en esta guía:

    Posición: 1  2  3  4  5  6  7
    Tipo:     P1 P2 D1 P4 D2 D3 D4

    D1 se coloca en la posición 3.
    D2 se coloca en la posición 5.
    D3 se coloca en la posición 6.
    D4 se coloca en la posición 7.

    Con paridad par:

    P1 = D1 XOR D2 XOR D4
    P2 = D1 XOR D3 XOR D4
    P4 = D2 XOR D3 XOR D4
    """
    d1, d2, d3, d4 = [int(bit) for bit in bits]

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


def calcular_eficiencia(k: int, n: int) -> float:
    """
    Calcula la eficiencia o tasa de código.

    η = k / n

    k: bits de datos.
    n: bits totales transmitidos.
    """
    return k / n


# ============================================================
# Tablas didácticas
# ============================================================

def construir_tabla_estructura_vacia() -> pd.DataFrame:
    """
    Tabla que muestra la estructura fija de posiciones de Hamming (7,4).
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
            "¿Potencia de 2?": ["Sí", "Sí", "No", "Sí", "No", "No", "No"],
            "Descripción": [
                "Bit de paridad ubicado en la posición 1",
                "Bit de paridad ubicado en la posición 2",
                "Primer bit de dato",
                "Bit de paridad ubicado en la posición 4",
                "Segundo bit de dato",
                "Tercer bit de dato",
                "Cuarto bit de dato",
            ],
        }
    )


def construir_tabla_posiciones(resultado: dict) -> pd.DataFrame:
    """
    Tabla que muestra cómo queda la palabra Hamming después de calcular
    las paridades.
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
            "Interpretación": [
                "Paridad calculada para posiciones 1, 3, 5 y 7",
                "Paridad calculada para posiciones 2, 3, 6 y 7",
                "Dato original D1",
                "Paridad calculada para posiciones 4, 5, 6 y 7",
                "Dato original D2",
                "Dato original D3",
                "Dato original D4",
            ],
        }
    )


def construir_tabla_cobertura(resultado: dict) -> pd.DataFrame:
    """
    Tabla que explica qué datos participan en cada bit de paridad.
    """
    return pd.DataFrame(
        {
            "Bit de paridad": ["P1", "P2", "P4"],
            "Posiciones cubiertas": ["1, 3, 5, 7", "2, 3, 6, 7", "4, 5, 6, 7"],
            "Datos que intervienen": ["D1, D2, D4", "D1, D3, D4", "D2, D3, D4"],
            "Operación XOR": [
                f"{resultado['D1']} ⊕ {resultado['D2']} ⊕ {resultado['D4']}",
                f"{resultado['D1']} ⊕ {resultado['D3']} ⊕ {resultado['D4']}",
                f"{resultado['D2']} ⊕ {resultado['D3']} ⊕ {resultado['D4']}",
            ],
            "Resultado": [resultado["P1"], resultado["P2"], resultado["P4"]],
            "Finalidad": [
                "Asegurar paridad par en el grupo revisado por P1",
                "Asegurar paridad par en el grupo revisado por P2",
                "Asegurar paridad par en el grupo revisado por P4",
            ],
        }
    )


def construir_tabla_diferencia_relleno_paridad() -> pd.DataFrame:
    """
    Tabla para aclarar la diferencia entre bits de paridad y bits de relleno.
    """
    return pd.DataFrame(
        {
            "Concepto": ["Bits de paridad", "Bits de relleno"],
            "¿Quién los genera?": [
                "El código Hamming mediante operaciones XOR",
                "La app los agrega solo si el último bloque no tiene 4 bits",
            ],
            "¿Para qué sirven?": [
                "Permiten que el receptor detecte y corrija errores simples",
                "Permiten completar el último bloque de 4 bits antes de codificar",
            ],
            "¿Forman parte de la redundancia de Hamming?": [
                "Sí",
                "No, solo completan la longitud del bloque",
            ],
            "Ejemplo": [
                "P1, P2 y P4",
                "Si quedan 2 bits incompletos, se agregan 2 ceros",
            ],
        }
    )


def construir_tabla_elementos_matriciales() -> pd.DataFrame:
    """
    Tabla explicativa de los elementos usados en el método matricial.
    """
    return pd.DataFrame(
        {
            "Elemento": ["m", "G", "c", "mod 2"],
            "Significado": [
                "Vector de datos",
                "Matriz generadora fija",
                "Palabra Hamming codificada",
                "Operación binaria módulo 2",
            ],
            "Descripción": [
                "Contiene los bits [D1 D2 D3 D4] ingresados por el estudiante",
                "Representa la regla de codificación del código Hamming (7,4)",
                "Contiene los bits [P1 P2 D1 P4 D2 D3 D4]",
                "Hace que las sumas se calculen como XOR",
            ],
            "Referencia teórica": [
                "Lin & Costello (2004)",
                "Lin & Costello (2004)",
                "Hamming (1950); Lin & Costello (2004)",
                "Forouzan (2013); Stallings (2015)",
            ],
        }
    )


def construir_tabla_columnas_matriz() -> pd.DataFrame:
    """
    Explica qué representa cada columna de la matriz generadora.
    """
    return pd.DataFrame(
        {
            "Columna": ["P1", "P2", "D1", "P4", "D2", "D3", "D4"],
            "Qué calcula o coloca": [
                "D1 ⊕ D2 ⊕ D4",
                "D1 ⊕ D3 ⊕ D4",
                "D1",
                "D2 ⊕ D3 ⊕ D4",
                "D2",
                "D3",
                "D4",
            ],
            "Interpretación": [
                "Primer bit de paridad",
                "Segundo bit de paridad",
                "Primer bit de dato ubicado en la posición 3",
                "Tercer bit de paridad ubicado en la posición 4",
                "Segundo bit de dato ubicado en la posición 5",
                "Tercer bit de dato ubicado en la posición 6",
                "Cuarto bit de dato ubicado en la posición 7",
            ],
        }
    )


def construir_tabla_filas_matriz() -> pd.DataFrame:
    """
    Explica cómo se interpreta cada fila de la matriz generadora.
    """
    return pd.DataFrame(
        {
            "Fila": ["D1", "D2", "D3", "D4"],
            "Fila de G": [
                "[1 1 1 0 0 0 0]",
                "[1 0 0 1 1 0 0]",
                "[0 1 0 1 0 1 0]",
                "[1 1 0 1 0 0 1]",
            ],
            "Interpretación": [
                "D1 participa en P1, P2 y en la posición D1",
                "D2 participa en P1, P4 y en la posición D2",
                "D3 participa en P2, P4 y en la posición D3",
                "D4 participa en P1, P2, P4 y en la posición D4",
            ],
        }
    )


# ============================================================
# Matriz generadora Hamming (7,4)
# ============================================================

def matriz_generadora_hamming_7_4() -> np.ndarray:
    """
    Matriz generadora fija para la estructura:

    [P1 P2 D1 P4 D2 D3 D4]

    Vector de entrada:

    m = [D1 D2 D3 D4]

    La matriz G no cambia con el mensaje. Lo que cambia es el vector m.
    """
    return np.array(
        [
            [1, 1, 1, 0, 0, 0, 0],  # D1 participa en P1, P2 y D1
            [1, 0, 0, 1, 1, 0, 0],  # D2 participa en P1, P4 y D2
            [0, 1, 0, 1, 0, 1, 0],  # D3 participa en P2, P4 y D3
            [1, 1, 0, 1, 0, 0, 1],  # D4 participa en P1, P2, P4 y D4
        ],
        dtype=int,
    )


def codificar_con_matriz(bits: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Codifica el mensaje usando la matriz generadora.

    c = mG mod 2
    """
    m = np.array([int(bit) for bit in bits], dtype=int)
    G = matriz_generadora_hamming_7_4()
    c = (m @ G) % 2
    return m, G, c


def dataframe_matriz(G: np.ndarray) -> pd.DataFrame:
    """
    Convierte la matriz generadora en un DataFrame para mostrarla en la app.
    """
    return pd.DataFrame(
        G,
        index=["D1", "D2", "D3", "D4"],
        columns=["P1", "P2", "D1", "P4", "D2", "D3", "D4"],
    )


# ============================================================
# Codificación por bloques
# ============================================================

def codificar_mensaje_por_bloques(bits: str) -> tuple[str, int, pd.DataFrame]:
    """
    Codifica una secuencia de cualquier longitud usando Hamming (7,4).

    La app no recalcula un nuevo código Hamming para toda la longitud del mensaje.
    Se mantiene Hamming (7,4) y se divide el mensaje en bloques de 4 bits.

    Si el último bloque no tiene 4 bits, se agregan ceros de relleno.
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
                "Datos del bloque": bloque,
                "D1": resultado["D1"],
                "D2": resultado["D2"],
                "D3": resultado["D3"],
                "D4": resultado["D4"],
                "P1": resultado["P1"],
                "P2": resultado["P2"],
                "P4": resultado["P4"],
                "Palabra Hamming (7 bits)": codigo,
            }
        )

    codigo_total = "".join(codigos)
    return codigo_total, padding, pd.DataFrame(filas)


# ============================================================
# Interfaz Streamlit
# ============================================================

def render_guia_03() -> None:
    st.title("Guía 3: Codificación Hamming (7,4) en el transmisor")

    st.markdown(
        """
Esta guía introduce la codificación Hamming (7,4) desde el punto de vista del transmisor.
Después de estudiar cómo el ruido puede provocar errores en una señal digital, ahora se
analiza cómo el transmisor puede agregar redundancia antes de enviar los datos.

El propósito principal es que el estudiante comprenda que Hamming no elimina el ruido del
canal, sino que agrega bits calculados de forma estructurada para que el receptor pueda
detectar y corregir errores simples en una etapa posterior. Esta idea pertenece al campo
del control de errores en comunicaciones digitales, donde se agregan bits redundantes para
aumentar la confiabilidad de la transmisión (Hamming, 1950; Lin & Costello, 2004).
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
6. Diferenciar bits de paridad y bits de relleno.
7. Calcular la eficiencia de codificación.
8. Comparar el método por posiciones con el método matricial.
9. Comprender que la matriz generadora es fija para el código Hamming definido.
"""
        )

    # ========================================================
    # Teoría
    # ========================================================

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En un sistema de comunicación digital, el canal puede introducir errores debido al ruido,
interferencias, atenuación o imperfecciones del medio de transmisión. Una forma de aumentar
la confiabilidad de la transmisión es agregar redundancia controlada. Esa redundancia permite
que el receptor tenga información adicional para detectar o corregir errores (Forouzan, 2013;
Stallings, 2015).

El código Hamming es una técnica de corrección de errores hacia adelante. Se llama así porque
el transmisor agrega bits adicionales antes de enviar la información, sin esperar a que el
receptor solicite una retransmisión. Estos bits adicionales no son arbitrarios: se calculan
a partir de los datos mediante operaciones de paridad (Hamming, 1950; Lin & Costello, 2004).

En esta guía se usa el código **Hamming (7,4)**:

- $k = 4$ bits de datos;
- $p = 3$ bits de paridad;
- $n = 7$ bits totales.

La relación general es:

$$
n = k + p
$$

Para determinar cuántos bits de paridad se necesitan se utiliza la condición:

$$
2^p \\geq k + p + 1
$$

donde:

- $k$ es la cantidad de bits de datos;
- $p$ es la cantidad de bits de paridad;
- $k+p$ es la cantidad total de posiciones de la palabra codificada;
- el término $+1$ representa el caso en el que no hay error.

El valor $p$ aparece en ambos lados porque los bits de paridad también ocupan posiciones
dentro de la palabra Hamming. Por eso no se resuelve como una ecuación algebraica común,
sino probando valores enteros de $p$ hasta encontrar el menor que cumple. Esta condición
está asociada a la capacidad del síndrome para representar todas las posiciones posibles
de error y el caso sin error (Hamming, 1950; Lin & Costello, 2004).

Para $k = 4$:

$$
p = 2: \\quad 2^2 \\geq 4 + 2 + 1 \\Rightarrow 4 \\geq 7
$$

No cumple.

$$
p = 3: \\quad 2^3 \\geq 4 + 3 + 1 \\Rightarrow 8 \\geq 8
$$

Sí cumple.

Por tanto:

$$
p = 3
$$

y:

$$
n = k + p = 4 + 3 = 7
$$

Por eso se usa Hamming (7,4).
"""
        )

        st.subheader("Estructura de la palabra Hamming (7,4)")

        st.markdown(
            """
Los bits de paridad se colocan en posiciones que son potencias de dos:

$$
1, 2, 4
$$

Los bits de datos se colocan en las posiciones restantes:

$$
3, 5, 6, 7
$$

En esta app se usa la estructura:

| Posición | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Tipo | P1 | P2 | D1 | P4 | D2 | D3 | D4 |

Esta estructura se mantiene en toda la guía y también en la Guía 4, donde el receptor
calcula el síndrome para corregir errores. La ubicación de los bits de paridad en
posiciones potencia de dos permite que cada posición del bloque tenga una combinación
única de comprobaciones de paridad (Hamming, 1950; Lin & Costello, 2004).
"""
        )

        st.dataframe(
            construir_tabla_estructura_vacia(),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Paridad par y operación XOR")

        st.markdown(
            """
El código implementado utiliza **paridad par**. Esto significa que cada grupo protegido
por un bit de paridad debe tener una cantidad par de unos. Si el grupo no cumple esa
condición, el receptor podrá detectar una inconsistencia al calcular el síndrome
(Forouzan, 2013; Stallings, 2015).

Las paridades usadas en esta guía son:

$$
P1 = D1 \\oplus D2 \\oplus D4
$$

$$
P2 = D1 \\oplus D3 \\oplus D4
$$

$$
P4 = D2 \\oplus D3 \\oplus D4
$$

El símbolo $\\oplus$ representa la operación XOR. En binario, XOR produce 1 cuando los
bits comparados son diferentes y produce 0 cuando son iguales.

Ejemplos de XOR:

- $0 \\oplus 0 = 0$
- $0 \\oplus 1 = 1$
- $1 \\oplus 0 = 1$
- $1 \\oplus 1 = 0$

Por esa razón, XOR es útil para calcular paridades: permite saber si un grupo de bits
tiene una cantidad par o impar de unos. Este principio es ampliamente utilizado en
mecanismos de detección y corrección de errores en comunicaciones digitales (Forouzan,
2013; Lin & Costello, 2004).
"""
        )

        st.subheader("Eficiencia de codificación")

        st.markdown(
            """
La eficiencia o tasa de código permite medir qué proporción de los bits transmitidos
son información útil. Se define como:

$$
\\eta = \\frac{k}{n}
$$

donde:

- $k$ es la cantidad de bits de datos;
- $n$ es la cantidad total de bits transmitidos.

Para Hamming (7,4):

$$
\\eta = \\frac{4}{7} \\approx 0.5714
$$

Esto significa que aproximadamente el 57.14% de los bits transmitidos son datos originales
y el resto corresponde a redundancia de paridad.

Esta redundancia aumenta la cantidad de bits transmitidos, pero permite que el receptor
tenga información adicional para detectar y corregir errores simples. En codificación de
canal, este intercambio entre redundancia y confiabilidad es un concepto fundamental
(Lin & Costello, 2004; Stallings, 2015).
"""
        )

        eficiencia = calcular_eficiencia(4, 7)

        c1, c2, c3 = st.columns(3)
        c1.metric("Bits de datos k", "4")
        c2.metric("Bits totales n", "7")
        c3.metric("Eficiencia η", f"{eficiencia:.4f}")

        st.subheader("¿Qué pasaría con un mensaje de 9 bits?")

        st.markdown(
            """
Es importante distinguir entre dos ideas:

**1. Diseñar un nuevo código para un bloque de 9 bits**

Si se quisiera diseñar un bloque Hamming para $k = 9$ bits de datos, se aplica:

$$
2^p \\geq k + p + 1
$$

Probando con $p = 3$:

$$
2^3 \\geq 9 + 3 + 1 \\Rightarrow 8 \\geq 13
$$

No cumple.

Probando con $p = 4$:

$$
2^4 \\geq 9 + 4 + 1 \\Rightarrow 16 \\geq 14
$$

Sí cumple.

Entonces, para proteger 9 bits como un solo bloque, se necesitarían 4 bits de paridad
y se tendría una palabra de 13 bits.

**2. Lo que hace esta app**

Esta app no diseña un nuevo código para cada longitud de mensaje. La app mantiene fijo
el código Hamming (7,4). Por eso, si el mensaje tiene 9 bits, se divide en bloques de 4:

- 9 bits originales;
- se agregan 3 bits de relleno;
- quedan 12 bits procesados;
- se forman 3 bloques de 4 bits;
- cada bloque produce 7 bits codificados;
- se transmiten 21 bits codificados.

Por tanto, en esta app Hamming siempre trabaja por bloques de 4 bits. Esta forma de
trabajo por bloques es coherente con el tratamiento de los códigos de bloque lineales,
donde una longitud fija de entrada produce una longitud fija de salida (Lin & Costello,
2004).
"""
        )

    # ========================================================
    # Codificación de un bloque
    # ========================================================

    with tabs[2]:
        st.header("Codificación de un bloque de 4 bits")

        st.markdown(
            """
En esta sección se codifica un único bloque de 4 bits. El objetivo es observar paso a
paso cómo se calculan las paridades antes de formar la palabra Hamming final.

La secuencia lógica es:

1. Identificar los bits de datos $D1$, $D2$, $D3$ y $D4$.
2. Calcular $P1$, $P2$ y $P4$ usando paridad par.
3. Colocar datos y paridades en la estructura fija.
4. Obtener la palabra Hamming de 7 bits.

Esto es importante porque el estudiante no debe ver la palabra Hamming como un resultado
automático sin explicación. Primero debe comprender de dónde salen los bits de paridad.
El procedimiento corresponde al uso de redundancia estructurada para control de errores
(Hamming, 1950; Forouzan, 2013).
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

            ejecutar = st.button("Codificar bloque con Hamming (7,4)")

        with col_info:
            st.info(
                """
La entrada se interpreta como:

m = [D1 D2 D3 D4]

Por ejemplo, si el mensaje es 1011, entonces:

D1 = 1  
D2 = 0  
D3 = 1  
D4 = 1
"""
            )

        if not validar_bits_4(bits):
            st.error("Debe ingresar exactamente 4 bits. Ejemplo válido: 1011.")
        elif ejecutar:
            resultado = calcular_paridades_hamming_7_4(bits)
            tabla_posiciones = construir_tabla_posiciones(resultado)
            tabla_cobertura = construir_tabla_cobertura(resultado)

            st.subheader("1. Identificación de los bits de datos")

            st.markdown(
                f"""
El mensaje ingresado es:

**{bits}**

Por la estructura del código Hamming (7,4), se interpreta como:

$$
m = [D1 \\quad D2 \\quad D3 \\quad D4]
$$

Por tanto:

$$
D1 = {resultado['D1']}
$$

$$
D2 = {resultado['D2']}
$$

$$
D3 = {resultado['D3']}
$$

$$
D4 = {resultado['D4']}
$$
"""
            )

            st.subheader("2. Cálculos realizados con paridad par")

            st.markdown(
                f"""
El código Hamming utilizado trabaja con **paridad par**. Esto significa que cada grupo
revisado por un bit de paridad debe contener una cantidad par de unos.

Los bits de paridad se calculan mediante operaciones XOR:

$$
P1 = D1 \\oplus D2 \\oplus D4
$$

Sustituyendo:

$$
P1 = {resultado['D1']} \\oplus {resultado['D2']} \\oplus {resultado['D4']} = {resultado['P1']}
$$

$$
P2 = D1 \\oplus D3 \\oplus D4
$$

Sustituyendo:

$$
P2 = {resultado['D1']} \\oplus {resultado['D3']} \\oplus {resultado['D4']} = {resultado['P2']}
$$

$$
P4 = D2 \\oplus D3 \\oplus D4
$$

Sustituyendo:

$$
P4 = {resultado['D2']} \\oplus {resultado['D3']} \\oplus {resultado['D4']} = {resultado['P4']}
$$

Estos tres bits de paridad no son elegidos manualmente. Son el resultado de aplicar
las reglas de paridad del código Hamming. Este uso de paridad es parte fundamental
de los códigos de control de errores (Hamming, 1950; Lin & Costello, 2004).
"""
            )

            st.subheader("3. Cobertura de cada bit de paridad")

            st.markdown(
                """
Esta tabla muestra qué posiciones y qué bits de datos intervienen en cada paridad.

La columna de operación XOR muestra el cálculo realizado. La finalidad de cada paridad
es que, en el receptor, sea posible detectar si una posición del bloque fue alterada.
"""
            )

            st.dataframe(tabla_cobertura, use_container_width=True, hide_index=True)

            st.subheader("4. Ubicación de datos y paridades")

            st.markdown(
                """
Una vez calculados los bits de paridad, se colocan junto con los bits de datos en la
estructura fija del código:

| Posición | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Tipo | P1 | P2 | D1 | P4 | D2 | D3 | D4 |

Esta ubicación es importante porque el receptor usará la misma estructura para calcular
el síndrome en la Guía 4.
"""
            )

            st.dataframe(tabla_posiciones, use_container_width=True, hide_index=True)

            st.subheader("5. Resultado de la codificación")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits de datos k", "4")
            c2.metric("Bits de paridad p", "3")
            c3.metric("Longitud total n", "7")
            c4.metric("Eficiencia η", f"{calcular_eficiencia(4, 7):.4f}")

            st.code(
                f"Mensaje original: {bits}\n"
                f"Palabra Hamming:  {resultado['codigo_str']}",
                language="text",
            )

            st.success(
                "La palabra Hamming final se obtiene colocando los bits en el orden [P1 P2 D1 P4 D2 D3 D4]."
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
Hamming (7,4) trabaja con bloques de 4 bits de datos. Esto significa que, aunque el
mensaje completo tenga 8, 9, 16 o más bits, la app no recalcula una nueva matriz ni un
nuevo código para toda la longitud. Lo que hace es dividir el mensaje en bloques de
4 bits y aplicar Hamming (7,4) a cada bloque.

Ejemplos:

- 4 bits de datos → 1 bloque → 7 bits codificados.
- 8 bits de datos → 2 bloques → 14 bits codificados.
- 16 bits de datos → 4 bloques → 28 bits codificados.

Si el mensaje no tiene longitud múltiplo de 4, se agregan ceros de relleno al final
para completar el último bloque. El procesamiento por bloques es característico de los
códigos de bloque lineales, donde una cantidad fija de bits de entrada produce una
cantidad fija de bits codificados (Lin & Costello, 2004).
"""
        )

        st.info(
            """
Diferencia importante:

Los bits de paridad son calculados por Hamming y sirven para corrección de errores.
Los bits de relleno solo completan el último bloque cuando faltan datos. No son bits
de paridad.
"""
        )

        st.subheader("Diferencia entre bits de paridad y bits de relleno")
        st.dataframe(
            construir_tabla_diferencia_relleno_paridad(),
            use_container_width=True,
            hide_index=True,
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
                    value="101100111",
                    help="Puede ingresar 4, 8, 9, 16 o más bits.",
                    key="g3_bits_largos_manual",
                ).strip()
            else:
                cantidad_bits = st.selectbox(
                    "Cantidad de bits de datos",
                    [4, 8, 9, 16, 32],
                    index=3,
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

            ejecutar_bloques = st.button("Codificar mensaje por bloques")

        with col_info:
            st.info(
                """
Cada bloque de 4 bits genera 3 bits de paridad y produce una palabra Hamming de 7 bits.

La tasa de código por bloque se mantiene:

η = 4/7

Cuando hay bits de relleno, la eficiencia efectiva respecto al mensaje original puede
ser menor, porque se transmiten bits adicionales que solo completan el bloque.
"""
            )

        if not validar_bits(bits_largos):
            st.error("El mensaje debe contener únicamente 0 y 1.")
        elif ejecutar_bloques:
            codigo_total, padding, tabla_bloques = codificar_mensaje_por_bloques(bits_largos)

            bits_rellenados, _ = rellenar_a_multiplo(bits_largos, 4)
            bloques = dividir_en_bloques(bits_rellenados, 4)

            cantidad_bloques = len(bloques)
            bits_codificados = len(codigo_total)
            bits_paridad_generados = cantidad_bloques * 3

            tasa_codigo = calcular_eficiencia(4, 7)
            eficiencia_efectiva = len(bits_largos) / bits_codificados if bits_codificados else 0

            st.subheader("Resumen de codificación por bloques")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits originales", len(bits_largos))
            c2.metric("Bits de relleno", padding)
            c3.metric("Bloques de 4 bits", cantidad_bloques)
            c4.metric("Bits codificados", bits_codificados)

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Bits de paridad generados", bits_paridad_generados)
            c6.metric("Tasa por bloque η", f"{tasa_codigo:.4f}")
            c7.metric("Eficiencia efectiva", f"{eficiencia_efectiva:.4f}")
            c8.metric("Redundancia efectiva", f"{1 - eficiencia_efectiva:.4f}")

            st.markdown("**Interpretación del caso**")
            st.code(
                f"Mensaje original:          {bits_largos}\n"
                f"Mensaje con relleno:       {bits_rellenados}\n"
                f"Bloques procesados:        {' | '.join(bloques)}\n"
                f"Palabra codificada total:  {codigo_total}",
                language="text",
            )

            if padding > 0:
                st.warning(
                    f"Se agregaron {padding} bits de relleno para completar el último bloque. "
                    "Estos bits no son paridad; solo permiten aplicar Hamming (7,4) al último bloque."
                )
            else:
                st.success(
                    "No fue necesario agregar bits de relleno porque la longitud del mensaje ya era múltiplo de 4."
                )
            st.info(
                f"""
**Definición de tasa, eficiencia y redundancia en la codificación por bloques**

En la codificación Hamming (7,4), cada bloque de entrada está formado por **k = 4 bits de datos**
y genera una palabra codificada de **n = 7 bits**. Los tres bits adicionales corresponden a bits
de paridad, los cuales agregan redundancia estructurada para que el receptor pueda detectar y
corregir errores simples.

La **tasa por bloque η** se define como la relación entre la cantidad de bits de datos útiles por
bloque y la cantidad total de bits codificados por bloque:

**η = k/n = 4/7 = {tasa_codigo:.4f}**

Este valor indica que, en cada palabra codificada de siete bits, cuatro bits corresponden a
información original y tres bits corresponden a redundancia de paridad. Su utilidad es mostrar
la eficiencia ideal del código Hamming (7,4), sin considerar todavía los bits de relleno.

La **eficiencia efectiva** considera el mensaje completo que se desea transmitir. Se calcula como:

**Eficiencia efectiva = bits originales / bits codificados**

En este caso:

**Eficiencia efectiva = {len(bits_largos)} / {bits_codificados} = {eficiencia_efectiva:.4f}**

Este valor puede ser menor que la tasa por bloque cuando el mensaje original no es múltiplo de
cuatro, porque la app agrega bits de relleno para completar el último bloque. Aunque estos bits
no son paridad, sí ocupan espacio dentro de la transmisión codificada.

La **redundancia efectiva** representa la fracción de la transmisión codificada que no corresponde
directamente a bits originales del mensaje. Se calcula como:

**Redundancia efectiva = 1 - eficiencia efectiva**

Para este caso:

**Redundancia efectiva = 1 - {eficiencia_efectiva:.4f} = {1 - eficiencia_efectiva:.4f}**

Esta redundancia incluye los bits de paridad generados por Hamming y, cuando existen, los bits de
relleno agregados para completar el último bloque. Su utilidad es medir el costo de transmisión
asociado al uso del código corrector.

En conjunto, estos parámetros permiten analizar el compromiso entre confiabilidad y eficiencia.
La tasa por bloque describe la eficiencia ideal de Hamming (7,4), mientras que la eficiencia
efectiva y la redundancia efectiva describen el comportamiento real del mensaje procesado por
la plataforma.
"""
            )
            st.subheader("Tabla de bloques codificados")

            st.dataframe(tabla_bloques, use_container_width=True, hide_index=True)

            st.session_state["guia_03_bloques_resultado"] = {
                "mensaje_original": bits_largos,
                "mensaje_rellenado": bits_rellenados,
                "codigo_total": codigo_total,
                "padding": padding,
                "bloques": cantidad_bloques,
                "bits_paridad": bits_paridad_generados,
                "tasa_codigo": tasa_codigo,
                "eficiencia_efectiva": eficiencia_efectiva,
            }

    # ========================================================
    # Método matricial
    # ========================================================

    with tabs[4]:
        st.header("Método matricial")

        st.markdown(
            """
El método matricial es una forma compacta de representar la misma codificación realizada
por posiciones y paridades.

En el método por posiciones se calculan $P1$, $P2$ y $P4$ usando XOR. En el método
matricial, esos mismos cálculos se agrupan dentro de una matriz generadora. Esta forma
de representación es común en el estudio de códigos lineales de bloque (Lin & Costello,
2004).

La operación principal es:

$$
c = mG \\mod 2
$$

donde:

- $m$ es el vector de datos;
- $G$ es la matriz generadora;
- $c$ es la palabra Hamming codificada;
- $\\mod 2$ indica que las operaciones se realizan en aritmética binaria, equivalente a XOR.

Para esta guía:

$$
m = [D1 \\quad D2 \\quad D3 \\quad D4]
$$

y la palabra resultante es:

$$
c = [P1 \\quad P2 \\quad D1 \\quad P4 \\quad D2 \\quad D3 \\quad D4]
$$
"""
        )

        st.info(
            """
La matriz generadora G no cambia cuando cambia el mensaje.

G pertenece al código Hamming (7,4) definido en esta guía. Representa la regla fija de
codificación. Lo que cambia en cada ejercicio es el vector m, es decir, los datos que
el estudiante ingresa.
"""
        )

        st.subheader("Elementos del método matricial")

        st.dataframe(
            construir_tabla_elementos_matriciales(),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Matriz generadora fija")

        G_fija = matriz_generadora_hamming_7_4()
        st.dataframe(dataframe_matriz(G_fija), use_container_width=True)

        st.markdown(
            """
La matriz anterior se lee por filas y columnas:

- Las filas representan los bits de datos $D1$, $D2$, $D3$ y $D4$.
- Las columnas representan las posiciones de salida $P1$, $P2$, $D1$, $P4$, $D2$, $D3$ y $D4$.
- Un valor 1 indica que ese bit de dato participa en esa posición de salida.
- Un valor 0 indica que no participa.

La matriz es fija porque pertenece a la estructura del código Hamming (7,4) que se está
usando. Si se mantuviera otra estructura de salida, entonces la matriz sería diferente.
Pero mientras se mantenga la estructura [P1 P2 D1 P4 D2 D3 D4], la matriz generadora
permanece igual.

Esta forma de representación mediante matriz generadora permite describir la codificación
como una operación algebraica sobre vectores binarios (Lin & Costello, 2004).
"""
        )

        st.subheader("Interpretación de las filas de G")

        st.dataframe(
            construir_tabla_filas_matriz(),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Interpretación de las columnas de G")

        st.dataframe(
            construir_tabla_columnas_matriz(),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Codificación matricial interactiva")

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

            resultado_posiciones = calcular_paridades_hamming_7_4(bits_matriz)["codigo_str"]
            resultado_matriz = "".join(str(bit) for bit in c)

            st.subheader("1. Vector de datos")

            st.markdown(
                f"""
El mensaje ingresado es:

**{bits_matriz}**

Por tanto:

$$
m = [D1 \\quad D2 \\quad D3 \\quad D4]
$$

$$
m = {m.tolist()}
$$
"""
            )

            st.subheader("2. Aplicación de la matriz generadora")

            st.markdown(
                """
Se multiplica el vector de datos por la matriz generadora fija:

$$
c = mG \\mod 2
$$

El módulo 2 hace que todas las sumas se interpreten como operaciones XOR.
"""
            )

            st.code(
                f"m = {m.tolist()}\n"
                f"c = m · G mod 2 = {c.tolist()}\n"
                f"Palabra código = {resultado_matriz}",
                language="text",
            )

            st.subheader("3. Comparación con el método por posiciones")

            comparacion = pd.DataFrame(
                [
                    {
                        "Método": "Por posiciones y paridad",
                        "Resultado": resultado_posiciones,
                    },
                    {
                        "Método": "Matricial",
                        "Resultado": resultado_matriz,
                    },
                ]
            )

            st.dataframe(comparacion, use_container_width=True, hide_index=True)

            if resultado_posiciones == resultado_matriz:
                st.success(
                    "El resultado por matriz coincide con el resultado por posiciones. "
                    "Esto confirma que ambos métodos representan la misma codificación."
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
            st.info("Codifique primero un bloque en la pestaña Codificación de un bloque.")

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
                            "Bits de paridad generados": resultado["bits_paridad"],
                            "Bits codificados": len(resultado["codigo_total"]),
                            "Eficiencia efectiva": resultado["eficiencia_efectiva"],
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
6. Ingrese un mensaje de 9 bits y observe que se agregan bits de relleno.
7. Ingrese un mensaje de 16 bits y observe que se divide en 4 bloques.
8. Compare la tasa por bloque $4/7$ con la eficiencia efectiva cuando hay relleno.
9. Verifique que el método matricial da el mismo resultado que el método por posiciones.
10. Explique por qué la matriz generadora no cambia cuando cambia el mensaje.
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
                st.success("Correcto. La eficiencia por bloque es η = k/n = 4/7.")
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

        pregunta_6 = st.radio(
            "Pregunta 6: ¿La matriz generadora G cambia cuando cambia el mensaje?",
            [
                "Sí, cambia con cada mensaje.",
                "No, G permanece fija para el código Hamming (7,4) definido.",
                "Solo cambia cuando el mensaje tiene muchos unos.",
                "Cambia cuando cambia la semilla.",
            ],
            index=None,
            key="g3_pregunta_6",
        )

        if pregunta_6:
            if pregunta_6 == "No, G permanece fija para el código Hamming (7,4) definido.":
                st.success("Correcto. Lo que cambia es el vector m; G representa la regla fija de codificación.")
            else:
                st.error("Revise la diferencia entre el mensaje m y la matriz generadora G.")

        pregunta_7 = st.radio(
            "Pregunta 7: ¿Cuál es la diferencia entre bits de paridad y bits de relleno?",
            [
                "Son exactamente lo mismo.",
                "Los bits de paridad se calculan con XOR; los de relleno solo completan bloques.",
                "Los bits de relleno corrigen errores y los de paridad no.",
                "Los bits de paridad solo aparecen en CRC.",
            ],
            index=None,
            key="g3_pregunta_7",
        )

        if pregunta_7:
            if pregunta_7 == "Los bits de paridad se calculan con XOR; los de relleno solo completan bloques.":
                st.success("Correcto. La paridad es redundancia de Hamming; el relleno solo completa el último bloque.")
            else:
                st.error("Revise la diferencia entre paridad y relleno.")

    # ========================================================
    # Conclusiones
    # ========================================================

    with tabs[6]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- Hamming (7,4) transforma 4 bits de datos en una palabra código de 7 bits.
- Los bits adicionales son bits de paridad calculados mediante XOR.
- Las posiciones de paridad son 1, 2 y 4.
- La estructura utilizada es [P1 P2 D1 P4 D2 D3 D4].
- La fórmula $2^p \\geq k + p + 1$ permite determinar cuántos bits de paridad se requieren.
- Para $k=4$, se necesitan $p=3$ bits de paridad.
- Los mensajes largos se dividen en bloques de 4 bits.
- Si el último bloque no está completo, se agregan bits de relleno.
- Los bits de relleno no son bits de paridad.
- La eficiencia por bloque de Hamming (7,4) es $\\eta = 4/7$.
- La matriz generadora $G$ permanece fija para el código definido.
- Lo que cambia en el método matricial es el vector de datos $m$ y, por tanto, la palabra código $c$.
- La teoría aplicada en esta guía se fundamenta en el control de errores, los códigos de bloque
  y la representación matricial de códigos lineales (Hamming, 1950; Lin & Costello, 2004).
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