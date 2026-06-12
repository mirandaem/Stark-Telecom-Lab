import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Validaciones
# ============================================================

def validar_bits(bits: str) -> bool:
    """
    Valida que una cadena contenga únicamente bits 0 y 1.
    """
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_generador(generador: str) -> bool:
    """
    Valida un polinomio generador binario para CRC.

    Un generador válido debe:
    - tener al menos 2 bits;
    - contener solo 0 y 1;
    - iniciar en 1;
    - terminar en 1.

    Ejemplos válidos:
    1011
    10011
    1101

    Ejemplos no válidos:
    0011  -> no inicia en 1
    1010  -> no termina en 1
    10A1  -> contiene caracteres no binarios
    """
    return (
        len(generador) >= 2
        and all(bit in "01" for bit in generador)
        and generador[0] == "1"
        and generador[-1] == "1"
    )


# ============================================================
# Utilidades teóricas para CRC
# ============================================================

def binario_a_polinomio(bits: str, variable: str = "x") -> str:
    """
    Convierte una cadena binaria en una representación polinómica.

    Ejemplo:
    1011 -> x^3 + x + 1
    """
    if not validar_bits(bits):
        return "Entrada no binaria"

    grado_maximo = len(bits) - 1
    terminos = []

    for i, bit in enumerate(bits):
        if bit == "1":
            grado = grado_maximo - i

            if grado == 0:
                terminos.append("1")
            elif grado == 1:
                terminos.append(variable)
            else:
                terminos.append(f"{variable}^{grado}")

    if not terminos:
        return "0"

    return " + ".join(terminos)


def construir_tabla_elementos_crc(
    datos: str,
    generador: str,
    residuo: str,
    trama: str,
) -> pd.DataFrame:
    """
    Construye una tabla para explicar los elementos de la ecuación CRC:

    T(x) = M(x)x^r + R(x)

    y su forma binaria:

    t = m || R
    """
    grado_r = len(generador) - 1
    datos_desplazados = datos + ("0" * grado_r)

    return pd.DataFrame(
        {
            "Elemento": [
                "m",
                "M(x)",
                "G(x)",
                "r",
                "M(x)x^r",
                "R(x)",
                "t = m || R",
                "T(x)",
            ],
            "Significado": [
                "Mensaje binario original",
                "Polinomio asociado al mensaje",
                "Polinomio generador CRC",
                "Grado del generador y cantidad de bits CRC",
                "Mensaje desplazado r posiciones",
                "Residuo obtenido por división módulo 2",
                "Trama binaria transmitida",
                "Polinomio de la trama transmitida",
            ],
            "Cómo se obtiene": [
                "Lo ingresa el usuario",
                "Se interpreta cada bit 1 como una potencia de x",
                "Lo ingresa el usuario como cadena binaria",
                "r = longitud(G) - 1",
                "Se agregan r ceros al final del mensaje",
                "R(x) = M(x)x^r mod G(x)",
                "Se concatena el mensaje con el residuo CRC",
                "T(x) = M(x)x^r + R(x), con suma módulo 2",
            ],
            "Valor en este ejemplo": [
                datos,
                binario_a_polinomio(datos),
                f"{generador} = {binario_a_polinomio(generador)}",
                grado_r,
                datos_desplazados,
                f"{residuo} = {binario_a_polinomio(residuo)}",
                trama,
                binario_a_polinomio(trama),
            ],
        }
    )


def construir_tabla_grado_generador() -> pd.DataFrame:
    """
    Tabla didáctica para explicar cómo cambia r según el generador.
    """
    ejemplos = ["1011", "10011", "11001", "100001"]

    filas = []

    for generador in ejemplos:
        grado_r = len(generador) - 1

        filas.append(
            {
                "Generador binario": generador,
                "G(x)": binario_a_polinomio(generador),
                "Longitud del generador": len(generador),
                "r = longitud - 1": grado_r,
                "Ceros agregados al mensaje": grado_r,
                "Bits CRC por trama": grado_r,
            }
        )

    return pd.DataFrame(filas)


def construir_tabla_total_bits_crc(grado_r: int) -> pd.DataFrame:
    """
    Explica que r no cambia por la cantidad de tramas.
    Lo que cambia es la cantidad total de bits CRC agregados.
    """
    cantidades_tramas = [1, 10, 100, 1000]

    filas = []

    for tramas in cantidades_tramas:
        filas.append(
            {
                "Cantidad de tramas": tramas,
                "Bits CRC por trama": grado_r,
                "Bits CRC totales agregados": tramas * grado_r,
                "Interpretación": (
                    "r no cambia; aumenta el total de bits CRC porque hay más tramas"
                ),
            }
        )

    return pd.DataFrame(filas)


# ============================================================
# CRC
# ============================================================

def division_modulo_2(dividendo: str, divisor: str) -> Tuple[str, pd.DataFrame]:
    """
    Realiza división módulo 2 para CRC.

    En aritmética módulo 2, la resta se implementa mediante XOR.
    Por eso, cuando el bit líder del segmento es 1, se aplica XOR con el divisor.

    Devuelve:
    - residuo;
    - tabla de pasos de la división.
    """
    trabajo = list(dividendo)
    divisor_bits = list(divisor)
    n = len(divisor_bits)

    pasos: List[Dict[str, object]] = []

    for i in range(len(dividendo) - n + 1):
        segmento_antes = "".join(trabajo[i:i + n])

        if trabajo[i] == "1":
            for j in range(n):
                trabajo[i + j] = str(int(trabajo[i + j]) ^ int(divisor_bits[j]))

            segmento_despues = "".join(trabajo[i:i + n])

            pasos.append(
                {
                    "Paso": len(pasos) + 1,
                    "Posición": i + 1,
                    "Segmento antes": segmento_antes,
                    "Operación": f"{segmento_antes} XOR {divisor}",
                    "Segmento después": segmento_despues,
                    "Trama temporal": "".join(trabajo),
                }
            )
        else:
            pasos.append(
                {
                    "Paso": len(pasos) + 1,
                    "Posición": i + 1,
                    "Segmento antes": segmento_antes,
                    "Operación": "No se aplica XOR porque el bit líder es 0",
                    "Segmento después": segmento_antes,
                    "Trama temporal": "".join(trabajo),
                }
            )

    residuo = "".join(trabajo[-(n - 1):])
    return residuo, pd.DataFrame(pasos)


def generar_crc(datos: str, generador: str) -> Tuple[str, str, pd.DataFrame]:
    """
    Genera el residuo CRC y la trama final.

    Procedimiento:
    1. Se determina r = longitud(generador) - 1.
    2. Se agregan r ceros al mensaje.
    3. Se divide módulo 2 entre el generador.
    4. El residuo se concatena al mensaje original.

    En forma polinómica:

    R(x) = M(x)x^r mod G(x)

    T(x) = M(x)x^r + R(x)

    En forma binaria:

    t = m || R

    donde || significa concatenación, no suma aritmética común.
    """
    ceros = "0" * (len(generador) - 1)
    dividendo = datos + ceros

    residuo, pasos = division_modulo_2(dividendo, generador)
    trama = datos + residuo

    return residuo, trama, pasos


def verificar_crc(trama: str, generador: str) -> Tuple[bool, str, pd.DataFrame]:
    """
    Verifica una trama con CRC.

    El receptor divide la trama recibida entre el mismo generador.

    Si el residuo es cero:
    - no se detecta error.

    Si el residuo es distinto de cero:
    - se detecta una inconsistencia.

    Importante:
    Residuo cero no significa garantía absoluta de que no hubo error.
    Significa que el CRC no detectó error.
    """
    residuo, pasos = division_modulo_2(trama, generador)
    valido = all(bit == "0" for bit in residuo)

    return valido, residuo, pasos


# ============================================================
# Utilidades de simulación
# ============================================================

def generar_bits_aleatorios(cantidad: int, semilla: int | None = None) -> str:
    rng = np.random.default_rng(semilla)
    bits = rng.integers(0, 2, size=cantidad)
    return "".join(str(bit) for bit in bits)


def dividir_en_tramas(bits: str, tamano_payload: int) -> List[str]:
    """
    Divide una secuencia de datos en tramas de tamaño fijo.

    Si la última trama queda incompleta, se rellena con ceros.
    Este relleno permite completar el tamaño del payload para la simulación.
    """
    tramas = []

    for i in range(0, len(bits), tamano_payload):
        bloque = bits[i:i + tamano_payload]

        if len(bloque) < tamano_payload:
            bloque = bloque + "0" * (tamano_payload - len(bloque))

        tramas.append(bloque)

    return tramas


def bits_a_simbolos_bpsk(bits: str) -> np.ndarray:
    """
    Convierte bits a símbolos BPSK normalizados.

    0 -> -1
    1 -> +1
    """
    bits_array = np.array([int(bit) for bit in bits])
    return np.where(bits_array == 1, 1.0, -1.0)


def transmitir_awgn(
    bits: str,
    sigma: float,
    semilla: int | None = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transmite bits usando:
    - BPSK normalizado: 0 -> -1, 1 -> +1;
    - ruido gaussiano AWGN;
    - decisión por umbral en cero.

    El modelo usado es:

    r = s + n

    donde:
    s = símbolo transmitido;
    n = ruido gaussiano;
    r = valor recibido.
    """
    rng = np.random.default_rng(semilla)

    simbolos = bits_a_simbolos_bpsk(bits)
    ruido = rng.normal(loc=0.0, scale=sigma, size=len(simbolos))
    recibido_analogico = simbolos + ruido
    bits_decididos = np.where(recibido_analogico >= 0, 1, 0)

    bits_rx = "".join(str(bit) for bit in bits_decididos)

    return bits_rx, simbolos, ruido, recibido_analogico


def contar_errores_bits(tx: str, rx: str) -> int:
    return sum(1 for a, b in zip(tx, rx) if a != b)


def calcular_potencias(
    simbolos: np.ndarray,
    ruido: np.ndarray,
) -> Tuple[float, float, float, float]:
    potencia_senal = float(np.mean(simbolos**2))
    potencia_ruido = float(np.mean(ruido**2))

    if potencia_ruido == 0:
        snr = math.inf
        snr_db = math.inf
    else:
        snr = potencia_senal / potencia_ruido
        snr_db = 10 * math.log10(snr)

    return potencia_senal, potencia_ruido, snr, snr_db


# ============================================================
# Simulación estadística CRC
# ============================================================

def simular_crc_estadistico(
    cantidad_bits_datos: int,
    tamano_payload: int,
    generador: str,
    sigma: float,
    semilla: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simula muchas tramas con CRC a través de un canal AWGN.

    Métricas calculadas:
    - BER del canal;
    - FER;
    - tasa de detección CRC;
    - tasa de errores no detectados;
    - potencia promedio de ruido;
    - SNR promedio.
    """
    datos = generar_bits_aleatorios(cantidad_bits_datos, semilla)
    tramas_datos = dividir_en_tramas(datos, tamano_payload)

    resultados = []

    total_bits_codificados = 0
    total_errores_bit = 0
    tramas_con_error = 0
    tramas_detectadas = 0
    tramas_no_detectadas = 0
    tramas_sin_error = 0

    potencias_senal = []
    potencias_ruido = []
    snrs = []
    snrs_db = []

    for i, payload in enumerate(tramas_datos, start=1):
        semilla_trama = None if semilla is None else semilla + i

        residuo_tx, trama_tx, _ = generar_crc(payload, generador)

        trama_rx, simbolos, ruido, _ = transmitir_awgn(
            trama_tx,
            sigma=sigma,
            semilla=semilla_trama,
        )

        valido_crc, residuo_rx, _ = verificar_crc(trama_rx, generador)

        errores_bit = contar_errores_bits(trama_tx, trama_rx)
        tiene_error = errores_bit > 0

        ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

        total_bits_codificados += len(trama_tx)
        total_errores_bit += errores_bit

        potencias_senal.append(ps)
        potencias_ruido.append(pn)
        snrs.append(snr)
        snrs_db.append(snr_db)

        if tiene_error:
            tramas_con_error += 1

            if valido_crc:
                tramas_no_detectadas += 1
                estado = "Error no detectado"
            else:
                tramas_detectadas += 1
                estado = "Error detectado"
        else:
            tramas_sin_error += 1
            estado = "Sin error"

        resultados.append(
            {
                "Trama": i,
                "Datos": payload,
                "CRC Tx": residuo_tx,
                "Trama Tx": trama_tx,
                "Trama Rx": trama_rx,
                "Errores de bit": errores_bit,
                "Residuo Rx": residuo_rx,
                "CRC válido": valido_crc,
                "Estado": estado,
            }
        )

    total_tramas = len(tramas_datos)

    ber = total_errores_bit / total_bits_codificados if total_bits_codificados else 0
    fer = tramas_con_error / total_tramas if total_tramas else 0
    tasa_deteccion = tramas_detectadas / tramas_con_error if tramas_con_error else 0
    tasa_no_detectada = tramas_no_detectadas / tramas_con_error if tramas_con_error else 0

    resumen = {
        "Tramas evaluadas": total_tramas,
        "Bits codificados evaluados": total_bits_codificados,
        "Errores de bit": total_errores_bit,
        "BER del canal": ber,
        "Tramas sin error": tramas_sin_error,
        "Tramas con error": tramas_con_error,
        "FER": fer,
        "Tramas detectadas por CRC": tramas_detectadas,
        "Errores no detectados por CRC": tramas_no_detectadas,
        "Tasa de detección CRC": tasa_deteccion,
        "Tasa de error no detectado": tasa_no_detectada,
        "Potencia señal promedio": float(np.mean(potencias_senal)),
        "Potencia ruido promedio": float(np.mean(potencias_ruido)),
        "SNR promedio": float(np.mean(snrs)) if all(np.isfinite(snrs)) else math.inf,
        "SNR dB promedio": float(np.mean(snrs_db)) if all(np.isfinite(snrs_db)) else math.inf,
        "σ": sigma,
        "σ²": sigma**2,
    }

    return pd.DataFrame(resultados), resumen


def comparar_sigmas_crc(
    cantidad_bits_datos: int,
    tamano_payload: int,
    generador: str,
    valores_sigma: List[float],
    semilla: int | None = None,
) -> pd.DataFrame:
    filas = []

    for i, sigma in enumerate(valores_sigma):
        semilla_escenario = None if semilla is None else semilla + (1000 * i)

        _, resumen = simular_crc_estadistico(
            cantidad_bits_datos=cantidad_bits_datos,
            tamano_payload=tamano_payload,
            generador=generador,
            sigma=sigma,
            semilla=semilla_escenario,
        )

        filas.append(resumen)

    return pd.DataFrame(filas)


# ============================================================
# Gráficas discretas y semilogarítmicas
# ============================================================

def graficar_trama_discreta(trama_tx: str, trama_rx: str) -> None:
    posiciones = np.arange(1, len(trama_tx) + 1)
    tx = np.array([int(bit) for bit in trama_tx])
    rx = np.array([int(bit) for bit in trama_rx])

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.stem(
        posiciones,
        tx,
        basefmt=" ",
        label="Bit transmitido",
    )

    ax.scatter(
        posiciones,
        rx,
        marker="x",
        label="Bit recibido",
    )

    ax.set_title("Trama transmitida y trama recibida como muestras discretas")
    ax.set_xlabel("Índice de bit")
    ax.set_ylabel("Valor del bit")
    ax.set_yticks([0, 1])
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


def graficar_senal_crc_discreta(
    simbolos: np.ndarray,
    ruido: np.ndarray,
    recibido_analogico: np.ndarray,
) -> None:
    posiciones = np.arange(1, len(simbolos) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.stem(
        posiciones,
        simbolos,
        basefmt=" ",
        label="Símbolo transmitido",
    )

    ax.scatter(
        posiciones,
        recibido_analogico,
        marker="x",
        label="Valor recibido",
    )

    ax.axhline(0, linestyle="--", linewidth=1, label="Umbral de decisión")

    ax.set_title("Símbolos y valores recibidos por muestra")
    ax.set_xlabel("Índice de bit")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    fig_ruido, ax_ruido = plt.subplots(figsize=(10, 3))

    ax_ruido.stem(
        posiciones,
        ruido,
        basefmt=" ",
    )

    ax_ruido.axhline(0, linestyle="--", linewidth=1)
    ax_ruido.set_title("Ruido gaussiano por muestra")
    ax_ruido.set_xlabel("Índice de bit")
    ax_ruido.set_ylabel("Ruido")
    ax_ruido.grid(True)

    st.pyplot(fig_ruido)
    plt.close(fig_ruido)


def graficar_ber_vs_sigma(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["BER ajustado"] = df_plot["BER del canal"].replace(0, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["σ"], df_plot["BER ajustado"], marker="o")

    ax.set_title("BER del canal vs σ")
    ax.set_xlabel("Desviación estándar σ")
    ax.set_ylabel("BER en escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)
    plt.close(fig)


def graficar_fer_vs_sigma(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["FER ajustado"] = df_plot["FER"].replace(0, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["σ"], df_plot["FER ajustado"], marker="o")

    ax.set_title("FER vs σ")
    ax.set_xlabel("Desviación estándar σ")
    ax.set_ylabel("FER en escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)
    plt.close(fig)


def graficar_no_detectados_vs_sigma(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot["No detectado ajustado"] = df_plot["Tasa de error no detectado"].replace(0, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["σ"], df_plot["No detectado ajustado"], marker="o")

    ax.set_title("Tasa de error no detectado vs σ")
    ax.set_xlabel("Desviación estándar σ")
    ax.set_ylabel("Tasa en escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)
    plt.close(fig)


def graficar_ber_vs_snr(df: pd.DataFrame) -> None:
    df_plot = df.copy()
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["SNR dB promedio"])
    df_plot["BER ajustado"] = df_plot["BER del canal"].replace(0, 1e-6)

    if df_plot.empty:
        st.info("No se puede graficar BER vs SNR cuando la SNR es infinita.")
        return

    df_plot = df_plot.sort_values("SNR dB promedio")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["SNR dB promedio"], df_plot["BER ajustado"], marker="o")

    ax.set_title("BER del canal vs SNR dB")
    ax.set_xlabel("SNR dB")
    ax.set_ylabel("BER en escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Interfaz Streamlit
# ============================================================

def render_guia_05() -> None:
    st.title("Guía 5: CRC y detección de errores remanentes")

    st.markdown(
        """
Esta guía estudia el Código de Redundancia Cíclica, conocido como CRC, como mecanismo
de detección de errores. A diferencia de Hamming, CRC no corrige bits alterados. Su
función es verificar si una trama recibida conserva una relación matemática esperada
con un polinomio generador.

La guía combina el procedimiento algebraico del CRC con simulaciones estadísticas sobre
muchas tramas transmitidas por un canal con ruido gaussiano. Además, mantiene una
representación discreta de bits y muestras para evitar interpretar las tramas digitales
como señales continuas.

CRC complementa a Hamming porque permite detectar errores remanentes que pueden quedar
después de una corrección incorrecta o incompleta (Forouzan, 2013; Stallings, 2015;
Lin & Costello, 2004).
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "CRC paso a paso",
            "Canal discreto",
            "Estadística con tramas",
            "Comparación",
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

Comprender el funcionamiento del CRC como técnica de detección de errores y analizar
su desempeño estadístico bajo condiciones de ruido.

**Objetivos específicos**

1. Representar una secuencia binaria como un polinomio sobre aritmética módulo 2.
2. Comprender la ecuación CRC $T(x) = M(x)x^r + R(x)$.
3. Identificar cómo se obtiene cada elemento de la ecuación CRC.
4. Calcular el residuo CRC a partir de un polinomio generador.
5. Explicar cómo se determina el grado $r$ del generador.
6. Construir una trama transmitida formada por datos y residuo CRC.
7. Verificar una trama recibida usando el mismo generador.
8. Observar el efecto del ruido sobre una trama con CRC mediante muestras discretas.
9. Evaluar muchas tramas para estimar BER, FER y tasa de detección.
10. Relacionar CRC con los límites de Hamming frente a errores múltiples.
"""
        )

    # ========================================================
    # Teoría
    # ========================================================

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
El Código de Redundancia Cíclica, conocido como CRC, es una técnica de detección de
errores basada en aritmética binaria módulo 2. En esta aritmética, la suma y la resta
se implementan mediante XOR. CRC es ampliamente usado en comunicaciones digitales y
redes porque permite verificar la integridad de una trama recibida (Forouzan, 2013;
Stallings, 2015).

La idea central es que el transmisor no envía únicamente los datos originales. Antes de
transmitir, calcula un residuo a partir de un polinomio generador. Ese residuo se agrega
al final del mensaje para formar una trama. El receptor divide la trama recibida entre
el mismo generador. Si el residuo de recepción es distinto de cero, se detecta una
inconsistencia.

CRC **detecta errores**, pero **no corrige errores**. Esto significa que puede indicar
que una trama está alterada, pero no dice qué bit está mal ni cómo repararlo.
"""
        )

        st.subheader("Representación polinómica")

        st.markdown(
            """
En CRC, una cadena binaria puede interpretarse como un polinomio. Por ejemplo:

```text
1101
```

se interpreta como:

$$
M(x) = x^3 + x^2 + 1
$$

porque los bits en 1 indican qué potencias de $x$ aparecen en el polinomio.

De la misma forma, un generador como:

```text
1011
```

representa:

$$
G(x) = x^3 + x + 1
$$

Esta representación permite realizar la división módulo 2 entre polinomios binarios
(Forouzan, 2013; Lin & Costello, 2004).
"""
        )

        st.subheader("Ecuación principal del CRC")

        st.markdown(
            """
La ecuación formal usada para construir la trama CRC es:

$$
T(x) = M(x)x^r + R(x)
$$

donde:

- $M(x)$ es el polinomio del mensaje original;
- $G(x)$ es el polinomio generador;
- $r$ es el grado del generador;
- $M(x)x^r$ representa el mensaje desplazado $r$ posiciones;
- $R(x)$ es el residuo CRC;
- $T(x)$ es la trama transmitida.

En binario, esto puede explicarse como:

$$
t = m || R
$$

donde $||$ significa **concatenación**. No significa suma aritmética común.

Por eso, cuando se escribe de forma simple “trama = mensaje + residuo”, debe entenderse
como:

```text
trama = mensaje concatenado con residuo CRC
```

No como una suma decimal o una suma binaria convencional.
"""
        )

        st.subheader("¿Cómo se encuentra R(x)?")

        st.markdown(
            """
El residuo CRC se obtiene mediante:

$$
R(x) = M(x)x^r \\bmod G(x)
$$

El procedimiento es:

1. Se toma el mensaje original $M(x)$.
2. Se determina $r$, que es el grado del generador $G(x)$.
3. Se multiplica $M(x)$ por $x^r$.
4. En binario, multiplicar por $x^r$ equivale a agregar $r$ ceros al final del mensaje.
5. Se divide $M(x)x^r$ entre $G(x)$ usando división módulo 2.
6. El residuo de esa división es $R(x)$.

Ejemplo conceptual:

```text
Datos:       1101
Generador:   1011
r:           3
Datos + r ceros: 1101000
```

Luego se divide:

```text
1101000 ÷ 1011
```

usando XOR. El residuo final de esa división es el CRC que se agrega al mensaje.
"""
        )

        st.subheader("¿Cómo cambia r?")

        st.markdown(
            """
El valor $r$ depende del polinomio generador, no del número de tramas.

En binario:

$$
r = \\text{longitud del generador} - 1
$$

Por ejemplo:

- Si el generador es `1011`, su longitud es 4, por tanto $r = 3$.
- Si el generador es `10011`, su longitud es 5, por tanto $r = 4$.
- Si el generador es `100001`, su longitud es 6, por tanto $r = 5$.

Lo que sí cambia con la cantidad de tramas es la cantidad total de bits CRC agregados.
Si $r = 3$ y se transmiten 100 tramas, entonces se agregan:

$$
100 \\times 3 = 300
$$

bits CRC en total.

Por tanto:

- $r$ depende del generador;
- el residuo por trama tiene $r$ bits;
- la cantidad total de bits CRC aumenta cuando aumenta el número de tramas.
"""
        )

        st.dataframe(
            construir_tabla_grado_generador(),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Verificación en el receptor")

        st.markdown(
            """
En el receptor, se recibe una trama $T_{rx}(x)$ y se divide entre el mismo generador:

$$
R_{rx}(x) = T_{rx}(x) \\bmod G(x)
$$

Si:

$$
R_{rx}(x) = 0
$$

entonces **no se detecta error**.

Si:

$$
R_{rx}(x) \\neq 0
$$

entonces **se detecta error**.

Es importante usar el lenguaje correcto: un residuo cero no garantiza matemáticamente
que no haya ocurrido ningún error. Significa que el CRC no detectó error con ese
generador. Algunos patrones de error pueden no ser detectados, dependiendo del generador
y del patrón de alteración (Forouzan, 2013; Stallings, 2015).
"""
        )

    # ========================================================
    # CRC paso a paso
    # ========================================================

    with tabs[2]:
        st.header("CRC paso a paso")

        st.markdown(
            """
En esta sección se calcula el CRC de forma manual asistida. El estudiante puede ingresar
los datos y el polinomio generador para observar cada paso de la división módulo 2.

Esta sección responde a las preguntas:

- ¿Cómo se encuentra $R(x)$?
- ¿Cómo se obtiene cada elemento de la ecuación?
- ¿Cómo se interpreta $t = m || R$?
- ¿Cómo cambia $r$ según el generador?
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            datos = st.text_input(
                "Datos binarios",
                value="1101",
                key="g5_datos_manual",
            ).strip()

            generador = st.text_input(
                "Polinomio generador",
                value="1011",
                key="g5_generador_manual",
            ).strip()

            ejecutar = st.button("Calcular CRC")

        with col_info:
            st.info(
                """
Ejemplo típico:

Datos: 1101  
Generador: 1011  

Si el generador tiene longitud 4, entonces:

r = 4 - 1 = 3

Por tanto, se agregan 3 ceros al mensaje antes de dividir.
"""
            )

        if not validar_bits(datos):
            st.error("Los datos deben contener únicamente 0 y 1.")
        elif not validar_generador(generador):
            st.error("El generador debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar:
            residuo, trama, pasos = generar_crc(datos, generador)
            grado_r = len(generador) - 1
            datos_desplazados = datos + ("0" * grado_r)
            tabla_elementos = construir_tabla_elementos_crc(datos, generador, residuo, trama)

            st.subheader("1. Elementos de la ecuación CRC")

            st.markdown(
                """
La ecuación formal es:

$$
T(x) = M(x)x^r + R(x)
$$

La siguiente tabla explica qué significa cada elemento y cómo se obtiene en este caso.
"""
            )

            st.dataframe(tabla_elementos, use_container_width=True, hide_index=True)

            st.subheader("2. Resultado del cálculo")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Longitud del generador", len(generador))
            c2.metric("Grado r", grado_r)
            c3.metric("Bits CRC", grado_r)
            c4.metric("Residuo", residuo)

            st.code(
                f"Datos m:             {datos}\n"
                f"M(x):                {binario_a_polinomio(datos)}\n"
                f"Generador G:         {generador}\n"
                f"G(x):                {binario_a_polinomio(generador)}\n"
                f"r:                   {grado_r}\n"
                f"Datos con r ceros:   {datos_desplazados}\n"
                f"Residuo R:           {residuo}\n"
                f"Trama t = m || R:    {trama}",
                language="text",
            )

            st.subheader("3. Pasos de la división módulo 2")

            st.markdown(
                """
En la tabla siguiente, cada paso muestra si se aplicó XOR con el generador. La operación
solo se realiza cuando el bit líder del segmento actual es 1. Al final, los últimos
$r$ bits resultantes forman el residuo CRC.
"""
            )

            st.dataframe(pasos, use_container_width=True, hide_index=True)

            st.subheader("4. Efecto del número de tramas sobre los bits CRC")

            st.markdown(
                """
El valor $r$ no cambia por transmitir más tramas. $r$ depende únicamente del generador.
Sin embargo, si se transmiten más tramas, se agregan más bits CRC en total.
"""
            )

            st.dataframe(
                construir_tabla_total_bits_crc(grado_r),
                use_container_width=True,
                hide_index=True,
            )

            st.session_state["g5_trama_manual_resultado"] = trama
            st.session_state["g5_generador_manual_resultado"] = generador
            st.session_state["g5_residuo_manual_resultado"] = residuo

    # ========================================================
    # Canal discreto
    # ========================================================

    with tabs[3]:
        st.header("Canal discreto y verificación CRC")

        st.markdown(
            """
En esta sección se transmite una trama con CRC a través de un canal con ruido gaussiano.
La trama se representa mediante símbolos BPSK, se suma ruido y el receptor decide cada
bit mediante un umbral.

Las gráficas se muestran como muestras discretas, no como señales continuas. Esto es
importante porque el receptor toma decisiones bit a bit.

El modelo usado es:

$$
r = s + n
$$

donde:

- $s$ es el símbolo transmitido;
- $n$ es el ruido;
- $r$ es el valor recibido.

Luego se aplica una decisión por umbral:

- si $r \\geq 0$, se decide bit 1;
- si $r < 0$, se decide bit 0.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            datos_senal = st.text_input(
                "Datos binarios",
                value="11010101",
                key="g5_datos_senal",
            ).strip()

            generador_senal = st.text_input(
                "Generador CRC",
                value="1011",
                key="g5_generador_senal",
            ).strip()

            sigma = st.slider(
                "Desviación estándar del ruido σ",
                min_value=0.0,
                max_value=2.0,
                value=0.50,
                step=0.05,
                key="g5_sigma_senal",
            )

            semilla = st.number_input(
                "Semilla",
                min_value=0,
                max_value=999999,
                value=25,
                step=1,
                key="g5_semilla_senal",
            )

            ejecutar_senal = st.button("Transmitir trama con CRC")

        with col_info:
            st.info(
                """
El receptor no conoce la trama original. Solo observa los bits decididos después del
ruido y aplica el CRC.

Si el residuo es distinto de cero, detecta error.
Si el residuo es cero, no detecta error.
"""
            )

            st.metric("Varianza del ruido σ²", f"{sigma**2:.4f}")

        if not validar_bits(datos_senal):
            st.error("Los datos deben ser binarios.")
        elif not validar_generador(generador_senal):
            st.error("El generador debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar_senal:
            residuo, trama_tx, _ = generar_crc(datos_senal, generador_senal)

            trama_rx, simbolos, ruido, recibido_analogico = transmitir_awgn(
                trama_tx,
                sigma=sigma,
                semilla=int(semilla),
            )

            valido, residuo_rx, pasos_rx = verificar_crc(trama_rx, generador_senal)
            errores_bit = contar_errores_bits(trama_tx, trama_rx)
            ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

            st.subheader("Resultados de transmisión")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Errores de bit", errores_bit)
            c2.metric("CRC válido", "Sí" if valido else "No")
            c3.metric("Residuo Rx", residuo_rx)
            c4.metric("SNR dB", "∞" if math.isinf(snr_db) else f"{snr_db:.2f}")

            st.code(
                f"Datos:              {datos_senal}\n"
                f"CRC generado:       {residuo}\n"
                f"Trama Tx:           {trama_tx}\n"
                f"Trama Rx:           {trama_rx}\n"
                f"Residuo receptor:   {residuo_rx}",
                language="text",
            )

            st.subheader("Trama discreta transmitida y recibida")
            graficar_trama_discreta(trama_tx, trama_rx)

            st.subheader("Símbolos, ruido y valores recibidos")
            graficar_senal_crc_discreta(simbolos, ruido, recibido_analogico)

            tabla_bits = pd.DataFrame(
                {
                    "Posición": np.arange(1, len(trama_tx) + 1),
                    "Bit Tx": list(trama_tx),
                    "Bit Rx": list(trama_rx),
                    "Estado": [
                        "Correcto" if a == b else "Error"
                        for a, b in zip(trama_tx, trama_rx)
                    ],
                }
            )

            st.subheader("Comparación bit a bit")
            st.dataframe(tabla_bits, use_container_width=True, hide_index=True)

            st.subheader("Verificación CRC en receptor")
            st.dataframe(pasos_rx, use_container_width=True, hide_index=True)

            if errores_bit == 0 and valido:
                st.success("La trama llegó sin errores y el CRC no detectó inconsistencia.")
            elif errores_bit > 0 and not valido:
                st.warning("La trama fue alterada y el CRC detectó el error.")
            elif errores_bit > 0 and valido:
                st.error(
                    "Ocurrió un error no detectado. La trama cambió, pero el residuo CRC fue cero."
                )

            st.session_state["g5_ultimo_canal"] = {
                "datos": datos_senal,
                "trama_tx": trama_tx,
                "trama_rx": trama_rx,
                "errores": errores_bit,
                "residuo_rx": residuo_rx,
                "crc_valido": valido,
                "sigma": sigma,
                "snr_db": snr_db,
            }

    # ========================================================
    # Estadística con tramas
    # ========================================================

    with tabs[4]:
        st.header("Estadística con muchas tramas")

        st.markdown(
            """
En esta sección se transmiten muchas tramas con CRC. El objetivo es estimar
estadísticamente cuántas tramas son alteradas por el canal y cuántas son detectadas
por el CRC.

Las métricas principales son:

$$
BER = \\frac{\\text{bits erróneos}}{\\text{bits transmitidos}}
$$

$$
FER = \\frac{\\text{tramas con error}}{\\text{tramas transmitidas}}
$$

La tasa de detección CRC se calcula como:

$$
\\text{Tasa de detección} =
\\frac{\\text{tramas con error detectadas por CRC}}{\\text{tramas con error}}
$$

Estas métricas permiten evaluar el desempeño del detector CRC bajo ruido (Forouzan,
2013; Stallings, 2015).
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits = st.selectbox(
                "Cantidad total de bits de datos",
                [100, 1000, 5000, 10000],
                index=3,
                key="g5_bits_estadistica",
            )

            tamano_payload = st.selectbox(
                "Tamaño de datos por trama",
                [4, 8, 16, 32],
                index=2,
                key="g5_payload_estadistica",
            )

            generador_est = st.text_input(
                "Generador CRC",
                value="1011",
                key="g5_generador_estadistica",
            ).strip()

            sigma_est = st.slider(
                "Desviación estándar del ruido σ",
                min_value=0.0,
                max_value=2.0,
                value=0.50,
                step=0.05,
                key="g5_sigma_estadistica",
            )

            semilla_est = st.number_input(
                "Semilla",
                min_value=0,
                max_value=999999,
                value=100,
                step=1,
                key="g5_semilla_estadistica",
            )

            ejecutar_est = st.button("Ejecutar simulación estadística")

        with col_info:
            st.info(
                """
Esta simulación permite estimar:

- BER del canal;
- FER o tasa de error de trama;
- tasa de detección del CRC;
- errores no detectados;
- SNR promedio.
"""
            )

            st.metric("Varianza σ²", f"{sigma_est**2:.4f}")

        if not validar_generador(generador_est):
            st.error("El generador debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar_est:
            df_resultados, resumen = simular_crc_estadistico(
                cantidad_bits_datos=cantidad_bits,
                tamano_payload=tamano_payload,
                generador=generador_est,
                sigma=sigma_est,
                semilla=int(semilla_est),
            )

            st.subheader("Resumen estadístico")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tramas evaluadas", int(resumen["Tramas evaluadas"]))
            c2.metric("BER canal", f"{resumen['BER del canal']:.6f}")
            c3.metric("FER", f"{resumen['FER']:.6f}")
            c4.metric("Detección CRC", f"{resumen['Tasa de detección CRC']:.4f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Errores de bit", int(resumen["Errores de bit"]))
            c6.metric("Tramas con error", int(resumen["Tramas con error"]))
            c7.metric("Errores no detectados", int(resumen["Errores no detectados por CRC"]))
            c8.metric(
                "SNR dB promedio",
                "∞" if math.isinf(resumen["SNR dB promedio"]) else f"{resumen['SNR dB promedio']:.2f}",
            )

            st.subheader("Tabla resumen")
            st.dataframe(pd.DataFrame([resumen]), use_container_width=True, hide_index=True)

            st.subheader("Primeras tramas evaluadas")
            st.dataframe(df_resultados.head(25), use_container_width=True, hide_index=True)

            conteo_estados = df_resultados["Estado"].value_counts().reset_index()
            conteo_estados.columns = ["Estado", "Cantidad"]

            st.subheader("Distribución de estados")
            st.bar_chart(conteo_estados.set_index("Estado"))

            st.session_state["g5_resumen_estadistico"] = resumen
            st.session_state["g5_df_estadistico"] = df_resultados.head(25)

    # ========================================================
    # Comparación
    # ========================================================

    with tabs[5]:
        st.header("Comparación de escenarios")

        st.markdown(
            """
En esta sección se comparan varios niveles de ruido. Esto permite observar cómo cambian
el BER, el FER, la SNR y la capacidad de detección del CRC.

Las métricas de error se muestran en escala semilogarítmica cuando corresponde. Esta
escala es útil porque el BER puede tomar valores muy pequeños y una escala lineal puede
ocultar diferencias importantes.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            bits_comp = st.selectbox(
                "Bits de datos por comparación",
                [1000, 5000, 10000],
                index=2,
                key="g5_bits_comp",
            )

            payload_comp = st.selectbox(
                "Tamaño de datos por trama",
                [4, 8, 16, 32],
                index=2,
                key="g5_payload_comp",
            )

            gen_comp = st.text_input(
                "Generador CRC",
                value="1011",
                key="g5_gen_comp",
            ).strip()

            sigmas_texto = st.text_input(
                "Valores de σ separados por coma",
                value="0.10, 0.30, 0.50, 0.80, 1.00",
                key="g5_sigmas_comp",
            )

            semilla_comp = st.number_input(
                "Semilla base",
                min_value=0,
                max_value=999999,
                value=700,
                step=1,
                key="g5_semilla_comp",
            )

            ejecutar_comp = st.button("Comparar escenarios")

        with col_info:
            st.info(
                """
Al aumentar σ, aumenta la potencia del ruido.

Al aumentar el ruido, disminuye la SNR y se espera mayor BER y FER.
"""
            )

        if not validar_generador(gen_comp):
            st.error("El generador debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar_comp:
            try:
                valores_sigma = [
                    float(valor.strip())
                    for valor in sigmas_texto.split(",")
                    if valor.strip() != ""
                ]
            except ValueError:
                st.error("Ingrese valores numéricos válidos separados por coma.")
                return

            if len(valores_sigma) == 0:
                st.error("Debe ingresar al menos un valor de σ.")
                return

            if any(valor < 0 for valor in valores_sigma):
                st.error("Los valores de σ no pueden ser negativos.")
                return

            df_comp = comparar_sigmas_crc(
                cantidad_bits_datos=bits_comp,
                tamano_payload=payload_comp,
                generador=gen_comp,
                valores_sigma=valores_sigma,
                semilla=int(semilla_comp),
            )

            st.subheader("Tabla comparativa")
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.subheader("BER del canal vs σ")
            graficar_ber_vs_sigma(df_comp)

            st.subheader("FER vs σ")
            graficar_fer_vs_sigma(df_comp)

            st.subheader("Tasa de errores no detectados vs σ")
            graficar_no_detectados_vs_sigma(df_comp)

            st.subheader("BER del canal vs SNR dB")
            graficar_ber_vs_snr(df_comp)

            st.session_state["g5_comparacion"] = df_comp

            st.markdown(
                """
**Lectura esperada**

- Si aumenta σ, aumenta σ².
- Si aumenta σ², aumenta la potencia promedio del ruido.
- Si aumenta la potencia del ruido, disminuye la SNR.
- Si disminuye la SNR, aumenta el BER.
- Si aumentan los errores de trama, el CRC debe detectar más tramas alteradas.
- Si existen errores no detectados, se evidencia que ningún detector es perfecto para todos los patrones.
"""
            )

    # ========================================================
    # Análisis y dinámica
    # ========================================================

    with tabs[6]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección integra la interpretación de resultados con actividades guiadas.
El objetivo es que el estudiante comprenda la función del CRC como detector, su
relación con Hamming y sus límites.
"""
        )

        if "g5_trama_manual_resultado" in st.session_state:
            st.subheader("Último CRC calculado paso a paso")

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Trama generada": st.session_state["g5_trama_manual_resultado"],
                            "Generador": st.session_state["g5_generador_manual_resultado"],
                            "Residuo CRC": st.session_state["g5_residuo_manual_resultado"],
                        }
                    ]
                )
            )
        else:
            st.info("Calcule primero un CRC en la pestaña CRC paso a paso.")

        if "g5_ultimo_canal" in st.session_state:
            st.subheader("Última transmisión por canal")

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Datos": st.session_state["g5_ultimo_canal"]["datos"],
                            "Trama Tx": st.session_state["g5_ultimo_canal"]["trama_tx"],
                            "Trama Rx": st.session_state["g5_ultimo_canal"]["trama_rx"],
                            "Errores": st.session_state["g5_ultimo_canal"]["errores"],
                            "Residuo Rx": st.session_state["g5_ultimo_canal"]["residuo_rx"],
                            "CRC válido": st.session_state["g5_ultimo_canal"]["crc_valido"],
                            "σ": st.session_state["g5_ultimo_canal"]["sigma"],
                        }
                    ]
                )
            )
        else:
            st.info("Ejecute primero una transmisión en la pestaña Canal discreto.")

        if "g5_resumen_estadistico" in st.session_state:
            st.subheader("Último resumen estadístico")

            st.dataframe(
                pd.DataFrame([st.session_state["g5_resumen_estadistico"]]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Ejecute una simulación estadística para ver el resumen.")

        if "g5_comparacion" in st.session_state:
            st.subheader("Última comparación")

            df_comp = st.session_state["g5_comparacion"]

            mejor_ber = df_comp.loc[df_comp["BER del canal"].idxmin()]
            peor_ber = df_comp.loc[df_comp["BER del canal"].idxmax()]

            st.markdown(
                f"""
- Menor BER observado: **{mejor_ber["BER del canal"]:.6f}** con $\\sigma = {mejor_ber["σ"]:.2f}$.
- Mayor BER observado: **{peor_ber["BER del canal"]:.6f}** con $\\sigma = {peor_ber["σ"]:.2f}$.
"""
            )
        else:
            st.info("Ejecute una comparación de escenarios para ver el análisis.")

        st.subheader("Actividades guiadas")

        st.markdown(
            """
Realice las siguientes actividades:

1. Calcule manualmente el CRC del mensaje `1101` usando el generador `1011`.
2. Identifique $m$, $M(x)$, $G(x)$, $r$, $R(x)$ y $T(x)$.
3. Explique por qué $t = m || R$ significa concatenación y no suma aritmética.
4. Cambie el generador a `10011` y observe cómo cambia $r$.
5. Transmita una trama con $\\sigma = 0.10$ y observe si hay errores.
6. Repita con $\\sigma = 0.80$ y compare el resultado.
7. Ejecute una simulación estadística con 10,000 bits.
8. Compare BER, FER y tasa de detección.
9. Observe si aparecen errores no detectados.
10. Explique por qué CRC detecta errores, pero no los corrige.
11. Relacione esta guía con la limitación de Hamming observada en la Guía 4.
"""
        )

        st.subheader("Preguntas de análisis")

        pregunta_1 = st.radio(
            "Pregunta 1: ¿Cuál es la función principal del CRC?",
            [
                "Corregir automáticamente todos los errores.",
                "Detectar errores en una trama.",
                "Eliminar el ruido del canal.",
                "Reducir la longitud del mensaje.",
            ],
            index=None,
            key="g5_pregunta_1",
        )

        if pregunta_1:
            if pregunta_1 == "Detectar errores en una trama.":
                st.success("Correcto. El CRC detecta errores, pero no los corrige.")
            else:
                st.error("Revise la diferencia entre detección y corrección de errores.")

        pregunta_2 = st.radio(
            "Pregunta 2: Si el residuo CRC en el receptor es distinto de cero, ¿qué se concluye?",
            [
                "No se detecta error.",
                "Se detecta una inconsistencia en la trama.",
                "El mensaje fue corregido.",
                "La SNR es infinita.",
            ],
            index=None,
            key="g5_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "Se detecta una inconsistencia en la trama.":
                st.success("Correcto. Un residuo distinto de cero indica error detectable.")
            else:
                st.error("Revise el proceso de verificación CRC.")

        pregunta_3 = st.radio(
            "Pregunta 3: ¿Cómo se obtiene el residuo CRC R(x)?",
            [
                "Se elige manualmente.",
                "Se obtiene como R(x) = M(x)x^r mod G(x).",
                "Es siempre igual al mensaje original.",
                "Se obtiene sumando decimalmente el mensaje y el generador.",
            ],
            index=None,
            key="g5_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "Se obtiene como R(x) = M(x)x^r mod G(x).":
                st.success("Correcto. El residuo se obtiene por división módulo 2.")
            else:
                st.error("Revise la expresión formal del residuo CRC.")

        pregunta_4 = st.radio(
            "Pregunta 4: ¿De qué depende r en CRC?",
            [
                "Del grado o longitud del generador.",
                "De la cantidad de ruido.",
                "Del número de unos del mensaje.",
                "De la cantidad de errores detectados.",
            ],
            index=None,
            key="g5_pregunta_4",
        )

        if pregunta_4:
            if pregunta_4 == "Del grado o longitud del generador.":
                st.success("Correcto. r = longitud del generador - 1.")
            else:
                st.error("Revise cómo se calcula r.")

        pregunta_5 = st.radio(
            "Pregunta 5: ¿Por qué CRC complementa a Hamming?",
            [
                "Porque CRC corrige los errores múltiples que Hamming no puede corregir.",
                "Porque CRC detecta errores remanentes que Hamming puede no corregir.",
                "Porque CRC reemplaza todos los bits de paridad.",
                "Porque CRC elimina la necesidad de medir BER.",
            ],
            index=None,
            key="g5_pregunta_5",
        )

        if pregunta_5:
            if pregunta_5 == "Porque CRC detecta errores remanentes que Hamming puede no corregir.":
                st.success("Correcto. CRC ayuda a detectar errores que quedan después de la corrección.")
            else:
                st.error("Revise el papel del CRC como detector, no como corrector.")

        pregunta_6 = st.radio(
            "Pregunta 6: ¿Por qué se usan gráficas semilogarítmicas para BER?",
            [
                "Porque BER puede tomar valores pequeños y la escala logarítmica facilita compararlos.",
                "Porque BER siempre es igual a cero.",
                "Porque CRC necesita gráficas continuas.",
                "Porque la escala logarítmica corrige errores.",
            ],
            index=None,
            key="g5_pregunta_6",
        )

        if pregunta_6:
            if pregunta_6 == "Porque BER puede tomar valores pequeños y la escala logarítmica facilita compararlos.":
                st.success("Correcto. La escala logarítmica permite observar diferencias pequeñas de BER.")
            else:
                st.error("Revise la interpretación de BER en escala logarítmica.")

    # ========================================================
    # Conclusiones
    # ========================================================

    with tabs[7]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- el CRC se basa en división módulo 2;
- la resta en módulo 2 se implementa mediante XOR;
- una cadena binaria puede representarse como un polinomio;
- la ecuación formal de construcción CRC es $T(x) = M(x)x^r + R(x)$;
- en forma binaria, la trama puede expresarse como $t = m || R$;
- $||$ significa concatenación, no suma aritmética común;
- el residuo se obtiene como $R(x) = M(x)x^r \\bmod G(x)$;
- $r$ depende del grado del generador;
- al aumentar el número de tramas, aumenta la cantidad total de bits CRC agregados;
- el receptor verifica dividiendo la trama recibida entre el mismo generador;
- un residuo distinto de cero indica error detectable;
- un residuo cero significa que no se detecta error, no garantía absoluta de ausencia de error;
- CRC no corrige errores, solo los detecta;
- el análisis con muchas tramas permite estimar BER, FER y tasa de detección;
- CRC complementa a Hamming porque permite detectar errores remanentes;
- las representaciones de bits y muestras deben tratarse de forma discreta;
- las curvas de desempeño como BER pueden representarse en escala semilogarítmica.

La teoría aplicada en esta guía se fundamenta en técnicas clásicas de detección de
errores, aritmética módulo 2, control de errores y evaluación estadística de sistemas
digitales (Forouzan, 2013; Stallings, 2015; Lin & Costello, 2004).
"""
        )

    # ========================================================
    # Referencias
    # ========================================================

    with tabs[8]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2013). *Data communications and networking* (5th ed.). McGraw-Hill Education.

Lin, S., & Costello, D. J. (2004). *Error control coding* (2nd ed.). Pearson.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Tanenbaum, A. S., & Wetherall, D. J. (2011). *Computer networks* (5th ed.). Pearson.
"""
        )