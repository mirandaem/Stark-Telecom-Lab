import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# Validaciones
# ============================================================

def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_generador(generador: str) -> bool:
    """
    Valida un polinomio generador binario para CRC básico.

    Debe:
    - tener al menos 2 bits;
    - contener solo 0 y 1;
    - iniciar en 1;
    - terminar en 1.
    """
    return (
        len(generador) >= 2
        and all(bit in "01" for bit in generador)
        and generador[0] == "1"
        and generador[-1] == "1"
    )


# ============================================================
# CRC
# ============================================================

def division_modulo_2(dividendo: str, divisor: str) -> Tuple[str, pd.DataFrame]:
    """
    Realiza división módulo 2 para CRC.

    La resta binaria se implementa mediante XOR.
    Devuelve el residuo y una tabla con los pasos.
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
    1. Se agregan r ceros al mensaje, donde r = grado del generador.
    2. Se divide módulo 2 entre el generador.
    3. El residuo se agrega al final del mensaje.
    """
    ceros = "0" * (len(generador) - 1)
    dividendo = datos + ceros

    residuo, pasos = division_modulo_2(dividendo, generador)
    trama = datos + residuo

    return residuo, trama, pasos


def verificar_crc(trama: str, generador: str) -> Tuple[bool, str, pd.DataFrame]:
    """
    Verifica una trama con CRC.

    Si el residuo es cero, no se detecta error.
    Si el residuo es distinto de cero, se detecta error.
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
    """
    tramas = []

    for i in range(0, len(bits), tamano_payload):
        bloque = bits[i:i + tamano_payload]

        if len(bloque) < tamano_payload:
            bloque = bloque + "0" * (tamano_payload - len(bloque))

        tramas.append(bloque)

    return tramas


def bits_a_simbolos_bpsk(bits: str) -> np.ndarray:
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
    Simula muchas tramas con CRC a través de canal AWGN.

    Métricas calculadas:
    - BER del canal;
    - FER;
    - tasa de detección;
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

def graficar_trama_discreta(trama_tx: str, trama_rx: str):
    posiciones = np.arange(1, len(trama_tx) + 1)
    tx = np.array([int(bit) for bit in trama_tx])
    rx = np.array([int(bit) for bit in trama_rx])

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.stem(
        posiciones,
        tx,
        linefmt="C0-",
        markerfmt="C0o",
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


def graficar_senal_crc_discreta(
    simbolos: np.ndarray,
    ruido: np.ndarray,
    recibido_analogico: np.ndarray,
):
    posiciones = np.arange(1, len(simbolos) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.stem(
        posiciones,
        simbolos,
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
        label="Símbolo transmitido",
    )

    ax.scatter(
        posiciones,
        recibido_analogico,
        marker="x",
        label="Valor recibido",
    )

    ax.axhline(0, linestyle="--", linewidth=1, label="Umbral")

    ax.set_title("Símbolos y valores recibidos por muestra")
    ax.set_xlabel("Índice de bit")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    fig_ruido, ax_ruido = plt.subplots(figsize=(10, 3))

    ax_ruido.stem(
        posiciones,
        ruido,
        linefmt="C1-",
        markerfmt="C1o",
        basefmt=" ",
    )

    ax_ruido.axhline(0, linestyle="--", linewidth=1)
    ax_ruido.set_title("Ruido gaussiano por muestra")
    ax_ruido.set_xlabel("Índice de bit")
    ax_ruido.set_ylabel("Ruido")
    ax_ruido.grid(True)

    st.pyplot(fig_ruido)


def graficar_ber_vs_sigma(df: pd.DataFrame):
    df_plot = df.copy()
    df_plot["BER ajustado"] = df_plot["BER del canal"].replace(0, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["σ"], df_plot["BER ajustado"], marker="o")

    ax.set_title("BER del canal vs σ")
    ax.set_xlabel("Desviación estándar σ")
    ax.set_ylabel("BER en escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)


def graficar_fer_vs_sigma(df: pd.DataFrame):
    df_plot = df.copy()
    df_plot["FER ajustado"] = df_plot["FER"].replace(0, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["σ"], df_plot["FER ajustado"], marker="o")

    ax.set_title("FER vs σ")
    ax.set_xlabel("Desviación estándar σ")
    ax.set_ylabel("FER en escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)


def graficar_no_detectados_vs_sigma(df: pd.DataFrame):
    df_plot = df.copy()
    df_plot["No detectado ajustado"] = df_plot["Tasa de error no detectado"].replace(0, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["σ"], df_plot["No detectado ajustado"], marker="o")

    ax.set_title("Tasa de error no detectado vs σ")
    ax.set_xlabel("Desviación estándar σ")
    ax.set_ylabel("Tasa en escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)


def graficar_ber_vs_snr(df: pd.DataFrame):
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


# ============================================================
# Interfaz Streamlit
# ============================================================

def render_guia_05() -> None:
    st.title("Guía 5: CRC y detección de errores remanentes")

    st.markdown(
        """
Esta guía estudia el Código de Redundancia Cíclica (CRC) como mecanismo de detección
de errores. A diferencia de Hamming, CRC no corrige bits alterados, sino que permite
identificar si una trama recibida presenta inconsistencias.

La guía combina el procedimiento algebraico del CRC con simulaciones estadísticas sobre
muchas tramas transmitidas por un canal con ruido gaussiano. Además, mantiene una
representación discreta de bits y muestras para evitar interpretar las tramas digitales
como señales continuas.
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

1. Representar una secuencia binaria mediante división módulo 2.
2. Calcular el residuo CRC a partir de un polinomio generador.
3. Construir una trama transmitida formada por datos y residuo CRC.
4. Verificar una trama recibida usando el mismo generador.
5. Observar el efecto del ruido sobre una trama con CRC mediante muestras discretas.
6. Evaluar muchas tramas para estimar BER, FER y tasa de detección.
7. Analizar errores detectados y errores no detectados.
8. Relacionar CRC con los límites de Hamming frente a errores múltiples.
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
errores basada en aritmética binaria módulo 2. En este tipo de operación, la suma y
la resta se realizan mediante XOR.

El transmisor toma un mensaje binario y lo divide entre un polinomio generador. El
residuo de esa división se agrega al final del mensaje para formar la trama transmitida.

Si el mensaje es $M(x)$ y el polinomio generador es $G(x)$, la trama transmitida puede
expresarse como:

$$
T(x) = M(x)x^r + R(x)
$$

donde:

- $r$ es el grado del polinomio generador;
- $R(x)$ es el residuo CRC;
- $T(x)$ es la trama transmitida.

En el receptor se divide la trama recibida entre el mismo generador:

$$
Residuo\\left(\\frac{T(x)}{G(x)}\\right)
$$

Si el residuo es cero, no se detecta error:

$$
Residuo = 0
$$

Si el residuo es diferente de cero, se detecta una inconsistencia:

$$
Residuo \\neq 0
$$

El CRC no corrige errores. Su función es detectar que la trama recibida no cumple la
relación esperada con el generador. Esto es especialmente importante después de observar
en la Guía 4 que Hamming puede fallar ante errores múltiples dentro del mismo bloque.

Para evaluar estadísticamente el desempeño se utilizan:

$$
BER = \\frac{\\text{bits erróneos}}{\\text{bits transmitidos}}
$$

$$
FER = \\frac{\\text{tramas con error}}{\\text{tramas transmitidas}}
$$

También se analiza la razón señal-ruido:

$$
SNR = \\frac{P_s}{P_n}
$$

y la relación entre la varianza del ruido y su potencia promedio:

$$
P_n \\approx \\sigma^2
$$
"""
        )

        st.info(
            """
Idea central: CRC detecta errores, pero no indica qué bit se dañó ni cómo corregirlo.
Por eso complementa a Hamming: Hamming corrige errores simples, mientras que CRC permite
verificar si quedan errores remanentes.
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

            ejecutar = st.button("Calcular CRC", width="stretch")

        with col_info:
            st.info(
                """
Ejemplo típico:

Datos: 1101  
Generador: 1011  

Si el generador tiene longitud 4, se agregan 3 ceros al mensaje antes de dividir.
"""
            )

        if not validar_bits(datos):
            st.error("Los datos deben contener únicamente 0 y 1.")
        elif not validar_generador(generador):
            st.error("El generador debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar:
            residuo, trama, pasos = generar_crc(datos, generador)

            st.subheader("Resultado")

            c1, c2, c3 = st.columns(3)
            c1.metric("Longitud del generador", len(generador))
            c2.metric("Bits CRC", len(generador) - 1)
            c3.metric("Residuo", residuo)

            st.code(
                f"Datos:              {datos}\n"
                f"Generador:          {generador}\n"
                f"Datos con ceros:    {datos + '0' * (len(generador) - 1)}\n"
                f"Residuo CRC:        {residuo}\n"
                f"Trama transmitida:  {trama}",
                language="text",
            )

            st.subheader("Pasos de la división módulo 2")
            st.dataframe(pasos, width="stretch", hide_index=True)

            st.session_state["g5_trama_manual_resultado"] = trama
            st.session_state["g5_generador_manual_resultado"] = generador

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

Las gráficas se muestran como muestras discretas, no como señales continuas.
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

            ejecutar_senal = st.button("Transmitir trama con CRC", width="stretch")

        with col_info:
            st.info(
                """
El receptor no conoce la trama original. Solo observa los bits decididos después del
ruido y aplica el CRC. Si el residuo no es cero, detecta error.
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
            st.dataframe(tabla_bits, width="stretch", hide_index=True)

            st.subheader("Verificación CRC en receptor")
            st.dataframe(pasos_rx, width="stretch", hide_index=True)

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

            ejecutar_est = st.button("Ejecutar simulación estadística", width="stretch")

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
            st.dataframe(pd.DataFrame([resumen]), width="stretch", hide_index=True)

            st.subheader("Primeras tramas evaluadas")
            st.dataframe(df_resultados.head(25), width="stretch", hide_index=True)

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

Las métricas de error se muestran en escala semilogarítmica cuando corresponde.
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

            ejecutar_comp = st.button("Comparar escenarios", width="stretch")

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
            st.dataframe(df_comp, width="stretch", hide_index=True)

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
El objetivo es que el estudiante comprenda la función del CRC como detector y sus límites.
"""
        )

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
                width="stretch",
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
2. Compare sus pasos con la tabla de división módulo 2 de la app.
3. Transmita una trama con $\\sigma = 0.10$ y observe si hay errores.
4. Repita con $\\sigma = 0.80$ y compare el resultado.
5. Ejecute una simulación estadística con 10,000 bits.
6. Compare BER, FER y tasa de detección.
7. Observe si aparecen errores no detectados.
8. Explique por qué CRC detecta errores, pero no los corrige.
9. Relacione esta guía con la limitación de Hamming observada en la Guía 4.
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
            "Pregunta 3: ¿Por qué se analiza CRC con muchas tramas?",
            [
                "Para obtener métricas estadísticas como BER, FER y tasa de detección.",
                "Para evitar calcular el residuo.",
                "Porque CRC solo funciona con mensajes largos.",
                "Porque con pocas tramas no existe ruido.",
            ],
            index=None,
            key="g5_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "Para obtener métricas estadísticas como BER, FER y tasa de detección.":
                st.success("Correcto. Muchas tramas permiten evaluar el desempeño de forma estadística.")
            else:
                st.error("Revise la importancia de las muestras grandes en simulación.")

        pregunta_4 = st.radio(
            "Pregunta 4: ¿Por qué CRC complementa a Hamming?",
            [
                "Porque CRC corrige los errores múltiples que Hamming no puede corregir.",
                "Porque CRC detecta errores remanentes que Hamming puede no corregir.",
                "Porque CRC reemplaza todos los bits de paridad.",
                "Porque CRC elimina la necesidad de medir BER.",
            ],
            index=None,
            key="g5_pregunta_4",
        )

        if pregunta_4:
            if pregunta_4 == "Porque CRC detecta errores remanentes que Hamming puede no corregir.":
                st.success("Correcto. CRC ayuda a detectar errores que quedan después de la corrección.")
            else:
                st.error("Revise el papel del CRC como detector, no como corrector.")

        pregunta_5 = st.radio(
            "Pregunta 5: ¿Por qué se usan gráficas semilogarítmicas para BER?",
            [
                "Porque BER puede tomar valores pequeños y la escala logarítmica facilita compararlos.",
                "Porque BER siempre es igual a cero.",
                "Porque CRC necesita gráficas continuas.",
                "Porque la escala logarítmica corrige errores.",
            ],
            index=None,
            key="g5_pregunta_5",
        )

        if pregunta_5:
            if pregunta_5 == "Porque BER puede tomar valores pequeños y la escala logarítmica facilita compararlos.":
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
- el residuo CRC se agrega a los datos antes de transmitir;
- el receptor verifica la trama repitiendo la división;
- un residuo distinto de cero indica error detectable;
- CRC no corrige errores, solo los detecta;
- el análisis con muchas tramas permite estimar BER, FER y tasa de detección;
- la presencia de errores no detectados muestra que ningún detector es absoluto;
- CRC complementa a Hamming porque permite detectar errores remanentes o no corregibles;
- las representaciones de bits y muestras deben tratarse de forma discreta;
- las curvas de desempeño como BER pueden representarse en escala semilogarítmica.
"""
        )

    # ========================================================
    # Referencias
    # ========================================================

    with tabs[8]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2012). *Data communications and networking* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Tanenbaum, A. S., & Wetherall, D. J. (2011). *Computer networks* (5th ed.). Pearson.

Lin, S., & Costello, D. J. (1983). *Error control coding: Fundamentals and applications*. Prentice-Hall.
"""
        )