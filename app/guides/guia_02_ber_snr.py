import math
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# Funciones de simulación
# ============================================================

def generar_bits_aleatorios(cantidad_bits: int, semilla: int | None = None) -> np.ndarray:
    """
    Genera una secuencia aleatoria de bits 0 y 1.
    """
    rng = np.random.default_rng(semilla)
    return rng.integers(0, 2, size=cantidad_bits)


def bits_a_simbolos_bpsk(bits: np.ndarray) -> np.ndarray:
    """
    Mapea bits a símbolos BPSK:
    0 -> -1
    1 -> +1
    """
    return np.where(bits == 1, 1.0, -1.0)


def agregar_ruido_awgn(
    simbolos: np.ndarray,
    sigma: float,
    semilla: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Agrega ruido gaussiano de media cero y desviación estándar sigma.
    """
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(loc=0.0, scale=sigma, size=len(simbolos))
    recibido = simbolos + ruido
    return ruido, recibido


def decidir_bits_por_umbral(valores_recibidos: np.ndarray) -> np.ndarray:
    """
    Regla de decisión:
    r >= 0 -> bit 1
    r < 0  -> bit 0
    """
    return np.where(valores_recibidos >= 0, 1, 0)


def calcular_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> tuple[int, float]:
    """
    Calcula el número de errores y la tasa de error de bit.
    """
    errores = int(np.sum(bits_tx != bits_rx))
    ber = errores / len(bits_tx)
    return errores, ber


def calcular_metricas(
    simbolos: np.ndarray,
    ruido: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Calcula potencia promedio de señal, potencia promedio de ruido,
    SNR lineal, SNR dB, Eb/N0 lineal y Eb/N0 dB.

    En este modelo simplificado:
    - Los símbolos BPSK tienen energía aproximada unitaria.
    - Por tanto, Eb se aproxima a 1.
    - Si el ruido es gaussiano de media cero, su potencia se aproxima a sigma^2.
    """
    potencia_senal = float(np.mean(simbolos**2))
    potencia_ruido = float(np.mean(ruido**2))

    if potencia_ruido == 0:
        snr_lineal = math.inf
        snr_db = math.inf
        ebn0_lineal = math.inf
        ebn0_db = math.inf
    else:
        snr_lineal = potencia_senal / potencia_ruido
        snr_db = 10 * math.log10(snr_lineal)

        # En este modelo didáctico BPSK con amplitud ±1,
        # Eb se aproxima a la energía promedio por bit.
        ebn0_lineal = snr_lineal
        ebn0_db = snr_db

    return potencia_senal, potencia_ruido, snr_lineal, snr_db, ebn0_lineal, ebn0_db


def simular_escenario(
    cantidad_bits: int,
    sigma: float,
    semilla: int | None = None,
) -> dict:
    """
    Ejecuta una simulación completa para un valor de sigma.
    """
    bits_tx = generar_bits_aleatorios(cantidad_bits, semilla)
    simbolos = bits_a_simbolos_bpsk(bits_tx)
    ruido, recibido = agregar_ruido_awgn(simbolos, sigma, semilla)
    bits_rx = decidir_bits_por_umbral(recibido)

    errores, ber = calcular_ber(bits_tx, bits_rx)

    (
        potencia_senal,
        potencia_ruido,
        snr_lineal,
        snr_db,
        ebn0_lineal,
        ebn0_db,
    ) = calcular_metricas(simbolos, ruido)

    return {
        "Bits simulados": cantidad_bits,
        "σ": sigma,
        "σ²": sigma**2,
        "Potencia señal": potencia_senal,
        "Potencia ruido": potencia_ruido,
        "SNR": snr_lineal,
        "SNR dB": snr_db,
        "Eb/N0": ebn0_lineal,
        "Eb/N0 dB": ebn0_db,
        "Errores": errores,
        "BER": ber,
    }


def simular_varios_escenarios(
    cantidad_bits: int,
    valores_sigma: List[float],
    semilla: int | None = None,
) -> pd.DataFrame:
    """
    Ejecuta múltiples simulaciones para varios valores de sigma.
    """
    resultados = []

    for i, sigma in enumerate(valores_sigma):
        semilla_escenario = None if semilla is None else semilla + i
        resultado = simular_escenario(cantidad_bits, sigma, semilla_escenario)
        resultados.append(resultado)

    return pd.DataFrame(resultados)


def comparar_semillas(
    cantidad_bits: int,
    sigma: float,
    semillas: List[int],
) -> pd.DataFrame:
    """
    Ejecuta la misma simulación con varias semillas para observar variación estadística.
    """
    filas = []

    for semilla in semillas:
        resultado = simular_escenario(cantidad_bits, sigma, semilla)
        filas.append(
            {
                "Semilla": semilla,
                "Bits simulados": cantidad_bits,
                "σ": sigma,
                "σ²": sigma**2,
                "Errores": resultado["Errores"],
                "BER": resultado["BER"],
                "SNR dB": resultado["SNR dB"],
                "Eb/N0 dB": resultado["Eb/N0 dB"],
            }
        )

    return pd.DataFrame(filas)


# ============================================================
# Funciones de gráficas
# ============================================================

def graficar_muestras_discretas(bits_tx: np.ndarray, bits_rx: np.ndarray, max_muestras: int = 80):
    """
    Grafica bits transmitidos y recibidos como muestras discretas.
    """
    n = min(len(bits_tx), max_muestras)
    posiciones = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.stem(posiciones, bits_tx[:n], linefmt="C0-", markerfmt="C0o", basefmt=" ")
    ax.scatter(posiciones, bits_rx[:n], marker="x", label="Bit decidido")

    ax.set_title("Muestras discretas de bits transmitidos y decididos")
    ax.set_xlabel("Índice de bit")
    ax.set_ylabel("Valor del bit")
    ax.set_yticks([0, 1])
    ax.grid(True)
    ax.legend(["Bit transmitido", "Bit decidido"])

    st.pyplot(fig)


def graficar_ber_vs_sigma2(df: pd.DataFrame):
    """
    Grafica BER vs sigma^2 con escala logarítmica en Y.
    """
    df_plot = df.copy()
    df_plot["BER ajustado"] = df_plot["BER"].replace(0, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["σ²"], df_plot["BER ajustado"], marker="o")

    ax.set_title("BER vs varianza del ruido σ²")
    ax.set_xlabel("Varianza del ruido σ²")
    ax.set_ylabel("BER escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)


def graficar_ber_vs_ebn0(df: pd.DataFrame):
    """
    Grafica BER vs Eb/N0 dB con escala logarítmica en Y.
    """
    df_plot = df.copy()
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["Eb/N0 dB"])
    df_plot["BER ajustado"] = df_plot["BER"].replace(0, 1e-6)

    if df_plot.empty:
        st.info("No se puede graficar BER vs Eb/N0 cuando Eb/N0 es infinito.")
        return

    df_plot = df_plot.sort_values("Eb/N0 dB")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["Eb/N0 dB"], df_plot["BER ajustado"], marker="o")

    ax.set_title("BER vs Eb/N0")
    ax.set_xlabel("Eb/N0 en dB")
    ax.set_ylabel("BER escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)


def graficar_ber_vs_snr(df: pd.DataFrame):
    """
    Grafica BER vs SNR dB con escala logarítmica en Y.
    """
    df_plot = df.copy()
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["SNR dB"])
    df_plot["BER ajustado"] = df_plot["BER"].replace(0, 1e-6)

    if df_plot.empty:
        st.info("No se puede graficar BER vs SNR cuando la SNR es infinita.")
        return

    df_plot = df_plot.sort_values("SNR dB")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(df_plot["SNR dB"], df_plot["BER ajustado"], marker="o")

    ax.set_title("BER vs SNR")
    ax.set_xlabel("SNR en dB")
    ax.set_ylabel("BER escala logarítmica")
    ax.grid(True, which="both")

    st.pyplot(fig)


# ============================================================
# Interfaz Streamlit
# ============================================================

def render_guia_02() -> None:
    st.title("Guía 2: BER, SNR y análisis estadístico con muchos bits")

    st.markdown(
        """
Esta guía estudia el desempeño de una transmisión digital afectada por ruido utilizando
muestras grandes de bits. El propósito es medir estadísticamente el efecto del ruido
mediante BER, SNR, varianza y una aproximación didáctica de Eb/N0.

A diferencia de la Guía 1, donde se observó el fenómeno con pocas muestras, aquí se busca
obtener tendencias más representativas mediante simulaciones con muchos bits.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Simulación estadística",
            "Comparación",
            "Análisis y dinámica",
            "Conclusiones",
            "Referencias",
        ]
    )

    with tabs[0]:
        st.header("Objetivos")

        st.markdown(
            """
**Objetivo general**

Analizar estadísticamente el desempeño de una transmisión digital afectada por ruido
mediante el cálculo de BER, SNR, varianza del ruido y Eb/N0.

**Objetivos específicos**

1. Generar secuencias binarias aleatorias de gran tamaño.
2. Simular una transmisión BPSK con ruido gaussiano.
3. Calcular BER experimental.
4. Relacionar desviación estándar, varianza y potencia del ruido.
5. Calcular SNR en escala lineal y en decibeles.
6. Introducir la relación Eb/N0 como métrica de desempeño en comunicaciones digitales.
7. Usar gráficas semilogarítmicas para analizar BER.
8. Comparar resultados bajo diferentes semillas y niveles de ruido.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En comunicaciones digitales, el desempeño del sistema se evalúa usualmente mediante
métricas estadísticas. Una de las más importantes es la tasa de error de bit:

$$
BER = \\frac{N_e}{N_t}
$$

donde:

- $N_e$ es el número de bits erróneos;
- $N_t$ es el número total de bits transmitidos.

El canal se modela como:

$$
r = s + n
$$

donde $s$ es el símbolo transmitido, $n$ es el ruido y $r$ es el valor recibido.

En esta guía se utiliza ruido gaussiano:

$$
n \\sim \\mathcal{N}(0,\\sigma^2)
$$

La desviación estándar $\\sigma$ controla la dispersión del ruido y la varianza se define como:

$$
\\sigma^2
$$

En el modelo de ruido de media cero, la potencia promedio del ruido puede aproximarse como:

$$
P_n \\approx \\sigma^2
$$

La razón señal-ruido se define como:

$$
SNR = \\frac{P_s}{P_n}
$$

y en decibeles:

$$
SNR_{dB} = 10\\log_{10}(SNR)
$$

Otra métrica usada en comunicaciones digitales es:

$$
\\frac{E_b}{N_0}
$$

donde:

- $E_b$ es la energía por bit;
- $N_0$ es la densidad espectral de potencia del ruido.

En este modelo didáctico BPSK con símbolos $+1$ y $-1$, la energía por bit se aproxima
como unitaria, por lo que $E_b/N_0$ se utiliza como una medida equivalente de desempeño
para comparar el efecto del ruido.

Las curvas de BER suelen representarse en escala semilogarítmica porque el BER puede tomar
valores muy pequeños. Esta escala permite observar diferencias que no serían visibles en
una escala lineal.
"""
        )

        st.info(
            """
Nota didáctica: En esta guía Eb/N0 se usa como aproximación educativa asociada al modelo BPSK
normalizado. En un sistema físico completo, su cálculo requeriría considerar energía de bit,
ancho de banda, densidad espectral de ruido y otros parámetros del sistema.
"""
        )

    with tabs[2]:
        st.header("Simulación estadística individual")

        st.markdown(
            """
En esta sección se ejecuta una simulación con una cantidad seleccionada de bits.
La gráfica de bits se limita a las primeras muestras para mantener la visualización clara
y discreta.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits = st.selectbox(
                "Cantidad de bits",
                [100, 1000, 5000, 10000],
                index=2,
                key="g2_bits_individual",
            )

            sigma = st.slider(
                "Desviación estándar del ruido σ",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.05,
                key="g2_sigma_individual",
            )

            semilla = st.number_input(
                "Semilla",
                min_value=0,
                max_value=999999,
                value=123,
                step=1,
                key="g2_semilla_individual",
            )

            ejecutar = st.button("Ejecutar simulación individual", width="stretch")

        with col_info:
            st.info(
                """
La semilla permite repetir el mismo experimento aleatorio.  
Si se mantienen los mismos parámetros y la misma semilla, el ruido y los bits generados
serán los mismos.
"""
            )

            st.metric("Varianza teórica σ²", f"{sigma**2:.4f}")

        if ejecutar:
            bits_tx = generar_bits_aleatorios(cantidad_bits, int(semilla))
            simbolos = bits_a_simbolos_bpsk(bits_tx)
            ruido, recibido = agregar_ruido_awgn(simbolos, sigma, int(semilla))
            bits_rx = decidir_bits_por_umbral(recibido)

            errores, ber = calcular_ber(bits_tx, bits_rx)

            (
                potencia_senal,
                potencia_ruido,
                snr_lineal,
                snr_db,
                ebn0_lineal,
                ebn0_db,
            ) = calcular_metricas(simbolos, ruido)

            resultado = {
                "Bits simulados": cantidad_bits,
                "σ": sigma,
                "σ²": sigma**2,
                "Potencia señal": potencia_senal,
                "Potencia ruido": potencia_ruido,
                "SNR": snr_lineal,
                "SNR dB": snr_db,
                "Eb/N0": ebn0_lineal,
                "Eb/N0 dB": ebn0_db,
                "Errores": errores,
                "BER": ber,
            }

            st.subheader("Resultados principales")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits simulados", f"{cantidad_bits:,}")
            c2.metric("Errores", errores)
            c3.metric("BER", f"{ber:.6f}")
            c4.metric("σ²", f"{sigma**2:.4f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Potencia ruido", f"{potencia_ruido:.4f}")
            c6.metric("SNR dB", "∞" if math.isinf(snr_db) else f"{snr_db:.2f}")
            c7.metric("Eb/N0 dB", "∞" if math.isinf(ebn0_db) else f"{ebn0_db:.2f}")
            c8.metric("Semilla", int(semilla))

            st.subheader("Tabla de métricas")
            st.dataframe(pd.DataFrame([resultado]), width="stretch", hide_index=True)

            st.subheader("Primeras muestras discretas")
            graficar_muestras_discretas(bits_tx, bits_rx)

            st.session_state["guia_02_resultado_individual"] = resultado

            st.markdown(
                f"""
**Lectura del resultado**

Para $\\sigma = {sigma:.2f}$, la varianza teórica del ruido es:

$$
\\sigma^2 = {sigma**2:.4f}
$$

El BER experimental obtenido fue:

$$
BER = {ber:.6f}
$$

Este resultado corresponde a una realización específica del experimento, definida por la semilla.
"""
            )

    with tabs[3]:
        st.header("Comparación de escenarios")

        st.markdown(
            """
En esta sección se comparan diferentes valores de desviación estándar.  
Las curvas de BER se muestran en escala semilogarítmica, como es habitual en análisis
de desempeño de sistemas digitales.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits_comp = st.selectbox(
                "Cantidad de bits por escenario",
                [100, 1000, 5000, 10000],
                index=3,
                key="g2_bits_comparacion",
            )

            valores_sigma_texto = st.text_input(
                "Valores de σ separados por coma",
                value="0.10, 0.20, 0.30, 0.50, 0.80, 1.00",
                key="g2_valores_sigma",
            )

            semilla_comp = st.number_input(
                "Semilla base",
                min_value=0,
                max_value=999999,
                value=500,
                step=1,
                key="g2_semilla_comp",
            )

            ejecutar_comp = st.button("Ejecutar comparación", width="stretch")

        with col_info:
            st.info(
                """
La comparación permite observar tendencias estadísticas:

- al aumentar σ, aumenta σ²;
- al aumentar σ², aumenta la potencia del ruido;
- al aumentar el ruido, disminuyen SNR y Eb/N0;
- cuando disminuye SNR, normalmente aumenta BER.
"""
            )

        if ejecutar_comp:
            try:
                valores_sigma = [
                    float(valor.strip())
                    for valor in valores_sigma_texto.split(",")
                    if valor.strip() != ""
                ]

                if len(valores_sigma) == 0:
                    st.error("Debe ingresar al menos un valor de σ.")
                    return

                if any(valor < 0 for valor in valores_sigma):
                    st.error("Los valores de σ no pueden ser negativos.")
                    return

            except ValueError:
                st.error("Ingrese valores numéricos válidos separados por coma.")
                return

            df_comparacion = simular_varios_escenarios(
                cantidad_bits=cantidad_bits_comp,
                valores_sigma=valores_sigma,
                semilla=int(semilla_comp),
            )

            st.subheader("Tabla comparativa")

            st.dataframe(df_comparacion, width="stretch", hide_index=True)

            st.subheader("BER vs σ²")
            graficar_ber_vs_sigma2(df_comparacion)

            st.subheader("BER vs SNR dB")
            graficar_ber_vs_snr(df_comparacion)

            st.subheader("BER vs Eb/N0 dB")
            graficar_ber_vs_ebn0(df_comparacion)

            st.session_state["guia_02_comparacion"] = df_comparacion

            st.markdown(
                """
**Lectura esperada**

Las curvas de BER se presentan con eje vertical logarítmico. Si algún valor de BER es cero,
se sustituye solo para la gráfica por un valor muy pequeño, porque el logaritmo de cero
no está definido.
"""
            )

    with tabs[4]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección combina interpretación, actividades guiadas y preguntas conceptuales.
"""
        )

        if "guia_02_resultado_individual" in st.session_state:
            st.subheader("Última simulación individual")

            st.dataframe(
                pd.DataFrame([st.session_state["guia_02_resultado_individual"]]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Ejecute primero una simulación individual.")

        if "guia_02_comparacion" in st.session_state:
            df = st.session_state["guia_02_comparacion"]

            st.subheader("Resumen de la última comparación")

            mejor_ber = df.loc[df["BER"].idxmin()]
            peor_ber = df.loc[df["BER"].idxmax()]

            st.markdown(
                f"""
- Menor BER observado: **{mejor_ber["BER"]:.6f}** con $\\sigma = {mejor_ber["σ"]:.2f}$.
- Mayor BER observado: **{peor_ber["BER"]:.6f}** con $\\sigma = {peor_ber["σ"]:.2f}$.
"""
            )
        else:
            st.info("Ejecute primero una comparación de escenarios.")

        st.subheader("Comparación por semillas")

        col_seed, col_seed_info = st.columns([1, 1])

        with col_seed:
            bits_semilla = st.selectbox(
                "Bits para comparar semillas",
                [100, 1000, 5000, 10000],
                index=2,
                key="g2_bits_semilla",
            )

            sigma_semilla = st.slider(
                "σ para comparación de semillas",
                min_value=0.0,
                max_value=2.0,
                value=0.50,
                step=0.05,
                key="g2_sigma_semilla",
            )

            semillas_texto = st.text_input(
                "Semillas separadas por coma",
                value="10, 20, 30, 40, 50",
                key="g2_semillas_texto",
            )

            ejecutar_semillas = st.button("Comparar semillas", width="stretch")

        with col_seed_info:
            st.info(
                """
Cambiar la semilla cambia la realización aleatoria del experimento.  
Con pocos bits, el BER puede variar más entre semillas.  
Con más bits, la estimación tiende a estabilizarse.
"""
            )

        if ejecutar_semillas:
            try:
                semillas = [
                    int(valor.strip())
                    for valor in semillas_texto.split(",")
                    if valor.strip() != ""
                ]
            except ValueError:
                st.error("Ingrese semillas numéricas enteras separadas por coma.")
                return

            df_semillas = comparar_semillas(
                cantidad_bits=bits_semilla,
                sigma=sigma_semilla,
                semillas=semillas,
            )

            st.dataframe(df_semillas, width="stretch", hide_index=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(df_semillas["Semilla"].astype(str), df_semillas["BER"])

            ax.set_title("BER obtenido con distintas semillas")
            ax.set_xlabel("Semilla")
            ax.set_ylabel("BER")
            ax.grid(True)

            st.pyplot(fig)

        st.subheader("Actividades guiadas")

        st.markdown(
            """
Realice las siguientes pruebas:

1. Ejecute una simulación con 100 bits y $\\sigma = 0.50$.
2. Repita con 10,000 bits y el mismo $\\sigma$.
3. Compare la estabilidad del BER.
4. Ejecute la comparación para varios valores de $\\sigma$.
5. Observe las curvas semilogarítmicas.
6. Cambie las semillas y observe la variación de resultados.
7. Explique por qué el BER no debe estimarse con una sola muestra pequeña.
"""
        )

        st.subheader("Preguntas de análisis")

        pregunta_1 = st.radio(
            "Pregunta 1: ¿Qué ocurre normalmente con el BER cuando aumenta σ?",
            [
                "Tiende a disminuir.",
                "Tiende a aumentar.",
                "Permanece siempre en cero.",
                "No tiene ninguna relación con el ruido.",
            ],
            index=None,
            key="g2_pregunta_1",
        )

        if pregunta_1:
            if pregunta_1 == "Tiende a aumentar.":
                st.success("Correcto. Al aumentar σ, el ruido tiene mayor dispersión y puede provocar más errores.")
            else:
                st.error("Revise la relación entre desviación estándar, ruido y decisiones incorrectas.")

        pregunta_2 = st.radio(
            "Pregunta 2: ¿Qué representa σ²?",
            [
                "La media del ruido.",
                "La varianza del ruido.",
                "El número de bits transmitidos.",
                "El residuo CRC.",
            ],
            index=None,
            key="g2_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "La varianza del ruido.":
                st.success("Correcto. σ² representa la varianza del ruido.")
            else:
                st.error("Revise la definición de varianza.")

        pregunta_3 = st.radio(
            "Pregunta 3: ¿Por qué se usa escala logarítmica para BER?",
            [
                "Porque BER puede tomar valores muy pequeños y la escala logarítmica facilita la comparación.",
                "Porque BER siempre es mayor que 1.",
                "Porque elimina los errores.",
                "Porque evita calcular SNR.",
            ],
            index=None,
            key="g2_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "Porque BER puede tomar valores muy pequeños y la escala logarítmica facilita la comparación.":
                st.success("Correcto. Las curvas de BER suelen representarse en escala semilogarítmica.")
            else:
                st.error("Revise por qué BER se representa comúnmente en escala logarítmica.")

        pregunta_4 = st.radio(
            "Pregunta 4: ¿Qué permite la semilla en la simulación?",
            [
                "Eliminar completamente el ruido.",
                "Repetir el experimento aleatorio bajo los mismos parámetros.",
                "Convertir Hamming en CRC.",
                "Aumentar automáticamente la SNR.",
            ],
            index=None,
            key="g2_pregunta_4",
        )

        if pregunta_4:
            if pregunta_4 == "Repetir el experimento aleatorio bajo los mismos parámetros.":
                st.success("Correcto. La semilla permite reproducibilidad.")
            else:
                st.error("Revise el concepto de semilla en simulaciones aleatorias.")

    with tabs[5]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- el BER experimental debe estimarse con una cantidad suficiente de bits;
- la desviación estándar $\\sigma$ controla la dispersión del ruido;
- la varianza $\\sigma^2$ se relaciona con la potencia promedio del ruido;
- la SNR mide la relación entre potencia de señal y potencia de ruido;
- $E_b/N_0$ es una métrica importante para comparar desempeño en comunicaciones digitales;
- al aumentar el ruido, disminuyen SNR y $E_b/N_0$;
- al disminuir SNR o $E_b/N_0$, el BER tiende a aumentar;
- las gráficas semilogarítmicas permiten observar mejor diferencias pequeñas de BER;
- las semillas permiten repetir experimentos y analizar variabilidad estadística.
"""
        )

    with tabs[6]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2012). *Data communications and networking* (5th ed.). McGraw-Hill.

Proakis, J. G., & Salehi, M. (2008). *Digital communications* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.
"""
        )