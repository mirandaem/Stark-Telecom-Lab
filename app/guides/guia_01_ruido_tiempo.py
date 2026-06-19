import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Utilidades generales
# ============================================================

def limpiar_bits(bits: str) -> str:
    return bits.strip().replace(" ", "").replace("\n", "").replace("\t", "")


def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def generar_bits_aleatorios(cantidad: int, semilla: int | None = None) -> str:
    rng = np.random.default_rng(semilla)
    bits = rng.integers(0, 2, size=cantidad)
    return "".join(str(bit) for bit in bits)


def bits_a_bpsk(bits: str) -> np.ndarray:
    bits_array = np.fromiter((int(bit) for bit in bits), dtype=int)
    return np.where(bits_array == 1, 1.0, -1.0)


def transmitir_awgn(
    bits: str,
    sigma: float,
    semilla: int | None = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Modelo base:

        r = s + n

    donde:
        s = símbolo transmitido
        n = ruido gaussiano
        r = valor recibido

    En esta app se usa BPSK normalizado:
        0 -> -1
        1 -> +1

    Por ello, la amplitud se interpreta como amplitud normalizada.
    """
    if len(bits) == 0:
        return "", np.array([]), np.array([]), np.array([])

    rng = np.random.default_rng(semilla)

    simbolos = bits_a_bpsk(bits)
    ruido = rng.normal(loc=0.0, scale=sigma, size=len(simbolos))
    recibido = simbolos + ruido
    bits_rx_array = np.where(recibido >= 0, 1, 0)
    bits_rx = "".join(str(bit) for bit in bits_rx_array)

    return bits_rx, simbolos, ruido, recibido


def contar_errores(bits_tx: str, bits_rx: str) -> int:
    longitud = min(len(bits_tx), len(bits_rx))

    if longitud == 0:
        return 0

    return sum(1 for a, b in zip(bits_tx[:longitud], bits_rx[:longitud]) if a != b)


def calcular_ber(bits_tx: str, bits_rx: str) -> float:
    longitud = min(len(bits_tx), len(bits_rx))

    if longitud == 0:
        return 0.0

    return contar_errores(bits_tx, bits_rx) / longitud


def calcular_metricas(
    simbolos: np.ndarray,
    ruido: np.ndarray,
    bits_tx: str,
    bits_rx: str,
) -> dict:
    potencia_senal = float(np.mean(simbolos**2)) if len(simbolos) > 0 else 0.0
    potencia_ruido = float(np.mean(ruido**2)) if len(ruido) > 0 else 0.0

    if potencia_ruido > 0:
        snr_lineal = potencia_senal / potencia_ruido
        snr_db = (
            10 * math.log10(snr_lineal)
            if np.isfinite(snr_lineal) and snr_lineal > 0
            else math.inf
        )
    else:
        snr_lineal = math.inf
        snr_db = math.inf

    errores = contar_errores(bits_tx, bits_rx)
    ber = calcular_ber(bits_tx, bits_rx)

    return {
        "potencia_senal": potencia_senal,
        "potencia_ruido": potencia_ruido,
        "snr_lineal": snr_lineal,
        "snr_db": snr_db,
        "errores": errores,
        "ber": ber,
    }


def construir_tabla_muestras(
    bits_tx: str,
    bits_rx: str,
    simbolos: np.ndarray,
    ruido: np.ndarray,
    recibido: np.ndarray,
    max_muestras: int,
) -> pd.DataFrame:
    n = min(max_muestras, len(bits_tx))

    datos = {
        "Índice de muestra [n]": np.arange(1, n + 1),
        "Bit Tx": list(bits_tx[:n]),
        "Símbolo Tx s[n] [amplitud normalizada]": simbolos[:n],
        "Ruido n[n] [amplitud normalizada]": ruido[:n],
        "Valor recibido r[n] [amplitud normalizada]": recibido[:n],
        "Bit Rx": list(bits_rx[:n]),
        "Estado": [
            "Correcto" if a == b else "Error"
            for a, b in zip(bits_tx[:n], bits_rx[:n])
        ],
    }

    return pd.DataFrame(datos)


def analizar_resultados(metricas: dict, sigma: float) -> str:
    ber = metricas["ber"]
    snr_db = metricas["snr_db"]
    errores = metricas["errores"]

    interpretacion = []

    interpretacion.append(
        f"Se transmitió la secuencia a través de un canal AWGN con desviación estándar "
        f"σ = {sigma:.3f} [amplitud normalizada], por lo que la varianza del ruido es "
        f"σ² = {sigma**2:.6f} [amplitud normalizada²]."
    )

    if np.isfinite(snr_db):
        interpretacion.append(
            f"La relación señal-ruido estimada fue aproximadamente {snr_db:.3f} dB. "
            f"Cuando la potencia del ruido aumenta, la SNR disminuye."
        )
    else:
        interpretacion.append(
            "La SNR calculada resultó infinita porque la potencia de ruido fue cero "
            "o numéricamente despreciable."
        )

    interpretacion.append(
        f"En esta ejecución se observaron {errores} errores de bit y una BER de {ber:.6f} "
        "[adimensional]. La BER no tiene unidad física porque es una razón entre bits "
        "erróneos y bits evaluados."
    )

    if errores == 0:
        interpretacion.append(
            "No se observaron errores en la secuencia analizada. Esto no significa que el canal sea perfecto, "
            "sino que, bajo esta semilla y esta longitud de prueba, el ruido no desplazó ninguna muestra "
            "al lado incorrecto del umbral."
        )
    elif ber < 0.1:
        interpretacion.append(
            "La BER es baja, lo que indica que el ruido alteró algunas muestras, pero la mayoría "
            "de decisiones binarias siguieron siendo correctas."
        )
    else:
        interpretacion.append(
            "La BER es relativamente alta, lo que indica que el ruido está afectando de manera significativa "
            "la decisión en el receptor."
        )

    interpretacion.append(
        "La semilla controla la reproducibilidad del experimento: si no cambia la semilla, el generador "
        "aleatorio produce la misma realización de ruido y, por tanto, se repiten los mismos resultados."
    )

    interpretacion.append(
        "En esta guía se observa una aproximación discreta de un proceso de ruido en el tiempo. "
        "El eje horizontal representa el índice de muestra [n], no segundos, porque la app no define "
        "una frecuencia de muestreo física."
    )

    return "\n\n".join(interpretacion)


# ============================================================
# Gráficas
# ============================================================

def graficar_senal_transmitida(simbolos: np.ndarray, max_muestras: int) -> None:
    n = min(max_muestras, len(simbolos))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.stem(x, simbolos[:n], basefmt=" ")
    ax.set_title("Gráfica 1. Señal transmitida BPSK vs índice de muestra")
    ax.set_xlabel("Índice de muestra [n]")
    ax.set_ylabel("Amplitud normalizada")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Los símbolos BPSK se representan con amplitud normalizada: bit 0 → -1 y bit 1 → +1."
    )


def graficar_ruido_tiempo(ruido: np.ndarray, max_muestras: int) -> None:
    """
    Representa el ruido en función del índice de muestra.

    Se mantiene la forma visual original: línea con puntos.
    """
    n = min(max_muestras, len(ruido))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(x, ruido[:n], marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_title("Gráfica 2. Señal de ruido AWGN vs índice de muestra")
    ax.set_xlabel("Índice de muestra [n]")
    ax.set_ylabel("Ruido n[n] [amplitud normalizada]")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "El ruido se muestra como una secuencia de muestras generadas por el canal AWGN. "
        "El eje horizontal representa el índice de muestra [n], no segundos."
    )


def graficar_superposicion(
    simbolos: np.ndarray,
    ruido: np.ndarray,
    recibido: np.ndarray,
    max_muestras: int,
) -> None:
    """
    Superpone señal transmitida, ruido y señal recibida.

    Se mantiene la forma visual original:
    - señal transmitida con stem;
    - ruido con línea y puntos;
    - señal recibida con marcadores tipo x.
    """
    n = min(max_muestras, len(simbolos))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.stem(x, simbolos[:n], basefmt=" ", label="Señal transmitida s[n]")
    ax.plot(x, ruido[:n], marker="o", label="Ruido n[n]")
    ax.scatter(x, recibido[:n], marker="x", label="Señal recibida r[n] = s[n] + n[n]")
    ax.set_title("Gráfica 3. Superposición de señal transmitida, ruido y señal recibida")
    ax.set_xlabel("Índice de muestra [n]")
    ax.set_ylabel("Amplitud normalizada")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "La señal recibida se obtiene mediante el modelo r[n] = s[n] + n[n]. "
        "La amplitud se expresa de forma normalizada."
    )


def graficar_recepcion_y_umbral(
    recibido: np.ndarray,
    bits_rx: str,
    max_muestras: int,
) -> None:
    """
    Muestra la señal recibida y el umbral de decisión.

    Se mantiene la forma visual original: línea con puntos.
    """
    n = min(max_muestras, len(recibido))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(x, recibido[:n], marker="o", label="Valores recibidos r[n]")
    ax.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Umbral de decisión = 0",
    )
    ax.set_title("Gráfica 4. Señal recibida y umbral de decisión")
    ax.set_xlabel("Índice de muestra [n]")
    ax.set_ylabel("Valor recibido r[n] [amplitud normalizada]")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "La línea roja representa el umbral de decisión. Si r[n] ≥ 0 se decide 1; si r[n] < 0 se decide 0."
    )

    tabla_decision = pd.DataFrame(
        {
            "Índice de muestra [n]": x,
            "r[n] [amplitud normalizada]": recibido[:n],
            "Decisión": list(bits_rx[:n]),
            "Regla aplicada": [
                "r[n] ≥ 0 → 1" if valor >= 0 else "r[n] < 0 → 0"
                for valor in recibido[:n]
            ],
        }
    )

    st.dataframe(
        tabla_decision,
        use_container_width=True,
        hide_index=True,
    )


def graficar_ber_vs_sigma(df_comp: pd.DataFrame) -> None:
    """
    Gráfica del experimento rápido BER vs σ.

    Se mantiene como curva logarítmica con marcadores, igual que la estructura original,
    pero con etiquetas académicas.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    df_plot = df_comp.copy()
    df_plot["BER ajustada"] = df_plot["BER"].replace(0, 1e-6)

    ax.semilogy(
        df_plot["σ"],
        df_plot["BER ajustada"],
        marker="o",
    )

    ax.set_title("Efecto de σ sobre la BER")
    ax.set_xlabel("Desviación estándar del ruido σ [amplitud normalizada]")
    ax.set_ylabel("BER [adimensional, escala logarítmica]")
    ax.grid(True, which="both")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "La BER es adimensional porque se calcula como la razón entre bits erróneos y bits evaluados. "
        "Los valores cero se ajustan visualmente para poder representarlos en escala logarítmica."
    )


# ============================================================
# Interfaz principal
# ============================================================

def render_guia_01() -> None:
    st.title("Guía 1: Introducción al ruido en el tiempo y a la transmisión digital básica")

    st.markdown(
        """
Esta guía introduce los conceptos fundamentales necesarios para comprender cómo el ruido afecta
una señal digital en un sistema de telecomunicaciones. El propósito central es que el estudiante
observe la relación entre una secuencia de bits, su representación como símbolos, la perturbación
debida al ruido y el proceso de decisión en el receptor.

La guía parte del modelo elemental de comunicación digital, en el cual una señal transmitida se
ve alterada por ruido durante su propagación a través del canal. En esta implementación se utiliza
un canal AWGN (Additive White Gaussian Noise), ampliamente empleado en el análisis de sistemas
digitales por su utilidad teórica y práctica (Proakis & Salehi, 2008; Forouzan, 2013; Stallings, 2015).
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Simulación interactiva",
            "Análisis y dinámica",
            "Actividad guiada",
            "Conclusiones",
            "Referencias",
        ]
    )

    # ========================================================
    # OBJETIVOS
    # ========================================================
    with tabs[0]:
        st.header("Objetivos")

        st.markdown(
            """
**Objetivo general**

Comprender cómo el ruido afecta una señal digital en el tiempo, observando el modelo de transmisión
básico, la representación BPSK de los bits y el proceso de decisión en el receptor.

**Objetivos específicos**

1. Identificar la relación entre bits y símbolos en una transmisión digital básica.
2. Comprender el modelo de canal con ruido gaussiano aditivo.
3. Interpretar la desviación estándar σ y la varianza σ² como parámetros del ruido.
4. Observar cómo cambia la señal recibida al variar el nivel de ruido.
5. Analizar el efecto de la semilla en la reproducibilidad de la simulación.
6. Calcular e interpretar BER, potencia de ruido y SNR.
7. Relacionar la teoría del ruido con las gráficas obtenidas en la app.
8. Reconocer las unidades o naturaleza de las magnitudes graficadas: bits, dB, amplitud normalizada e índice de muestra.
"""
        )

    # ========================================================
    # TEORÍA
    # ========================================================
    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
### 1. Comunicación digital y representación de bits

En un sistema de comunicación digital, la información se representa mediante bits. Para poder
transmitirlos físicamente, esos bits se convierten en señales o símbolos aptos para el medio
de transmisión. Una forma muy simple de modelar este proceso consiste en asignar un símbolo
numérico a cada bit. En esta guía se emplea una representación BPSK normalizada:

$$
0 \\rightarrow -1
$$

$$
1 \\rightarrow +1
$$

Esta representación permite estudiar de forma clara el efecto del ruido sobre la señal transmitida,
ya que el receptor deberá decidir si el valor recibido corresponde al símbolo asociado al 0 o al 1
(Proakis & Salehi, 2008).

Como los símbolos se modelan con valores normalizados, las gráficas de amplitud no se expresan
en voltios. Por eso se indica **amplitud normalizada** en los ejes verticales.

### 2. Modelo del canal con ruido

Cuando una señal atraviesa un canal, puede verse afectada por perturbaciones aleatorias. En esta guía
se emplea el modelo AWGN, cuyo comportamiento puede expresarse como:

$$
r = s + n
$$

donde:

- $s$ es la señal o símbolo transmitido;
- $n$ es el ruido gaussiano aditivo;
- $r$ es la señal recibida.

Se denomina "aditivo" porque el ruido se suma a la señal transmitida; "gaussiano" porque se modela
con una distribución normal; y "blanco" porque su densidad espectral de potencia se considera uniforme
en el rango de frecuencias de interés (Proakis & Salehi, 2008; Stallings, 2015).

### 3. Media, desviación estándar y varianza

En una variable aleatoria gaussiana, la media y la dispersión son fundamentales. En canales AWGN
se suele asumir que la media del ruido es cero:

$$
\\mu_n = 0
$$

La desviación estándar del ruido se representa con $\\sigma$ y mide la dispersión de las muestras
respecto a la media. En esta app, $\\sigma$ se interpreta en unidades de **amplitud normalizada**,
porque la señal transmitida se representa con símbolos +1 y -1.

La varianza se expresa como:

$$
\\sigma^2
$$

Por tanto, en las métricas se interpreta como:

$$
\\sigma^2 \\; [\\text{amplitud normalizada}^2]
$$

Cuando $\\sigma$ aumenta, las muestras de ruido se dispersan más alrededor de cero. Como consecuencia,
la señal recibida tiene mayor probabilidad de desplazarse hacia el lado equivocado del umbral de decisión,
provocando errores de bit (Forouzan, 2013).

### 4. Decisión en el receptor

El receptor observa el valor recibido $r$ y aplica una regla de decisión sencilla:

- si $r \\geq 0$, decide bit 1;
- si $r < 0$, decide bit 0.

Esta regla se basa en que, sin ruido, el símbolo +1 representa el bit 1 y el símbolo -1 representa
el bit 0. Si el ruido es pequeño, la muestra recibida permanece cerca del símbolo correcto; si el ruido
es grande, puede cruzar el umbral y provocar una decisión errónea (Sklar, 2001; Proakis & Salehi, 2008).

### 5. BER: tasa de error de bit

La BER (Bit Error Rate) es una métrica fundamental en telecomunicaciones digitales y se define como:

$$
BER = \\frac{N_{errores}}{N_{bits}}
$$

donde:

- $N_{errores}$ es el número de bits recibidos incorrectamente;
- $N_{bits}$ es la cantidad total de bits analizados.

La BER permite evaluar qué tan afectada fue la transmisión por el ruido. No tiene unidades físicas,
porque es una razón entre cantidades de bits. Por eso se interpreta como **BER [adimensional]**.
Si el canal está poco afectado, la BER será baja; si el ruido es fuerte, la BER tenderá a aumentar
(Forouzan, 2013; Stallings, 2015).

### 6. Relación señal-ruido

La relación señal-ruido (SNR) compara la potencia de la señal con la potencia del ruido:

$$
SNR = \\frac{P_s}{P_n}
$$

y en decibeles:

$$
SNR_{dB} = 10 \\log_{10}(SNR)
$$

La SNR expresada en decibeles se reporta como **SNR [dB]**. Una SNR alta indica que la señal domina
sobre el ruido; una SNR baja indica que el ruido tiene un peso más importante y puede afectar la calidad
de la recepción (Proakis & Salehi, 2008).

### 7. Semilla y reproducibilidad

La semilla es el valor inicial del generador de números pseudoaleatorios. En esta guía se utiliza para
generar el ruido gaussiano. Si se usa la misma semilla con los mismos parámetros, el experimento produce
los mismos resultados. Esto es importante en entornos de laboratorio, ya que permite repetir pruebas y
comparar observaciones de forma controlada.

### 8. Interpretación temporal del ruido

Aunque el ruido es un proceso aleatorio, en simulación suele observarse como una secuencia de muestras
a lo largo del tiempo o del índice de observación. En esta app no se define una frecuencia de muestreo
física; por esa razón, el eje horizontal se expresa como:

$$
\\text{Índice de muestra } [n]
$$

Esto permite estudiar el comportamiento del ruido en el tiempo de forma discreta y su influencia directa
sobre la señal recibida.
"""
        )

        st.info(
            """
Cuadro de interpretación teórica:

- Si σ aumenta, la dispersión del ruido aumenta.
- Si la dispersión del ruido aumenta, la señal recibida se aleja más de la señal transmitida.
- Si la señal recibida cruza el umbral incorrecto, aparece un error de bit.
- Si aumentan los errores de bit, aumenta la BER.
- Si la potencia de ruido aumenta, disminuye la SNR.
- Las amplitudes de la app son normalizadas, no voltios.
- El eje horizontal de las gráficas es el índice de muestra [n], no segundos.
"""
        )

    # ========================================================
    # SIMULACIÓN INTERACTIVA
    # ========================================================
    with tabs[2]:
        st.header("Simulación interactiva")

        st.markdown(
            """
En esta sección el estudiante puede generar una secuencia de bits, observar su representación
como símbolos BPSK, analizar el comportamiento del ruido en el tiempo y estudiar cómo se forma
la señal recibida en el receptor.

La simulación permite visualizar cuatro elementos importantes del proceso de transmisión digital:

1. la señal transmitida como una secuencia de símbolos discretos;
2. el ruido gaussiano generado para cada muestra;
3. la superposición entre señal transmitida, ruido y señal recibida;
4. la decisión final del receptor usando un umbral.

El propósito de esta sección es relacionar la teoría del modelo:

$$
r = s + n
$$

con las gráficas obtenidas en la app. De esta forma, el estudiante puede observar cómo el ruido
modifica la señal transmitida y cómo esa modificación puede producir errores de bit.
"""
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            modo_entrada = st.radio(
                "Modo de entrada",
                ["Mensaje manual", "Bits aleatorios"],
                key="g1_modo_entrada",
            )

            if modo_entrada == "Mensaje manual":
                bits_tx = st.text_area(
                    "Ingrese la secuencia binaria",
                    value="1011001110010110",
                    key="g1_bits_manual",
                )
                bits_tx = limpiar_bits(bits_tx)
            else:
                cantidad_bits = st.selectbox(
                    "Cantidad de bits [bits]",
                    [8, 16, 32, 64, 128],
                    index=1,
                    key="g1_cantidad_bits",
                )
                semilla_datos = st.number_input(
                    "Semilla para los datos",
                    min_value=0,
                    max_value=999999,
                    value=101,
                    step=1,
                    key="g1_semilla_datos",
                )
                bits_tx = generar_bits_aleatorios(
                    int(cantidad_bits),
                    semilla=int(semilla_datos),
                )
                st.code(bits_tx, language="text")

            sigma = st.slider(
                "Desviación estándar del ruido σ [amplitud normalizada]",
                min_value=0.0,
                max_value=2.0,
                value=0.35,
                step=0.05,
                key="g1_sigma",
            )

            semilla_ruido = st.number_input(
                "Semilla del ruido",
                min_value=0,
                max_value=999999,
                value=1234,
                step=1,
                key="g1_semilla_ruido",
            )

            max_muestras = st.slider(
                "Cantidad de muestras a mostrar en gráficas [muestras]",
                min_value=4,
                max_value=40,
                value=16,
                step=1,
                key="g1_max_muestras",
            )

            ejecutar = st.button("Ejecutar simulación", key="g1_ejecutar")

        with col2:
            st.markdown(
                """
**Parámetros de observación**

- La semilla permite repetir exactamente el mismo experimento.
- La desviación estándar $\\sigma$ controla la dispersión del ruido.
- En esta app, $\\sigma$ se expresa en amplitud normalizada.
- La varianza del ruido es $\\sigma^2$ y se interpreta como amplitud normalizada².
- El número de muestras visibles afecta solamente la visualización, no la teoría del modelo.
"""
            )

            st.metric("Varianza del ruido σ² [amplitud normalizada²]", f"{sigma**2:.6f}")

        if not validar_bits(bits_tx):
            st.error("La secuencia debe contener únicamente 0 y 1.")
        elif ejecutar:
            bits_rx, simbolos, ruido, recibido = transmitir_awgn(
                bits_tx,
                sigma=float(sigma),
                semilla=int(semilla_ruido),
            )

            metricas = calcular_metricas(
                simbolos,
                ruido,
                bits_tx,
                bits_rx,
            )

            tabla = construir_tabla_muestras(
                bits_tx,
                bits_rx,
                simbolos,
                ruido,
                recibido,
                int(max_muestras),
            )

            st.subheader("Métricas principales")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Errores [bits]", f"{metricas['errores']}")
            m2.metric("BER [adimensional]", f"{metricas['ber']:.6f}")
            m3.metric(
                "SNR [dB]",
                "∞" if not np.isfinite(metricas["snr_db"]) else f"{metricas['snr_db']:.3f}",
            )
            m4.metric(
                "Potencia de ruido [amplitud normalizada²]",
                f"{metricas['potencia_ruido']:.6f}",
            )

            st.subheader("Gráficas")

            graficar_senal_transmitida(simbolos, int(max_muestras))
            graficar_ruido_tiempo(ruido, int(max_muestras))
            graficar_superposicion(simbolos, ruido, recibido, int(max_muestras))
            graficar_recepcion_y_umbral(recibido, bits_rx, int(max_muestras))

            st.subheader("Tabla de muestras")

            st.dataframe(
                tabla,
                use_container_width=True,
                hide_index=True,
            )

            snr_texto = "∞" if not np.isfinite(metricas["snr_db"]) else f"{metricas['snr_db']:.3f} dB"

            st.info(
                f"""
Cuadrito de interpretación:

- Se transmitieron {len(bits_tx)} bits.
- Con σ = {sigma:.3f} [amplitud normalizada], la varianza del ruido fue σ² = {sigma**2:.6f} [amplitud normalizada²].
- La BER observada fue {metricas['ber']:.6f} [adimensional].
- La SNR estimada fue {snr_texto}.
- Si repite la simulación con la misma semilla del ruido, obtendrá la misma forma de onda y los mismos resultados.
- Si aumenta σ, la onda de ruido tiende a alejar más la señal recibida de la señal transmitida.
"""
            )

            st.session_state["g1_bits_tx_resultado"] = bits_tx
            st.session_state["g1_bits_rx_resultado"] = bits_rx
            st.session_state["g1_simbolos_resultado"] = simbolos
            st.session_state["g1_ruido_resultado"] = ruido
            st.session_state["g1_recibido_resultado"] = recibido
            st.session_state["g1_metricas_resultado"] = metricas
            st.session_state["g1_sigma_resultado"] = sigma

    # ========================================================
    # ANÁLISIS Y DINÁMICA
    # ========================================================
    with tabs[3]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección vincula directamente los resultados obtenidos en la simulación con su interpretación
teórica. El propósito es que el estudiante no solo vea las gráficas, sino que entienda qué significan
y cómo se relacionan entre sí.
"""
        )

        if "g1_metricas_resultado" not in st.session_state:
            st.info("Ejecute primero la simulación interactiva para ver el análisis automático.")
        else:
            metricas = st.session_state["g1_metricas_resultado"]
            sigma = st.session_state["g1_sigma_resultado"]

            st.markdown(analizar_resultados(metricas, sigma))

            st.subheader("Experimento rápido: efecto de cambiar σ")

            valores_sigma = [0.10, 0.30, 0.50, 0.80, 1.00]
            bits_base = st.session_state["g1_bits_tx_resultado"]

            filas = []

            for i, sig in enumerate(valores_sigma):
                bits_rx_tmp, simbolos_tmp, ruido_tmp, _ = transmitir_awgn(
                    bits_base,
                    sigma=sig,
                    semilla=500 + i,
                )

                metricas_tmp = calcular_metricas(
                    simbolos_tmp,
                    ruido_tmp,
                    bits_base,
                    bits_rx_tmp,
                )

                filas.append(
                    {
                        "σ [amplitud normalizada]": sig,
                        "σ": sig,
                        "σ² [amplitud normalizada²]": sig**2,
                        "Errores [bits]": metricas_tmp["errores"],
                        "BER [adimensional]": metricas_tmp["ber"],
                        "BER": metricas_tmp["ber"],
                        "SNR [dB]": (
                            metricas_tmp["snr_db"]
                            if np.isfinite(metricas_tmp["snr_db"])
                            else np.nan
                        ),
                    }
                )

            df_comp = pd.DataFrame(filas)

            st.dataframe(
                df_comp[
                    [
                        "σ [amplitud normalizada]",
                        "σ² [amplitud normalizada²]",
                        "Errores [bits]",
                        "BER [adimensional]",
                        "SNR [dB]",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            graficar_ber_vs_sigma(df_comp)

            st.markdown(
                """
Interpretación del experimento:

- La tabla muestra cómo cambia la BER cuando cambia σ.
- Cuando σ aumenta, también aumenta la varianza del ruido.
- Al aumentar la varianza del ruido, la señal recibida presenta mayor dispersión.
- Esa mayor dispersión eleva la probabilidad de cruzar el umbral incorrecto.
- Por ello, en general, la BER tiende a aumentar cuando σ aumenta.
- La BER se muestra como una magnitud adimensional.
- La SNR se expresa en dB.
- σ se interpreta en amplitud normalizada porque los símbolos BPSK se modelan con -1 y +1.
"""
            )

    # ========================================================
    # ACTIVIDAD GUIADA
    # ========================================================
    with tabs[4]:
        st.header("Actividad guiada para el estudiante")

        st.markdown(
            """
### Actividad 1. Interpretación de símbolos

1. Ingrese una secuencia binaria de 8 a 16 bits.
2. Identifique qué símbolo BPSK corresponde a cada bit.
3. Observe la gráfica 1 y describa cómo se representan los bits 0 y 1.
4. Explique por qué el eje vertical se expresa como amplitud normalizada.

### Actividad 2. Observación del ruido

1. Fije una semilla del ruido.
2. Ejecute la simulación con σ = 0.10.
3. Luego repita con σ = 0.80.
4. Compare la gráfica 2 en ambos casos y describa cómo cambió la amplitud del ruido.
5. Explique por qué σ se mide en amplitud normalizada dentro de esta app.

### Actividad 3. Superposición de señal y ruido

1. Observe la gráfica 3.
2. Explique qué representa cada elemento:
   - señal transmitida;
   - ruido;
   - señal recibida.
3. Describa cómo el ruido modifica la posición de la señal recibida respecto a la transmitida.

### Actividad 4. Umbral de decisión

1. Observe la gráfica 4.
2. Explique la función de la línea roja horizontal.
3. Identifique si existen muestras que cruzan el umbral de forma incorrecta.
4. Compare esas observaciones con la tabla de decisión.

### Actividad 5. Papel de la semilla

1. Ejecute dos veces la simulación con la misma semilla del ruido.
2. Verifique si los resultados se repiten.
3. Cambie la semilla y observe qué cambia.

### Preguntas de reflexión

- ¿Qué significa que el ruido tenga media cero?
- ¿Por qué aumentar σ aumenta el riesgo de error?
- ¿Qué representa la varianza σ² en este contexto?
- ¿Por qué la BER no siempre es la misma al cambiar la semilla?
- ¿Por qué una SNR baja suele asociarse con una BER mayor?
- ¿Por qué el eje horizontal se llama índice de muestra [n] y no tiempo en segundos?
"""
        )

    # ========================================================
    # CONCLUSIONES
    # ========================================================
    with tabs[5]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe haber comprendido que la transmisión digital no consiste
únicamente en enviar bits, sino en enviar símbolos a través de un canal que puede alterarlos por efecto
del ruido. El modelo AWGN permite estudiar ese fenómeno de forma controlada y observar cómo la perturbación
aleatoria modifica la señal recibida.

También debe quedar claro que la desviación estándar $\\sigma$ y la varianza $\\sigma^2$ determinan la
intensidad estadística del ruido. En esta app, ambas magnitudes se interpretan con respecto a una amplitud
normalizada, porque la señal BPSK se modela con niveles -1 y +1. A mayor dispersión, mayor probabilidad
de error. Esta relación se refleja en la BER y en la SNR, dos métricas esenciales para el análisis de
sistemas de telecomunicaciones digitales.

Finalmente, la guía permite comprender que la semilla es una herramienta fundamental para repetir experimentos
y que la observación temporal del ruido ayuda a construir una intuición inicial sólida antes de avanzar hacia
etapas posteriores del proyecto, como detección y corrección de errores con Hamming y CRC.
"""
        )

    # ========================================================
    # REFERENCIAS
    # ========================================================
    with tabs[6]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2013). *Data communications and networking* (5th ed.). McGraw-Hill Education.

Proakis, J. G., & Salehi, M. (2008). *Digital communications* (5th ed.). McGraw-Hill.

Sklar, B. (2001). *Digital communications: Fundamentals and applications* (2nd ed.). Prentice Hall.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Tanenbaum, A. S., & Wetherall, D. J. (2011). *Computer networks* (5th ed.). Pearson.
"""
        )