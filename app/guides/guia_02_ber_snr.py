import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Utilidades generales
# ============================================================

def limpiar_bits(bits: str) -> str:
    """
    Limpia espacios, saltos de línea y tabulaciones de una secuencia binaria.
    """
    return bits.strip().replace(" ", "").replace("\n", "").replace("\t", "")


def validar_bits(bits: str) -> bool:
    """
    Valida que una secuencia contenga únicamente 0 y 1.
    """
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def generar_bits_aleatorios(cantidad: int, semilla: int | None = None) -> str:
    """
    Genera una secuencia binaria aleatoria reproducible mediante semilla.
    """
    rng = np.random.default_rng(semilla)
    bits = rng.integers(0, 2, size=cantidad)
    return "".join(str(bit) for bit in bits)


def bits_a_bpsk(bits: str) -> np.ndarray:
    """
    Convierte bits a símbolos BPSK normalizados.

    0 -> -1
    1 -> +1
    """
    bits_array = np.fromiter((int(bit) for bit in bits), dtype=int)
    return np.where(bits_array == 1, 1.0, -1.0)


def transmitir_awgn(
    bits: str,
    sigma: float,
    semilla: int | None = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transmite una secuencia binaria usando BPSK y canal AWGN.

    Modelo:

    r = s + n

    donde:
    s = símbolo transmitido;
    n = ruido gaussiano;
    r = valor recibido.

    Decisión:
    r >= 0 -> bit 1
    r < 0  -> bit 0
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
    """
    Cuenta errores de bit comparando dos secuencias.
    """
    longitud = min(len(bits_tx), len(bits_rx))

    if longitud == 0:
        return 0

    return sum(1 for a, b in zip(bits_tx[:longitud], bits_rx[:longitud]) if a != b)


def calcular_ber(bits_tx: str, bits_rx: str) -> float:
    """
    Calcula la tasa de error de bit.

    BER = errores / bits transmitidos
    """
    longitud = min(len(bits_tx), len(bits_rx))

    if longitud == 0:
        return 0.0

    return contar_errores(bits_tx, bits_rx) / longitud


def calcular_metricas(
    bits_tx: str,
    bits_rx: str,
    simbolos: np.ndarray,
    ruido: np.ndarray,
) -> Dict[str, float | int]:
    """
    Calcula métricas básicas de transmisión.
    """
    errores = contar_errores(bits_tx, bits_rx)
    ber = calcular_ber(bits_tx, bits_rx)

    potencia_senal = float(np.mean(simbolos**2)) if len(simbolos) else 0.0
    potencia_ruido = float(np.mean(ruido**2)) if len(ruido) else 0.0

    if potencia_ruido == 0:
        snr_lineal = math.inf
        snr_db = math.inf
    else:
        snr_lineal = potencia_senal / potencia_ruido
        snr_db = 10 * math.log10(snr_lineal)

    return {
        "bits": len(bits_tx),
        "errores": errores,
        "BER": ber,
        "potencia_senal": potencia_senal,
        "potencia_ruido": potencia_ruido,
        "SNR_lineal": snr_lineal,
        "SNR_dB": snr_db,
    }


# ============================================================
# Aproximación teórica Eb/N0 y BER
# ============================================================

def calcular_q(x: float) -> float:
    """
    Función Q usando erfc.

    Q(x) = 0.5 * erfc(x / sqrt(2))
    """
    return 0.5 * math.erfc(x / math.sqrt(2))


def aproximar_eb_n0_desde_sigma(sigma: float) -> float:
    """
    Aproximación didáctica de Eb/N0 para BPSK normalizado.

    En esta app:
    - los símbolos BPSK son -1 y +1;
    - se asume tiempo de bit normalizado Tb = 1;
    - por tanto, Eb ≈ 1;
    - el ruido real agregado tiene varianza sigma²;
    - para AWGN baseband real, sigma² ≈ N0/2.

    Entonces:

    N0 ≈ 2 sigma²

    Eb/N0 ≈ 1 / (2 sigma²)
    """
    if sigma <= 0:
        return math.inf

    return 1 / (2 * sigma**2)


def aproximar_snr_desde_sigma(sigma: float) -> float:
    """
    Aproximación de SNR usando símbolos normalizados.

    Como la potencia media de BPSK normalizado es aproximadamente 1:

    SNR ≈ 1 / sigma²
    """
    if sigma <= 0:
        return math.inf

    return 1 / (sigma**2)


def lineal_a_db(valor: float) -> float:
    """
    Convierte una magnitud lineal a decibeles.
    """
    if valor <= 0:
        return -math.inf

    if math.isinf(valor):
        return math.inf

    return 10 * math.log10(valor)


def ber_teorica_bpsk_awgn(sigma: float) -> float:
    """
    Aproximación teórica de BER para BPSK coherente en AWGN.

    Para BPSK:

    Pb = Q(sqrt(2 Eb/N0))

    Con la aproximación de esta app:

    Eb/N0 ≈ 1 / (2 sigma²)

    Por tanto:

    Pb ≈ Q(1 / sigma)

    Esta fórmula se usa como referencia teórica, no como sustituto de la simulación.
    """
    if sigma <= 0:
        return 0.0

    eb_n0 = aproximar_eb_n0_desde_sigma(sigma)
    return calcular_q(math.sqrt(2 * eb_n0))


# ============================================================
# Construcción de tablas
# ============================================================

def construir_tabla_muestras(
    bits_tx: str,
    bits_rx: str,
    simbolos: np.ndarray,
    ruido: np.ndarray,
    recibido: np.ndarray,
    max_muestras: int,
) -> pd.DataFrame:
    """
    Construye tabla de muestras discretas.
    """
    n = min(max_muestras, len(bits_tx))

    return pd.DataFrame(
        {
            "Índice": np.arange(1, n + 1),
            "Bit Tx": list(bits_tx[:n]),
            "Símbolo Tx s": simbolos[:n],
            "Ruido n": ruido[:n],
            "Recibido r = s + n": recibido[:n],
            "Bit Rx": list(bits_rx[:n]),
            "Estado": [
                "Correcto" if a == b else "Error"
                for a, b in zip(bits_tx[:n], bits_rx[:n])
            ],
        }
    )


def construir_tabla_eb_aproximacion() -> pd.DataFrame:
    """
    Tabla explicativa para Eb, N0, sigma² y aproximaciones usadas.
    """
    return pd.DataFrame(
        {
            "Concepto": [
                "Eb",
                "Tb",
                "BPSK normalizado",
                "σ",
                "σ²",
                "N0",
                "Eb/N0",
                "BER teórica BPSK",
            ],
            "Significado": [
                "Energía por bit",
                "Tiempo de bit",
                "Representación usada en la app",
                "Desviación estándar del ruido",
                "Varianza del ruido",
                "Densidad espectral de potencia de ruido",
                "Relación energía por bit a densidad de ruido",
                "Probabilidad aproximada de error para BPSK coherente",
            ],
            "Aproximación usada en esta guía": [
                "Eb ≈ 1",
                "Tb = 1",
                "0 → -1 y 1 → +1",
                "Parámetro elegido por el usuario",
                "σ²",
                "N0 ≈ 2σ²",
                "Eb/N0 ≈ 1/(2σ²)",
                "Pb ≈ Q(sqrt(2Eb/N0))",
            ],
            "Interpretación": [
                "La energía queda normalizada para simplificar el análisis",
                "Permite comparar potencia y energía sin cambiar escala temporal",
                "La potencia promedio de señal es aproximadamente 1",
                "Controla cuánto se dispersa el ruido",
                "Representa la potencia estadística del ruido en la simulación",
                "Se aproxima desde la varianza del ruido real agregado",
                "A mayor Eb/N0, menor probabilidad de error",
                "Sirve como referencia frente a la BER simulada",
            ],
        }
    )


def construir_tabla_metricas_teoricas() -> pd.DataFrame:
    """
    Tabla general de métricas teóricas usadas en la guía.
    """
    return pd.DataFrame(
        {
            "Métrica": [
                "BER",
                "SNR",
                "SNR dB",
                "Eb/N0",
                "Eb/N0 dB",
                "σ²",
            ],
            "Expresión": [
                "BER = errores / bits transmitidos",
                "SNR = Ps / Pn",
                "SNRdB = 10log10(SNR)",
                "Eb/N0 ≈ 1/(2σ²)",
                "10log10(Eb/N0)",
                "σ²",
            ],
            "Uso en la guía": [
                "Medir errores obtenidos por simulación",
                "Comparar potencia de señal con potencia de ruido",
                "Representar la SNR en decibeles",
                "Relacionar energía por bit con ruido",
                "Comparar escenarios de ruido en escala logarítmica",
                "Analizar BER en función de la varianza",
            ],
        }
    )


# ============================================================
# Simulación estadística
# ============================================================

def simular_escenario(
    bits_tx: str,
    sigma: float,
    semilla_ruido: int,
) -> Dict[str, float | int]:
    """
    Simula un escenario para un valor de sigma.
    """
    bits_rx, simbolos, ruido, recibido = transmitir_awgn(
        bits_tx,
        sigma=sigma,
        semilla=semilla_ruido,
    )

    metricas = calcular_metricas(bits_tx, bits_rx, simbolos, ruido)

    snr_aprox = aproximar_snr_desde_sigma(sigma)
    eb_n0_aprox = aproximar_eb_n0_desde_sigma(sigma)
    ber_teorica = ber_teorica_bpsk_awgn(sigma)

    return {
        "σ": sigma,
        "σ²": sigma**2,
        "Bits evaluados": len(bits_tx),
        "Errores": int(metricas["errores"]),
        "BER simulada": float(metricas["BER"]),
        "BER teórica BPSK": float(ber_teorica),
        "Potencia señal medida": float(metricas["potencia_senal"]),
        "Potencia ruido medida": float(metricas["potencia_ruido"]),
        "SNR medida": float(metricas["SNR_lineal"]) if np.isfinite(metricas["SNR_lineal"]) else math.inf,
        "SNR dB medida": float(metricas["SNR_dB"]) if np.isfinite(metricas["SNR_dB"]) else math.inf,
        "SNR aproximada": float(snr_aprox) if np.isfinite(snr_aprox) else math.inf,
        "SNR dB aproximada": float(lineal_a_db(snr_aprox)) if np.isfinite(lineal_a_db(snr_aprox)) else math.inf,
        "Eb/N0 aproximado": float(eb_n0_aprox) if np.isfinite(eb_n0_aprox) else math.inf,
        "Eb/N0 dB aproximado": float(lineal_a_db(eb_n0_aprox)) if np.isfinite(lineal_a_db(eb_n0_aprox)) else math.inf,
    }


def simular_barrido_sigmas(
    cantidad_bits: int,
    valores_sigma: List[float],
    semilla_datos: int,
    semilla_ruido_base: int,
) -> pd.DataFrame:
    """
    Simula BER para varios valores de sigma.

    Se usa la misma secuencia de datos para todos los escenarios, lo que permite
    comparar el efecto del ruido de forma más controlada.
    """
    bits_tx = generar_bits_aleatorios(cantidad_bits, semilla=semilla_datos)

    filas = []

    for i, sigma in enumerate(valores_sigma):
        fila = simular_escenario(
            bits_tx=bits_tx,
            sigma=sigma,
            semilla_ruido=semilla_ruido_base + i,
        )
        filas.append(fila)

    return pd.DataFrame(filas)


def generar_valores_sigma_desde_texto(texto: str) -> List[float]:
    """
    Convierte texto con valores separados por coma en lista de floats.
    """
    valores = []

    for parte in texto.split(","):
        parte = parte.strip()

        if parte == "":
            continue

        valores.append(float(parte))

    return valores


# ============================================================
# Gráficas corregidas
# ============================================================

def obtener_piso_visual(df: pd.DataFrame) -> float:
    """
    Define un piso visual para poder graficar BER = 0 en escala logarítmica.

    La tabla conserva la BER real. El piso solo afecta la gráfica.
    """
    if "Bits evaluados" not in df.columns or df.empty:
        return 1e-6

    max_bits = int(df["Bits evaluados"].max())

    if max_bits <= 0:
        return 1e-6

    return max(1 / max_bits, 1e-8)


def graficar_ber_vs_snr(df: pd.DataFrame) -> None:
    """
    Corrección solicitada:
    - BER vs SNR debe usar SNR en el eje X.
    - El eje X se ordena de menor a mayor SNR dB.
    - El eje Y usa escala logarítmica porque BER puede ser muy pequeña.
    - Los valores BER = 0 se grafican con un piso visual, sin alterar la tabla.
    """
    if df.empty:
        st.info("No hay datos para graficar.")
        return

    piso = obtener_piso_visual(df)

    df_plot = df.copy()
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
    df_plot = df_plot.dropna(subset=["SNR dB aproximada"])

    if df_plot.empty:
        st.info("No hay valores finitos de SNR para graficar BER vs SNR.")
        return

    df_plot = df_plot.sort_values("SNR dB aproximada")
    df_plot["BER simulada visual"] = df_plot["BER simulada"].replace(0, piso)
    df_plot["BER teórica visual"] = df_plot["BER teórica BPSK"].replace(0, piso)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.semilogy(
        df_plot["SNR dB aproximada"],
        df_plot["BER simulada visual"],
        marker="o",
        label="BER simulada",
    )

    ax.semilogy(
        df_plot["SNR dB aproximada"],
        df_plot["BER teórica visual"],
        marker="x",
        label="BER teórica BPSK",
    )

    ax.set_title("BER vs SNR dB")
    ax.set_xlabel("SNR aproximada (dB)")
    ax.set_ylabel("BER en escala logarítmica")
    ax.grid(True, which="both")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    if (df["BER simulada"] == 0).any():
        st.caption(
            f"Nota: las BER simuladas iguales a 0 se grafican con un piso visual de {piso:.2e} "
            "para poder mostrarlas en escala logarítmica. La tabla conserva el valor real."
        )


def graficar_ber_vs_varianza(df: pd.DataFrame) -> None:
    """
    Corrección solicitada:
    - BER vs Varianza debe usar σ² en el eje X.
    - El eje X se ordena de menor a mayor varianza.
    - El eje Y usa escala logarítmica.
    - No se debe confundir varianza con SNR.
    """
    if df.empty:
        st.info("No hay datos para graficar.")
        return

    piso = obtener_piso_visual(df)

    df_plot = df.copy()
    df_plot = df_plot.sort_values("σ²")
    df_plot["BER simulada visual"] = df_plot["BER simulada"].replace(0, piso)
    df_plot["BER teórica visual"] = df_plot["BER teórica BPSK"].replace(0, piso)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.semilogy(
        df_plot["σ²"],
        df_plot["BER simulada visual"],
        marker="o",
        label="BER simulada",
    )

    ax.semilogy(
        df_plot["σ²"],
        df_plot["BER teórica visual"],
        marker="x",
        label="BER teórica BPSK",
    )

    ax.set_title("BER vs varianza del ruido")
    ax.set_xlabel("Varianza del ruido σ²")
    ax.set_ylabel("BER en escala logarítmica")
    ax.grid(True, which="both")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    if (df["BER simulada"] == 0).any():
        st.caption(
            f"Nota: las BER simuladas iguales a 0 se grafican con un piso visual de {piso:.2e} "
            "para poder mostrarlas en escala logarítmica. La tabla conserva el valor real."
        )


def graficar_ber_vs_ebn0(df: pd.DataFrame) -> None:
    """
    Gráfica complementaria para desarrollar la teoría de Eb/N0.
    """
    if df.empty:
        st.info("No hay datos para graficar.")
        return

    piso = obtener_piso_visual(df)

    df_plot = df.copy()
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
    df_plot = df_plot.dropna(subset=["Eb/N0 dB aproximado"])

    if df_plot.empty:
        st.info("No hay valores finitos de Eb/N0 para graficar.")
        return

    df_plot = df_plot.sort_values("Eb/N0 dB aproximado")
    df_plot["BER simulada visual"] = df_plot["BER simulada"].replace(0, piso)
    df_plot["BER teórica visual"] = df_plot["BER teórica BPSK"].replace(0, piso)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.semilogy(
        df_plot["Eb/N0 dB aproximado"],
        df_plot["BER simulada visual"],
        marker="o",
        label="BER simulada",
    )

    ax.semilogy(
        df_plot["Eb/N0 dB aproximado"],
        df_plot["BER teórica visual"],
        marker="x",
        label="BER teórica BPSK",
    )

    ax.set_title("BER vs Eb/N0 aproximado")
    ax.set_xlabel("Eb/N0 aproximado (dB)")
    ax.set_ylabel("BER en escala logarítmica")
    ax.grid(True, which="both")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


def graficar_muestras_discretas(
    bits_tx: str,
    bits_rx: str,
    simbolos: np.ndarray,
    recibido: np.ndarray,
    max_muestras: int,
) -> None:
    """
    Gráfica de muestras discretas para evitar representación continua de bits.
    """
    n = min(max_muestras, len(bits_tx))
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.stem(
        x,
        simbolos[:n],
        basefmt=" ",
        label="Símbolo transmitido",
    )

    ax.scatter(
        x,
        recibido[:n],
        marker="x",
        label="Valor recibido",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        label="Umbral de decisión",
    )

    ax.set_title("Muestras discretas transmitidas y recibidas")
    ax.set_xlabel("Índice de muestra")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# Análisis automático
# ============================================================

def interpretar_barrido(df: pd.DataFrame) -> str:
    """
    Genera interpretación automática del barrido.
    """
    if df.empty:
        return "No hay datos suficientes para interpretar."

    df_sigma = df.sort_values("σ")
    menor_sigma = df_sigma.iloc[0]
    mayor_sigma = df_sigma.iloc[-1]

    texto = []

    texto.append(
        f"Con σ = {menor_sigma['σ']:.3f}, la varianza fue σ² = {menor_sigma['σ²']:.6f} "
        f"y la BER simulada fue {menor_sigma['BER simulada']:.6f}."
    )

    texto.append(
        f"Con σ = {mayor_sigma['σ']:.3f}, la varianza fue σ² = {mayor_sigma['σ²']:.6f} "
        f"y la BER simulada fue {mayor_sigma['BER simulada']:.6f}."
    )

    texto.append(
        "En general, al aumentar σ también aumenta σ². Como σ² representa la potencia estadística "
        "del ruido en esta simulación, se espera que la señal recibida se disperse más alrededor "
        "del símbolo transmitido."
    )

    texto.append(
        "Cuando la dispersión del ruido aumenta, existe mayor probabilidad de que una muestra cruce "
        "el umbral de decisión y produzca un error de bit. Por eso, la BER tiende a aumentar con la varianza."
    )

    texto.append(
        "En la gráfica BER vs SNR, la tendencia esperada es inversa: al aumentar la SNR, el BER tiende "
        "a disminuir. Esta es la razón por la que BER vs SNR y BER vs varianza no deben interpretarse "
        "como la misma gráfica."
    )

    texto.append(
        "La aproximación de Eb/N0 permite comparar la simulación con la expresión teórica de BPSK en AWGN. "
        "En esta app se asume energía por bit normalizada, por lo que Eb ≈ 1 y Eb/N0 ≈ 1/(2σ²)."
    )

    return "\n\n".join(texto)


# ============================================================
# Interfaz principal
# ============================================================

def render_guia_02() -> None:
    st.title("Guía 2: BER, SNR, varianza del ruido y aproximación Eb/N0")

    st.markdown(
        """
Esta guía estudia la relación entre el ruido y la tasa de error de bit en un sistema
digital básico. Se analiza cómo la desviación estándar del ruido, la varianza, la SNR
y la aproximación de energía por bit afectan la probabilidad de error.

Las correcciones implementadas en esta versión son:

- ajuste de la gráfica **BER vs SNR**;
- ajuste de la gráfica **BER vs varianza**;
- desarrollo teórico de la aproximación de **Eb/N0**;
- refuerzo de teoría con referencias dentro del texto;
- uso de escala logarítmica para BER;
- conservación de tablas con valores reales aunque las gráficas usen piso visual para BER = 0.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Simulación puntual",
            "BER vs SNR",
            "BER vs varianza",
            "Eb/N0",
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

Analizar la relación entre ruido, SNR, varianza, energía por bit y BER en una transmisión
digital con modulación BPSK sobre un canal AWGN.

**Objetivos específicos**

1. Calcular la BER a partir de bits transmitidos y recibidos.
2. Comprender la relación entre σ, σ² y potencia de ruido.
3. Interpretar la SNR como relación entre potencia de señal y potencia de ruido.
4. Corregir la lectura de la gráfica BER vs SNR.
5. Corregir la lectura de la gráfica BER vs varianza.
6. Desarrollar la aproximación de Eb/N0 para BPSK normalizado.
7. Comparar BER simulada con una referencia teórica BPSK sobre AWGN.
8. Usar escala logarítmica para observar valores pequeños de BER.
9. Diferenciar resultados puntuales de resultados estadísticos con muchos bits.
"""
        )

    # ========================================================
    # Teoría
    # ========================================================

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En comunicaciones digitales, una de las métricas más importantes es la tasa de error
de bit o BER. Esta métrica mide la proporción de bits recibidos incorrectamente respecto
al total de bits transmitidos (Proakis & Salehi, 2008; Forouzan, 2013).

La definición es:

$$
BER = \\frac{N_{errores}}{N_{bits}}
$$

donde:

- $N_{errores}$ es la cantidad de bits recibidos incorrectamente;
- $N_{bits}$ es la cantidad total de bits transmitidos.

La BER es especialmente importante porque permite evaluar cuantitativamente la calidad
de un sistema de transmisión digital.
"""
        )

        st.subheader("Canal AWGN y modelo de recepción")

        st.markdown(
            """
En esta guía se usa un canal AWGN, es decir, ruido blanco gaussiano aditivo. El modelo
base de recepción es:

$$
r = s + n
$$

donde:

- $s$ es el símbolo transmitido;
- $n$ es el ruido gaussiano;
- $r$ es el valor recibido.

El ruido se modela como una variable aleatoria gaussiana de media cero:

$$
n \\sim \\mathcal{N}(0, \\sigma^2)
$$

Esto significa que el ruido se dispersa alrededor de cero, y su nivel de dispersión
está determinado por la varianza $\\sigma^2$ (Proakis & Salehi, 2008; Sklar, 2001).
"""
        )

        st.subheader("Relación entre σ y σ²")

        st.markdown(
            """
La desviación estándar del ruido se representa como $\\sigma$. La varianza se obtiene
elevando ese valor al cuadrado:

$$
\\sigma^2
$$

En esta simulación, al aumentar $\\sigma$, también aumenta $\\sigma^2$. Como $\\sigma^2$
representa la potencia estadística del ruido agregado a las muestras, valores mayores
de varianza implican mayor dispersión de la señal recibida.

Por tanto:

- si $\\sigma$ aumenta, el ruido se dispersa más;
- si $\\sigma^2$ aumenta, aumenta la potencia del ruido;
- si aumenta la potencia del ruido, disminuye la SNR;
- si disminuye la SNR, aumenta la probabilidad de error.

Esta relación es central para interpretar las gráficas de BER (Stallings, 2015).
"""
        )

        st.subheader("SNR")

        st.markdown(
            """
La relación señal-ruido se define como:

$$
SNR = \\frac{P_s}{P_n}
$$

donde:

- $P_s$ es la potencia promedio de la señal;
- $P_n$ es la potencia promedio del ruido.

En decibeles:

$$
SNR_{dB} = 10 \\log_{10}(SNR)
$$

En esta app, como los símbolos BPSK se representan mediante $-1$ y $+1$, la potencia
promedio de la señal es aproximadamente:

$$
P_s \\approx 1
$$

Por ello, una aproximación didáctica es:

$$
SNR \\approx \\frac{1}{\\sigma^2}
$$

Esta aproximación ayuda a ordenar correctamente la gráfica BER vs SNR (Proakis & Salehi,
2008; Forouzan, 2013).
"""
        )

        st.subheader("Aproximación de Eb/N0")

        st.markdown(
            """
Una observación importante para esta guía es desarrollar la aproximación de $E_b/N_0$.

$E_b$ significa **energía por bit**. En un sistema real, la energía por bit depende de
la potencia de la señal y del tiempo de bit. Sin embargo, en esta app se usa un modelo
normalizado para facilitar el análisis:

- símbolos BPSK: $0 \\rightarrow -1$ y $1 \\rightarrow +1$;
- tiempo de bit normalizado: $T_b = 1$;
- potencia promedio de símbolo: $P_s \\approx 1$.

Con estas condiciones:

$$
E_b \\approx 1
$$

Para ruido AWGN real en banda base, la varianza del ruido puede relacionarse de forma
aproximada con $N_0$ mediante:

$$
\\sigma^2 \\approx \\frac{N_0}{2}
$$

Por tanto:

$$
N_0 \\approx 2\\sigma^2
$$

y:

$$
\\frac{E_b}{N_0} \\approx \\frac{1}{2\\sigma^2}
$$

Esta aproximación permite comparar la simulación con la expresión teórica de BER para
BPSK coherente sobre AWGN (Proakis & Salehi, 2008; Sklar, 2001).
"""
        )

        st.dataframe(
            construir_tabla_eb_aproximacion(),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("BER teórica para BPSK sobre AWGN")

        st.markdown(
            """
Para BPSK coherente en AWGN, una referencia teórica común para la probabilidad de error
de bit es:

$$
P_b = Q\\left(\\sqrt{2\\frac{E_b}{N_0}}\\right)
$$

Usando la aproximación anterior:

$$
\\frac{E_b}{N_0} \\approx \\frac{1}{2\\sigma^2}
$$

se obtiene:

$$
P_b \\approx Q\\left(\\frac{1}{\\sigma}\\right)
$$

Esta expresión no reemplaza la simulación, pero permite comparar si la tendencia obtenida
por la app es razonable. En simulaciones con pocos bits pueden existir diferencias por
aleatoriedad; con más bits, la BER simulada tiende a comportarse de forma más estable
(Proakis & Salehi, 2008).
"""
        )

        st.subheader("Diferencia entre BER vs SNR y BER vs varianza")

        st.markdown(
            """
La gráfica **BER vs SNR** debe interpretarse así:

- eje X: SNR en dB;
- eje Y: BER;
- tendencia esperada: al aumentar la SNR, la BER disminuye.

La gráfica **BER vs varianza** debe interpretarse así:

- eje X: varianza $\\sigma^2$;
- eje Y: BER;
- tendencia esperada: al aumentar la varianza, la BER aumenta.

Estas dos gráficas no deben confundirse. La SNR disminuye cuando aumenta la varianza
del ruido, por eso sus tendencias se ven en sentidos opuestos.
"""
        )

        st.dataframe(
            construir_tabla_metricas_teoricas(),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # Simulación puntual
    # ========================================================

    with tabs[2]:
        st.header("Simulación puntual")

        st.markdown(
            """
En esta sección se transmite una secuencia binaria con un único valor de $\\sigma$.
El objetivo es observar la BER, la SNR y las decisiones del receptor para un caso específico.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            modo = st.radio(
                "Modo de entrada",
                ["Bits aleatorios", "Mensaje manual"],
                key="g2_modo_puntual",
            )

            if modo == "Bits aleatorios":
                cantidad_bits = st.selectbox(
                    "Cantidad de bits",
                    [16, 32, 64, 128, 1000, 5000],
                    index=2,
                    key="g2_cantidad_puntual",
                )

                semilla_datos = st.number_input(
                    "Semilla de datos",
                    min_value=0,
                    max_value=999999,
                    value=202,
                    step=1,
                    key="g2_semilla_datos_puntual",
                )

                bits_tx = generar_bits_aleatorios(
                    int(cantidad_bits),
                    semilla=int(semilla_datos),
                )

                st.code(f"Primeros bits generados: {bits_tx[:120]}", language="text")
            else:
                bits_tx = st.text_area(
                    "Ingrese bits",
                    value="1011001110001111",
                    key="g2_bits_manual_puntual",
                )
                bits_tx = limpiar_bits(bits_tx)

            sigma = st.slider(
                "Desviación estándar del ruido σ",
                min_value=0.0,
                max_value=2.0,
                value=0.40,
                step=0.05,
                key="g2_sigma_puntual",
            )

            semilla_ruido = st.number_input(
                "Semilla del ruido",
                min_value=0,
                max_value=999999,
                value=303,
                step=1,
                key="g2_semilla_ruido_puntual",
            )

            max_muestras = st.slider(
                "Muestras a mostrar",
                min_value=8,
                max_value=80,
                value=32,
                step=1,
                key="g2_max_muestras_puntual",
            )

            ejecutar = st.button("Ejecutar simulación puntual", key="g2_ejecutar_puntual")

        with col_info:
            st.info(
                """
La simulación puntual permite observar un caso específico.

Para conclusiones estadísticas más confiables, use la pestaña BER vs SNR o BER vs varianza
con muchos bits.
"""
            )

            st.metric("Varianza σ²", f"{sigma**2:.6f}")

            ebn0 = aproximar_eb_n0_desde_sigma(sigma)
            snr_aprox = aproximar_snr_desde_sigma(sigma)

            st.metric(
                "SNR aproximada dB",
                "∞" if math.isinf(snr_aprox) else f"{lineal_a_db(snr_aprox):.3f}",
            )

            st.metric(
                "Eb/N0 aproximado dB",
                "∞" if math.isinf(ebn0) else f"{lineal_a_db(ebn0):.3f}",
            )

        if not validar_bits(bits_tx):
            st.error("La secuencia debe contener únicamente 0 y 1.")
        elif ejecutar:
            bits_rx, simbolos, ruido, recibido = transmitir_awgn(
                bits_tx,
                sigma=float(sigma),
                semilla=int(semilla_ruido),
            )

            metricas = calcular_metricas(bits_tx, bits_rx, simbolos, ruido)
            tabla = construir_tabla_muestras(
                bits_tx,
                bits_rx,
                simbolos,
                ruido,
                recibido,
                int(max_muestras),
            )

            st.subheader("Métricas del caso")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits evaluados", int(metricas["bits"]))
            c2.metric("Errores", int(metricas["errores"]))
            c3.metric("BER", f"{metricas['BER']:.6f}")
            c4.metric(
                "SNR dB medida",
                "∞" if math.isinf(metricas["SNR_dB"]) else f"{metricas['SNR_dB']:.3f}",
            )

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Potencia señal", f"{metricas['potencia_senal']:.6f}")
            c6.metric("Potencia ruido", f"{metricas['potencia_ruido']:.6f}")
            c7.metric("σ", f"{sigma:.3f}")
            c8.metric("σ²", f"{sigma**2:.6f}")

            st.subheader("Muestras discretas")

            graficar_muestras_discretas(
                bits_tx,
                bits_rx,
                simbolos,
                recibido,
                int(max_muestras),
            )

            st.subheader("Tabla de primeras muestras")

            st.dataframe(tabla, use_container_width=True, hide_index=True)

            st.markdown(
                f"""
**Interpretación**

Con $\\sigma = {sigma:.3f}$, la varianza del ruido es:

$$
\\sigma^2 = {sigma**2:.6f}
$$

La BER obtenida en esta simulación fue:

$$
BER = {metricas["BER"]:.6f}
$$

Si la BER fue cero, no significa que el sistema sea perfecto. Significa que, para esta
cantidad de bits, esta semilla y este valor de $\\sigma$, no se observaron errores.
Al aumentar la cantidad de bits, la estimación estadística de BER se vuelve más estable.
"""
            )

            st.session_state["g2_puntual"] = {
                "bits_tx": bits_tx,
                "bits_rx": bits_rx,
                "metricas": metricas,
                "sigma": sigma,
                "tabla": tabla,
            }

    # ========================================================
    # BER vs SNR
    # ========================================================

    with tabs[3]:
        st.header("BER vs SNR")

        st.markdown(
            """
Esta sección corrige la gráfica **BER vs SNR**. La SNR se coloca en el eje X en dB y la
BER en el eje Y con escala logarítmica.

La tendencia esperada es:

$$
\\text{si SNR aumenta, BER disminuye}
$$

Esto ocurre porque una SNR mayor indica que la señal tiene más potencia relativa que el
ruido.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits_snr = st.selectbox(
                "Cantidad de bits para barrido",
                [1000, 5000, 10000, 50000, 100000],
                index=2,
                key="g2_bits_snr",
            )

            sigmas_snr_texto = st.text_input(
                "Valores de σ separados por coma",
                value="0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.10",
                key="g2_sigmas_snr",
            )

            semilla_datos_snr = st.number_input(
                "Semilla de datos",
                min_value=0,
                max_value=999999,
                value=1000,
                step=1,
                key="g2_semilla_datos_snr",
            )

            semilla_ruido_snr = st.number_input(
                "Semilla base de ruido",
                min_value=0,
                max_value=999999,
                value=2000,
                step=1,
                key="g2_semilla_ruido_snr",
            )

            ejecutar_snr = st.button("Generar BER vs SNR", key="g2_ejecutar_snr")

        with col_info:
            st.info(
                """
La gráfica usa SNR aproximada:

SNR ≈ 1/σ²

Esto se debe a que la potencia promedio de los símbolos BPSK normalizados es aproximadamente 1.
"""
            )

        if ejecutar_snr:
            try:
                valores_sigma = generar_valores_sigma_desde_texto(sigmas_snr_texto)
            except ValueError:
                st.error("Ingrese valores numéricos válidos separados por coma.")
                return

            if len(valores_sigma) == 0:
                st.error("Debe ingresar al menos un valor de σ.")
                return

            if any(sigma < 0 for sigma in valores_sigma):
                st.error("Los valores de σ no pueden ser negativos.")
                return

            df_snr = simular_barrido_sigmas(
                cantidad_bits=int(cantidad_bits_snr),
                valores_sigma=valores_sigma,
                semilla_datos=int(semilla_datos_snr),
                semilla_ruido_base=int(semilla_ruido_snr),
            )

            st.subheader("Tabla del barrido")

            st.dataframe(
                df_snr[
                    [
                        "σ",
                        "σ²",
                        "Bits evaluados",
                        "Errores",
                        "BER simulada",
                        "BER teórica BPSK",
                        "SNR dB aproximada",
                        "Eb/N0 dB aproximado",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Gráfica corregida: BER vs SNR")

            graficar_ber_vs_snr(df_snr)

            st.session_state["g2_df_snr"] = df_snr

    # ========================================================
    # BER vs varianza
    # ========================================================

    with tabs[4]:
        st.header("BER vs varianza del ruido")

        st.markdown(
            """
Esta sección corrige la gráfica **BER vs varianza**. La varianza $\\sigma^2$ se coloca
en el eje X y la BER se coloca en el eje Y con escala logarítmica.

La tendencia esperada es:

$$
\\text{si } \\sigma^2 \\text{ aumenta, BER aumenta}
$$

Esto ocurre porque una varianza mayor implica ruido más disperso y mayor probabilidad
de cruzar el umbral de decisión.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits_var = st.selectbox(
                "Cantidad de bits para barrido",
                [1000, 5000, 10000, 50000, 100000],
                index=2,
                key="g2_bits_var",
            )

            sigmas_var_texto = st.text_input(
                "Valores de σ separados por coma",
                value="0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.10",
                key="g2_sigmas_var",
            )

            semilla_datos_var = st.number_input(
                "Semilla de datos",
                min_value=0,
                max_value=999999,
                value=3000,
                step=1,
                key="g2_semilla_datos_var",
            )

            semilla_ruido_var = st.number_input(
                "Semilla base de ruido",
                min_value=0,
                max_value=999999,
                value=4000,
                step=1,
                key="g2_semilla_ruido_var",
            )

            ejecutar_var = st.button("Generar BER vs varianza", key="g2_ejecutar_var")

        with col_info:
            st.info(
                """
Esta gráfica no usa SNR en el eje X.

Usa directamente σ², por eso la tendencia esperada va en sentido opuesto a BER vs SNR.
"""
            )

        if ejecutar_var:
            try:
                valores_sigma = generar_valores_sigma_desde_texto(sigmas_var_texto)
            except ValueError:
                st.error("Ingrese valores numéricos válidos separados por coma.")
                return

            if len(valores_sigma) == 0:
                st.error("Debe ingresar al menos un valor de σ.")
                return

            if any(sigma < 0 for sigma in valores_sigma):
                st.error("Los valores de σ no pueden ser negativos.")
                return

            df_var = simular_barrido_sigmas(
                cantidad_bits=int(cantidad_bits_var),
                valores_sigma=valores_sigma,
                semilla_datos=int(semilla_datos_var),
                semilla_ruido_base=int(semilla_ruido_var),
            )

            st.subheader("Tabla del barrido")

            st.dataframe(
                df_var[
                    [
                        "σ",
                        "σ²",
                        "Bits evaluados",
                        "Errores",
                        "BER simulada",
                        "BER teórica BPSK",
                        "SNR dB aproximada",
                        "Eb/N0 dB aproximado",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Gráfica corregida: BER vs varianza")

            graficar_ber_vs_varianza(df_var)

            st.session_state["g2_df_var"] = df_var

    # ========================================================
    # Eb/N0
    # ========================================================

    with tabs[5]:
        st.header("Aproximación Eb/N0")

        st.markdown(
            """
Esta sección desarrolla la aproximación de $E_b/N_0$ solicitada en las correcciones.

En un análisis formal de comunicaciones digitales, $E_b/N_0$ es una métrica muy usada
porque relaciona la energía por bit con la densidad espectral de potencia del ruido.
Esto permite comparar sistemas de forma más general que usando solamente amplitudes o
varianza.

En esta app se usa una aproximación didáctica:

$$
E_b \\approx 1
$$

porque los símbolos BPSK están normalizados a $-1$ y $+1$, y se toma $T_b = 1$.

Además:

$$
N_0 \\approx 2\\sigma^2
$$

Por tanto:

$$
\\frac{E_b}{N_0} \\approx \\frac{1}{2\\sigma^2}
$$

Con esta relación se puede construir una curva BER vs $E_b/N_0$ y compararla con la
referencia teórica BPSK:

$$
P_b = Q\\left(\\sqrt{2\\frac{E_b}{N_0}}\\right)
$$
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits_eb = st.selectbox(
                "Cantidad de bits",
                [1000, 5000, 10000, 50000, 100000],
                index=2,
                key="g2_bits_eb",
            )

            sigmas_eb_texto = st.text_input(
                "Valores de σ separados por coma",
                value="0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.10",
                key="g2_sigmas_eb",
            )

            semilla_datos_eb = st.number_input(
                "Semilla de datos",
                min_value=0,
                max_value=999999,
                value=5000,
                step=1,
                key="g2_semilla_datos_eb",
            )

            semilla_ruido_eb = st.number_input(
                "Semilla base de ruido",
                min_value=0,
                max_value=999999,
                value=6000,
                step=1,
                key="g2_semilla_ruido_eb",
            )

            ejecutar_eb = st.button("Generar BER vs Eb/N0", key="g2_ejecutar_eb")

        with col_info:
            st.info(
                """
Esta gráfica es una referencia teórica y experimental.

La BER simulada depende de la cantidad de bits y de la semilla.
La BER teórica representa la tendencia ideal para BPSK coherente en AWGN.
"""
            )

        if ejecutar_eb:
            try:
                valores_sigma = generar_valores_sigma_desde_texto(sigmas_eb_texto)
            except ValueError:
                st.error("Ingrese valores numéricos válidos separados por coma.")
                return

            if len(valores_sigma) == 0:
                st.error("Debe ingresar al menos un valor de σ.")
                return

            if any(sigma < 0 for sigma in valores_sigma):
                st.error("Los valores de σ no pueden ser negativos.")
                return

            df_eb = simular_barrido_sigmas(
                cantidad_bits=int(cantidad_bits_eb),
                valores_sigma=valores_sigma,
                semilla_datos=int(semilla_datos_eb),
                semilla_ruido_base=int(semilla_ruido_eb),
            )

            st.subheader("Tabla Eb/N0")

            st.dataframe(
                df_eb[
                    [
                        "σ",
                        "σ²",
                        "Eb/N0 aproximado",
                        "Eb/N0 dB aproximado",
                        "BER simulada",
                        "BER teórica BPSK",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("BER vs Eb/N0 aproximado")

            graficar_ber_vs_ebn0(df_eb)

            st.session_state["g2_df_eb"] = df_eb

    # ========================================================
    # Análisis y dinámica
    # ========================================================

    with tabs[6]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección integra los resultados de las simulaciones con la interpretación teórica.
El objetivo es que el estudiante pueda leer las curvas correctamente.
"""
        )

        if "g2_df_snr" in st.session_state:
            st.subheader("Análisis de BER vs SNR")

            df_snr = st.session_state["g2_df_snr"]

            st.dataframe(df_snr, use_container_width=True, hide_index=True)

            st.markdown(interpretar_barrido(df_snr))
        else:
            st.info("Ejecute primero la pestaña BER vs SNR.")

        if "g2_df_var" in st.session_state:
            st.subheader("Análisis de BER vs varianza")

            df_var = st.session_state["g2_df_var"]

            menor_var = df_var.sort_values("σ²").iloc[0]
            mayor_var = df_var.sort_values("σ²").iloc[-1]

            st.markdown(
                f"""
Con la menor varianza evaluada, $\\sigma^2 = {menor_var["σ²"]:.6f}$, la BER simulada fue
**{menor_var["BER simulada"]:.6f}**.

Con la mayor varianza evaluada, $\\sigma^2 = {mayor_var["σ²"]:.6f}$, la BER simulada fue
**{mayor_var["BER simulada"]:.6f}**.

La interpretación esperada es que la BER tienda a crecer cuando aumenta $\\sigma^2$,
porque el ruido tiene mayor potencia estadística.
"""
            )
        else:
            st.info("Ejecute primero la pestaña BER vs varianza.")

        st.subheader("Actividades guiadas")

        st.markdown(
            """
Realice las siguientes actividades:

1. Ejecute una simulación puntual con pocos bits y observe si la BER cambia al modificar la semilla.
2. Ejecute BER vs SNR con 10,000 bits.
3. Observe si la BER disminuye cuando aumenta la SNR.
4. Ejecute BER vs varianza con los mismos valores de σ.
5. Observe si la BER aumenta cuando aumenta σ².
6. Explique por qué BER vs SNR y BER vs varianza tienen tendencias opuestas.
7. Ejecute BER vs Eb/N0.
8. Compare la BER simulada con la BER teórica.
9. Explique por qué con pocos bits puede haber diferencias grandes entre teoría y simulación.
10. Explique por qué con muchos bits la estimación de BER suele ser más estable.
"""
        )

        st.subheader("Preguntas de análisis")

        pregunta_1 = st.radio(
            "Pregunta 1: ¿Qué ocurre normalmente con la BER cuando aumenta la SNR?",
            [
                "La BER aumenta.",
                "La BER disminuye.",
                "La BER no tiene relación con la SNR.",
                "La BER siempre es cero.",
            ],
            index=None,
            key="g2_pregunta_1",
        )

        if pregunta_1:
            if pregunta_1 == "La BER disminuye.":
                st.success("Correcto. Una SNR mayor indica que la señal domina más sobre el ruido.")
            else:
                st.error("Revise la relación entre SNR y BER.")

        pregunta_2 = st.radio(
            "Pregunta 2: ¿Qué ocurre normalmente con la BER cuando aumenta la varianza σ²?",
            [
                "La BER tiende a aumentar.",
                "La BER tiende a disminuir.",
                "La BER se vuelve negativa.",
                "La varianza no afecta al ruido.",
            ],
            index=None,
            key="g2_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "La BER tiende a aumentar.":
                st.success("Correcto. Mayor varianza implica ruido más disperso y más probabilidad de error.")
            else:
                st.error("Revise la relación entre varianza del ruido y BER.")

        pregunta_3 = st.radio(
            "Pregunta 3: En esta app, ¿cuál es la aproximación usada para Eb/N0?",
            [
                "Eb/N0 ≈ 1/(2σ²)",
                "Eb/N0 ≈ 2σ²",
                "Eb/N0 ≈ BER",
                "Eb/N0 ≈ σ",
            ],
            index=None,
            key="g2_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "Eb/N0 ≈ 1/(2σ²)":
                st.success("Correcto. Se asume Eb ≈ 1 y N0 ≈ 2σ².")
            else:
                st.error("Revise la sección de aproximación Eb/N0.")

        pregunta_4 = st.radio(
            "Pregunta 4: ¿Por qué se usa escala logarítmica para BER?",
            [
                "Porque la BER puede tomar valores muy pequeños.",
                "Porque la BER siempre es mayor que 1.",
                "Porque la escala logarítmica elimina errores.",
                "Porque la varianza siempre es cero.",
            ],
            index=None,
            key="g2_pregunta_4",
        )

        if pregunta_4:
            if pregunta_4 == "Porque la BER puede tomar valores muy pequeños.":
                st.success("Correcto. La escala logarítmica permite comparar mejor valores pequeños.")
            else:
                st.error("Revise la interpretación de curvas BER.")

    # ========================================================
    # Conclusiones
    # ========================================================

    with tabs[7]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- la BER mide la proporción de bits recibidos incorrectamente;
- el canal AWGN se modela como $r = s + n$;
- el ruido se modela como una variable gaussiana de media cero y varianza $\\sigma^2$;
- al aumentar $\\sigma$, aumenta $\\sigma^2$;
- al aumentar $\\sigma^2$, aumenta la potencia estadística del ruido;
- al aumentar el ruido, disminuye la SNR;
- al aumentar la SNR, la BER tiende a disminuir;
- al aumentar la varianza, la BER tiende a aumentar;
- BER vs SNR y BER vs varianza no son la misma gráfica;
- la aproximación $E_b/N_0 \\approx 1/(2\\sigma^2)$ permite comparar la simulación con teoría BPSK;
- la BER teórica para BPSK en AWGN se puede expresar como $P_b = Q(\\sqrt{2E_b/N_0})$;
- la escala logarítmica es adecuada para visualizar BER;
- las simulaciones con más bits producen estimaciones más estables.

La teoría aplicada en esta guía se fundamenta en comunicaciones digitales, ruido AWGN,
BER, SNR y análisis de desempeño de sistemas BPSK (Proakis & Salehi, 2008; Sklar, 2001;
Forouzan, 2013; Stallings, 2015).
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

Proakis, J. G., & Salehi, M. (2008). *Digital communications* (5th ed.). McGraw-Hill.

Sklar, B. (2001). *Digital communications: Fundamentals and applications* (2nd ed.). Prentice Hall.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.
"""
        )