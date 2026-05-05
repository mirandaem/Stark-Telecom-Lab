import numpy as np
import pandas as pd
import streamlit as st


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


def calcular_potencias_y_snr(simbolos: np.ndarray, ruido: np.ndarray) -> tuple[float, float, float, float]:
    """
    Calcula potencia promedio de señal, potencia promedio de ruido,
    SNR lineal y SNR en dB.
    """
    potencia_senal = float(np.mean(simbolos**2))
    potencia_ruido = float(np.mean(ruido**2))

    if potencia_ruido == 0:
        snr_lineal = np.inf
        snr_db = np.inf
    else:
        snr_lineal = potencia_senal / potencia_ruido
        snr_db = 10 * np.log10(snr_lineal)

    return potencia_senal, potencia_ruido, snr_lineal, snr_db


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
    potencia_senal, potencia_ruido, snr_lineal, snr_db = calcular_potencias_y_snr(simbolos, ruido)

    return {
        "Bits simulados": cantidad_bits,
        "σ": sigma,
        "σ²": sigma**2,
        "Potencia señal": potencia_senal,
        "Potencia ruido": potencia_ruido,
        "SNR": snr_lineal,
        "SNR dB": snr_db,
        "Errores": errores,
        "BER": ber,
    }


def simular_varios_escenarios(
    cantidad_bits: int,
    valores_sigma: list[float],
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


def render_guia_02() -> None:
    st.title("Guía 2: BER, SNR y análisis estadístico con muchos bits")

    st.markdown(
        """
Esta guía estudia el desempeño de una transmisión digital afectada por ruido utilizando
una cantidad grande de bits. El propósito es observar cómo el ruido modifica la tasa de
error de bit y cómo la razón señal-ruido permite comparar distintos escenarios del canal.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Simulación individual",
            "Comparación de escenarios",
            "Análisis",
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

Analizar estadísticamente el desempeño de una transmisión digital afectada por ruido
mediante el cálculo de BER y SNR sobre grandes cantidades de bits.

**Objetivos específicos**

1. Generar secuencias binarias aleatorias de gran tamaño.
2. Simular una transmisión digital con ruido gaussiano.
3. Calcular el BER experimental.
4. Relacionar la desviación estándar del ruido con su varianza.
5. Calcular potencia promedio de señal y potencia promedio de ruido.
6. Calcular la razón señal-ruido en escala lineal y en decibeles.
7. Comparar estadísticamente distintos escenarios de ruido.
8. Interpretar la relación entre ruido, SNR y BER.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En la guía anterior se observó cómo el ruido puede alterar una señal digital en el tiempo.
Sin embargo, analizar pocos bits no siempre permite obtener una medición representativa
del desempeño del sistema. Por esta razón, en esta guía se trabaja con secuencias más largas.

La tasa de error de bit se define como:

$$
BER = \\frac{N_e}{N_t}
$$

donde:

- $N_e$ es el número de bits erróneos;
- $N_t$ es el número total de bits transmitidos.

Cuando se simula una gran cantidad de bits, el BER experimental permite estimar con mayor
estabilidad el comportamiento del sistema frente al ruido.

La razón señal-ruido se define como:

$$
SNR = \\frac{P_s}{P_n}
$$

donde:

- $P_s$ es la potencia promedio de la señal;
- $P_n$ es la potencia promedio del ruido.

En decibeles, la SNR se expresa como:

$$
SNR_{dB} = 10 \\log_{10}(SNR)
$$

En esta guía se utiliza una representación BPSK básica:

$$
0 \\rightarrow -1
$$

$$
1 \\rightarrow +1
$$

Como los símbolos transmitidos tienen magnitud unitaria, la potencia promedio de la señal
es aproximadamente:

$$
P_s \\approx 1
$$

Si el ruido es gaussiano de media cero:

$$
n \\sim \\mathcal{N}(0, \\sigma^2)
$$

entonces su potencia promedio se aproxima mediante su varianza:

$$
P_n \\approx \\sigma^2
$$

Por tanto, al aumentar $\\sigma$, aumenta la potencia del ruido, disminuye la SNR y se espera
un aumento en el BER.
"""
        )

    with tabs[2]:
        st.header("Simulación individual")

        st.markdown(
            """
En esta sección se ejecuta una simulación para un único valor de desviación estándar del ruido.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits = st.selectbox(
                "Cantidad de bits",
                [100, 1000, 10000, 100000],
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

            usar_semilla = st.checkbox(
                "Usar semilla fija",
                value=True,
                key="g2_usar_semilla_individual",
            )

            semilla = None
            if usar_semilla:
                semilla = st.number_input(
                    "Semilla",
                    min_value=0,
                    max_value=999999,
                    value=123,
                    step=1,
                    key="g2_semilla_individual",
                )

            ejecutar = st.button("Ejecutar simulación individual", use_container_width=True)

        with col_info:
            st.info(
                """
Use esta sección para observar un caso específico. Cambie la cantidad de bits y la desviación
estándar del ruido para analizar cómo varían el BER y la SNR.
"""
            )

            st.metric("Varianza teórica σ²", f"{sigma**2:.4f}")

        if ejecutar:
            resultado = simular_escenario(cantidad_bits, sigma, semilla)

            st.subheader("Resultados de la simulación")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits simulados", f"{resultado['Bits simulados']:,}")
            c2.metric("Errores", resultado["Errores"])
            c3.metric("BER", f"{resultado['BER']:.6f}")
            c4.metric("SNR dB", "∞" if np.isinf(resultado["SNR dB"]) else f"{resultado['SNR dB']:.2f} dB")

            st.subheader("Parámetros y métricas")

            df_resultado = pd.DataFrame([resultado])
            st.dataframe(df_resultado, use_container_width=True, hide_index=True)

            st.markdown(
                f"""
**Interpretación del caso**

- La desviación estándar seleccionada fue $\\sigma = {sigma:.2f}$.
- La varianza teórica del ruido es $\\sigma^2 = {sigma**2:.4f}$.
- Se transmitieron {cantidad_bits:,} bits.
- Se detectaron {resultado["Errores"]} errores.
- El BER experimental fue {resultado["BER"]:.6f}.
"""
            )

            if resultado["BER"] == 0:
                st.success(
                    "No se observaron errores en esta simulación. Esto puede ocurrir cuando el ruido es bajo o la muestra no es suficientemente grande."
                )
            else:
                st.warning(
                    "Se observaron errores. Esto indica que el ruido provocó decisiones incorrectas en el receptor."
                )

            st.session_state["guia_02_resultado_individual"] = resultado

    with tabs[3]:
        st.header("Comparación de escenarios")

        st.markdown(
            """
En esta sección se ejecutan varias simulaciones con diferentes valores de desviación estándar.
El objetivo es comparar cómo cambian el BER y la SNR cuando aumenta el nivel de ruido.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits_comp = st.selectbox(
                "Cantidad de bits por escenario",
                [100, 1000, 10000, 100000],
                index=2,
                key="g2_bits_comparacion",
            )

            valores_sigma_texto = st.text_input(
                "Valores de σ separados por coma",
                value="0.10, 0.30, 0.50, 0.80, 1.00, 1.20",
                key="g2_valores_sigma",
            )

            usar_semilla_comp = st.checkbox(
                "Usar semilla fija para comparación",
                value=True,
                key="g2_usar_semilla_comp",
            )

            semilla_comp = None
            if usar_semilla_comp:
                semilla_comp = st.number_input(
                    "Semilla base",
                    min_value=0,
                    max_value=999999,
                    value=500,
                    step=1,
                    key="g2_semilla_comp",
                )

            ejecutar_comp = st.button("Ejecutar comparación", use_container_width=True)

        with col_info:
            st.info(
                """
Ejecute varios escenarios para observar la relación entre desviación estándar, varianza,
potencia de ruido, SNR y BER.
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

                if any(sigma < 0 for sigma in valores_sigma):
                    st.error("Los valores de σ no pueden ser negativos.")
                    return

            except ValueError:
                st.error("Ingrese valores numéricos válidos separados por coma.")
                return

            df_comparacion = simular_varios_escenarios(
                cantidad_bits=cantidad_bits_comp,
                valores_sigma=valores_sigma,
                semilla=semilla_comp,
            )

            st.subheader("Tabla comparativa")

            st.dataframe(df_comparacion, use_container_width=True, hide_index=True)

            st.subheader("Gráfica BER vs σ")

            df_ber_sigma = df_comparacion[["σ", "BER"]].set_index("σ")
            st.line_chart(df_ber_sigma)

            st.subheader("Gráfica BER vs SNR dB")

            df_snr = df_comparacion.replace([np.inf, -np.inf], np.nan).dropna(subset=["SNR dB"])
            if not df_snr.empty:
                df_ber_snr = df_snr[["SNR dB", "BER"]].sort_values("SNR dB").set_index("SNR dB")
                st.line_chart(df_ber_snr)
            else:
                st.info("No se puede graficar BER vs SNR dB cuando la SNR es infinita.")

            st.markdown(
                """
**Lectura esperada**

- Al aumentar σ, aumenta σ².
- Al aumentar σ², aumenta la potencia del ruido.
- Al aumentar la potencia del ruido, disminuye la SNR.
- Al disminuir la SNR, se espera que aumente el BER.
"""
            )

            st.session_state["guia_02_comparacion"] = df_comparacion

    with tabs[4]:
        st.header("Análisis de resultados")

        st.markdown(
            """
A partir de las simulaciones realizadas, analice:

1. ¿Qué ocurre con el BER al aumentar σ?
2. ¿Qué ocurre con la SNR al aumentar la potencia del ruido?
3. ¿Por qué el BER estimado con 100 bits puede ser menos estable que con 10,000 bits?
4. ¿Qué relación existe entre σ² y la potencia promedio del ruido?
5. ¿Por qué estas métricas serán útiles para evaluar posteriormente Hamming y CRC?
"""
        )

        if "guia_02_comparacion" in st.session_state:
            df = st.session_state["guia_02_comparacion"]

            st.subheader("Última comparación ejecutada")
            st.dataframe(df, use_container_width=True, hide_index=True)

            mejor_ber = df.loc[df["BER"].idxmin()]
            peor_ber = df.loc[df["BER"].idxmax()]

            st.markdown(
                f"""
**Observación automática**

- El menor BER observado fue {mejor_ber["BER"]:.6f} con σ = {mejor_ber["σ"]:.2f}.
- El mayor BER observado fue {peor_ber["BER"]:.6f} con σ = {peor_ber["σ"]:.2f}.
"""
            )
        else:
            st.info("Ejecute primero una comparación de escenarios.")

    with tabs[5]:
        st.header("Dinámica de aprendizaje")

        st.markdown(
            """
Realice las siguientes actividades dentro de la plataforma:

1. Ejecute una simulación individual con 100 bits y σ = 0.50.
2. Ejecute otra simulación con 10,000 bits y σ = 0.50.
3. Compare ambos valores de BER.
4. Ejecute una comparación con σ = 0.10, 0.30, 0.50, 0.80 y 1.00.
5. Observe la tendencia de BER respecto a σ.
6. Observe la tendencia de BER respecto a SNR dB.
"""
        )

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
            "Pregunta 2: ¿Qué representa la SNR?",
            [
                "La relación entre potencia de señal y potencia de ruido.",
                "La cantidad de bits transmitidos.",
                "El número total de errores.",
                "La varianza del mensaje.",
            ],
            index=None,
            key="g2_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "La relación entre potencia de señal y potencia de ruido.":
                st.success("Correcto. La SNR compara la potencia útil con la potencia del ruido.")
            else:
                st.error("Revise la definición de razón señal-ruido.")

        pregunta_3 = st.radio(
            "Pregunta 3: ¿Por qué conviene estimar BER con muchos bits?",
            [
                "Porque el cálculo se vuelve menos representativo.",
                "Porque permite obtener una estimación estadística más estable.",
                "Porque evita completamente el ruido.",
                "Porque elimina la necesidad de calcular SNR.",
            ],
            index=None,
            key="g2_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "Porque permite obtener una estimación estadística más estable.":
                st.success("Correcto. Con más bits, el BER experimental suele representar mejor el desempeño del sistema.")
            else:
                st.error("Revise la diferencia entre una muestra pequeña y una muestra grande.")

    with tabs[6]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- el BER experimental debe estimarse con una cantidad suficiente de bits;
- la desviación estándar σ controla la dispersión del ruido;
- la varianza σ² se relaciona con la potencia promedio del ruido;
- la SNR mide la relación entre potencia de señal y potencia de ruido;
- al aumentar el ruido, la SNR disminuye;
- al disminuir la SNR, el BER tiende a aumentar;
- estas métricas permitirán comparar posteriormente sistemas sin corrección, con Hamming y con Hamming + CRC.
"""
        )

    with tabs[7]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2012). *Data communications and networking* (5th ed.). McGraw-Hill.

Proakis, J. G., & Salehi, M. (2008). *Digital communications* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.
"""
        )