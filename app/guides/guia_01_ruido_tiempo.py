import numpy as np
import pandas as pd
import streamlit as st


def validar_bits(bits: str) -> bool:
    """
    Valida que la secuencia contenga únicamente bits.
    """
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def bits_a_simbolos(bits: str) -> np.ndarray:
    """
    Mapea bits a símbolos tipo BPSK:
    0 -> -1
    1 -> +1
    """
    bits_array = np.array([int(bit) for bit in bits])
    simbolos = np.where(bits_array == 1, 1.0, -1.0)
    return simbolos


def generar_ruido_gaussiano(
    cantidad: int,
    media: float,
    sigma: float,
    semilla: int | None = None,
) -> np.ndarray:
    """
    Genera muestras de ruido gaussiano.
    """
    rng = np.random.default_rng(semilla)
    return rng.normal(loc=media, scale=sigma, size=cantidad)


def decidir_bits(valores_recibidos: np.ndarray) -> np.ndarray:
    """
    Regla de decisión por umbral:
    r >= 0 -> bit 1
    r < 0  -> bit 0
    """
    return np.where(valores_recibidos >= 0, 1, 0)


def calcular_ber(bits_originales: np.ndarray, bits_decididos: np.ndarray) -> tuple[int, float]:
    """
    Calcula número de errores y BER.
    """
    errores = int(np.sum(bits_originales != bits_decididos))
    ber = errores / len(bits_originales)
    return errores, ber


def construir_tabla_resultados(
    bits: str,
    simbolos: np.ndarray,
    ruido: np.ndarray,
    recibida: np.ndarray,
    bits_decididos: np.ndarray,
) -> pd.DataFrame:
    """
    Construye una tabla con el proceso completo de transmisión y decisión.
    """
    bits_originales = np.array([int(bit) for bit in bits])

    return pd.DataFrame(
        {
            "Posición": np.arange(1, len(bits) + 1),
            "Bit transmitido": bits_originales,
            "Símbolo transmitido": simbolos,
            "Ruido n": ruido,
            "Valor recibido r = s + n": recibida,
            "Bit decidido": bits_decididos,
            "Estado": np.where(bits_originales == bits_decididos, "Correcto", "Error"),
        }
    )


def render_guia_01() -> None:
    st.title("Guía 1: Canal, ruido y señal en el tiempo")

    st.markdown(
        """
Esta guía introduce el comportamiento básico de una transmisión digital en presencia de ruido.
El objetivo es observar cómo una secuencia binaria puede representarse como una señal, cómo el
ruido altera esa señal y cómo el receptor toma decisiones que pueden producir errores.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Simulación",
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

Comprender cómo el ruido afecta una señal digital en el tiempo y cómo dicha alteración puede
provocar errores en la decisión del receptor.

**Objetivos específicos**

1. Identificar los bloques básicos de un sistema de comunicación digital.
2. Representar una secuencia binaria mediante símbolos digitales.
3. Interpretar el ruido como una variable aleatoria.
4. Modificar parámetros del ruido, como media, desviación estándar y varianza.
5. Observar cómo el ruido altera la señal recibida en el tiempo.
6. Aplicar una regla de decisión por umbral.
7. Relacionar los errores observados con la necesidad de técnicas de detección y corrección.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
Un sistema de comunicación digital tiene como propósito transportar información desde un
transmisor hacia un receptor a través de un canal. La información se representa mediante bits,
pero para su análisis puede asociarse con niveles de señal.

En un modelo básico, la señal recibida puede expresarse como:

$$
r = s + n
$$

donde:

- $r$ es la señal recibida;
- $s$ es la señal transmitida;
- $n$ representa el ruido introducido por el canal.

En esta guía, el ruido se modela como una variable aleatoria gaussiana:

$$
n \\sim \\mathcal{N}(\\mu, \\sigma^2)
$$

donde:

- $\\mu$ es la media del ruido;
- $\\sigma$ es la desviación estándar;
- $\\sigma^2$ es la varianza.

Para representar los bits como señal se utiliza un mapeo binario elemental:

$$
0 \\rightarrow -1
$$

$$
1 \\rightarrow +1
$$

Luego, el receptor decide el bit recibido utilizando un umbral en cero:

$$
r \\geq 0 \\Rightarrow \\hat{b} = 1
$$

$$
r < 0 \\Rightarrow \\hat{b} = 0
$$

El error ocurre cuando el ruido desplaza el valor recibido hacia el lado incorrecto del umbral.
"""
        )

    with tabs[2]:
        st.header("Simulación interactiva")

        st.markdown(
            """
Ingrese una secuencia binaria y modifique los parámetros del ruido. La aplicación mostrará
la señal transmitida, el ruido agregado, la señal recibida y la decisión final del receptor.
"""
        )

        col_entrada, col_info = st.columns([1, 1])

        with col_entrada:
            bits = st.text_input(
                "Secuencia binaria transmitida",
                value="1011001",
                help="Ingrese únicamente ceros y unos.",
            ).strip()

            media = st.number_input(
                "Media del ruido μ",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Para ruido AWGN ideal se suele usar media cero.",
            )

            sigma = st.slider(
                "Desviación estándar del ruido σ",
                min_value=0.0,
                max_value=2.0,
                value=0.30,
                step=0.05,
                help="A mayor σ, mayor dispersión del ruido.",
            )

            usar_semilla = st.checkbox(
                "Usar semilla fija",
                value=True,
                help="Permite repetir exactamente el mismo experimento.",
            )

            semilla = None
            if usar_semilla:
                semilla = st.number_input(
                    "Semilla",
                    min_value=0,
                    max_value=999999,
                    value=42,
                    step=1,
                )

            ejecutar = st.button("Ejecutar simulación", use_container_width=True)

        with col_info:
            st.info(
                """
La desviación estándar controla la intensidad del ruido. La varianza se calcula como:

σ² = σ · σ

Cuando σ aumenta, las muestras del ruido pueden alejarse más de la media y provocar
errores de decisión.
"""
            )

            st.metric("Varianza del ruido σ²", f"{sigma**2:.4f}")

        if not validar_bits(bits):
            st.error("La secuencia ingresada no es válida. Use únicamente caracteres 0 y 1.")
            return

        if ejecutar:
            bits_originales = np.array([int(bit) for bit in bits])
            simbolos = bits_a_simbolos(bits)
            ruido = generar_ruido_gaussiano(
                cantidad=len(bits),
                media=media,
                sigma=sigma,
                semilla=semilla,
            )
            recibida = simbolos + ruido
            bits_decididos = decidir_bits(recibida)
            errores, ber = calcular_ber(bits_originales, bits_decididos)

            tabla = construir_tabla_resultados(
                bits=bits,
                simbolos=simbolos,
                ruido=ruido,
                recibida=recibida,
                bits_decididos=bits_decididos,
            )

            st.subheader("Resultados principales")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bits transmitidos", len(bits))
            c2.metric("Errores", errores)
            c3.metric("BER", f"{ber:.4f}")
            c4.metric("Varianza σ²", f"{sigma**2:.4f}")

            st.subheader("Secuencia transmitida y decidida")

            secuencia_decidida = "".join(str(bit) for bit in bits_decididos)

            st.code(
                f"Bits transmitidos: {bits}\n"
                f"Bits decididos:    {secuencia_decidida}",
                language="text",
            )

            st.subheader("Tabla del proceso de transmisión")

            st.dataframe(
                tabla,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Señal transmitida y señal recibida en el tiempo")

            datos_grafica = pd.DataFrame(
                {
                    "Posición": np.arange(1, len(bits) + 1),
                    "Señal transmitida s": simbolos,
                    "Señal recibida r": recibida,
                    "Umbral de decisión": np.zeros(len(bits)),
                }
            ).set_index("Posición")

            st.line_chart(datos_grafica)

            st.subheader("Ruido generado")

            datos_ruido = pd.DataFrame(
                {
                    "Posición": np.arange(1, len(bits) + 1),
                    "Ruido n": ruido,
                }
            ).set_index("Posición")

            st.line_chart(datos_ruido)

            st.markdown(
                f"""
**Cálculo del BER**

$$
BER = \\frac{{N_e}}{{N_t}} = \\frac{{{errores}}}{{{len(bits)}}} = {ber:.4f}
$$
"""
            )

            if errores == 0:
                st.success(
                    "No se produjeron errores en esta simulación. La señal recibida no cruzó el umbral incorrectamente."
                )
            else:
                st.warning(
                    "Se produjeron errores. Al menos un valor recibido fue desplazado por el ruido hacia el lado incorrecto del umbral."
                )

            st.session_state["guia_01_resultados"] = {
                "bits": bits,
                "sigma": sigma,
                "varianza": sigma**2,
                "errores": errores,
                "ber": ber,
            }

    with tabs[3]:
        st.header("Análisis de resultados")

        st.markdown(
            """
Analice los resultados obtenidos a partir de la simulación:

1. Observe la gráfica de la señal transmitida y la señal recibida.
2. Identifique si algún valor recibido cruzó el umbral de decisión.
3. Compare los bits transmitidos con los bits decididos.
4. Interprete el valor de BER obtenido.
5. Repita la simulación aumentando la desviación estándar del ruido.
"""
        )

        if "guia_01_resultados" in st.session_state:
            resultados = st.session_state["guia_01_resultados"]

            st.write("Última simulación ejecutada:")

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Secuencia": resultados["bits"],
                            "σ": resultados["sigma"],
                            "σ²": resultados["varianza"],
                            "Errores": resultados["errores"],
                            "BER": resultados["ber"],
                        }
                    ]
                )
            )
        else:
            st.info("Ejecute primero una simulación en la pestaña Simulación.")

    with tabs[4]:
        st.header("Dinámica de aprendizaje")

        st.markdown(
            """
Realice las siguientes pruebas con la misma secuencia binaria:

- Prueba 1: σ = 0.10
- Prueba 2: σ = 0.50
- Prueba 3: σ = 1.00
- Prueba 4: σ = 1.50

Para cada caso, registre:

- varianza σ²;
- número de errores;
- BER;
- observación sobre la señal recibida.
"""
        )

        respuesta_1 = st.radio(
            "Pregunta 1: ¿Qué representa σ² en el modelo de ruido?",
            [
                "La media del ruido.",
                "La varianza del ruido.",
                "El número de bits transmitidos.",
                "El umbral de decisión.",
            ],
            index=None,
        )

        if respuesta_1:
            if respuesta_1 == "La varianza del ruido.":
                st.success("Correcto. σ² representa la varianza del ruido.")
            else:
                st.error("Revise la relación entre desviación estándar y varianza.")

        respuesta_2 = st.radio(
            "Pregunta 2: ¿Cuándo ocurre un error de decisión?",
            [
                "Cuando la señal transmitida es igual a cero.",
                "Cuando el ruido desplaza la señal recibida al lado incorrecto del umbral.",
                "Cuando la varianza es exactamente cero.",
                "Cuando el bit transmitido es 1.",
            ],
            index=None,
        )

        if respuesta_2:
            if respuesta_2 == "Cuando el ruido desplaza la señal recibida al lado incorrecto del umbral.":
                st.success("Correcto. El error ocurre cuando el receptor decide el bit equivocado.")
            else:
                st.error("Revise el papel del umbral de decisión en el receptor.")

        respuesta_3 = st.radio(
            "Pregunta 3: Si aumenta σ, ¿qué se espera que ocurra con la dispersión del ruido?",
            [
                "Disminuye.",
                "Aumenta.",
                "Permanece siempre igual.",
            ],
            index=None,
        )

        if respuesta_3:
            if respuesta_3 == "Aumenta.":
                st.success("Correcto. Una mayor desviación estándar implica mayor dispersión del ruido.")
            else:
                st.error("Revise el significado de la desviación estándar.")

    with tabs[5]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- una secuencia binaria puede representarse mediante niveles de señal;
- el canal puede alterar la señal mediante ruido;
- el ruido puede modelarse como una variable aleatoria;
- la desviación estándar controla la dispersión del ruido;
- la varianza corresponde a σ²;
- el receptor decide los bits mediante un umbral;
- los errores ocurren cuando el ruido provoca una decisión incorrecta;
- el BER permite cuantificar el desempeño de la transmisión.
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