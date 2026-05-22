import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def bits_a_pulsos(bits: str) -> np.ndarray:
    return np.array([int(bit) for bit in bits])


def bits_a_simbolos_bpsk(bits: str) -> np.ndarray:
    bits_array = np.array([int(bit) for bit in bits])
    return np.where(bits_array == 1, 1.0, -1.0)


def generar_ruido_gaussiano(
    cantidad: int,
    media: float,
    sigma: float,
    semilla: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return rng.normal(loc=media, scale=sigma, size=cantidad)


def decidir_bits(valores_recibidos: np.ndarray) -> np.ndarray:
    return np.where(valores_recibidos >= 0, 1, 0)


def calcular_ber(bits_originales: np.ndarray, bits_decididos: np.ndarray) -> tuple[int, float]:
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


def graficar_pulsos_bits(bits: str):
    posiciones = np.arange(1, len(bits) + 1)
    pulsos = bits_a_pulsos(bits)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.step(posiciones, pulsos, where="mid")
    ax.scatter(posiciones, pulsos)

    ax.set_title("Pulsos discretos de bits transmitidos")
    ax.set_xlabel("Índice de bit")
    ax.set_ylabel("Valor del bit")
    ax.set_yticks([0, 1])
    ax.grid(True)

    st.pyplot(fig)


def graficar_simbolos_bpsk(bits: str, simbolos: np.ndarray):
    posiciones = np.arange(1, len(bits) + 1)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.stem(posiciones, simbolos)

    ax.set_title("Símbolos BPSK por bit")
    ax.set_xlabel("Índice de bit")
    ax.set_ylabel("Símbolo transmitido")
    ax.set_yticks([-1, 0, 1])
    ax.grid(True)

    st.pyplot(fig)


def graficar_ruido_discreto(ruido: np.ndarray):
    posiciones = np.arange(1, len(ruido) + 1)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.stem(posiciones, ruido)
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title("Ruido gaussiano por muestra")
    ax.set_xlabel("Índice de muestra")
    ax.set_ylabel("Valor de ruido")
    ax.grid(True)

    st.pyplot(fig)


def graficar_senal_recibida(simbolos: np.ndarray, recibida: np.ndarray):
    posiciones = np.arange(1, len(simbolos) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.stem(posiciones, simbolos, label="Símbolo transmitido")
    ax.scatter(posiciones, recibida, marker="x", label="Valor recibido")

    ax.axhline(0, linestyle="--", linewidth=1, label="Umbral de decisión")

    ax.set_title("Símbolos transmitidos y valores recibidos")
    ax.set_xlabel("Índice de bit")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)


def render_guia_01() -> None:
    st.title("Guía 1: Canal, ruido y señal en el tiempo")

    st.markdown(
        """
Esta guía introduce el comportamiento básico de una transmisión digital en presencia de ruido.
El objetivo es observar cómo una secuencia binaria puede representarse como pulsos y símbolos,
cómo el ruido altera esos valores y cómo el receptor puede tomar decisiones incorrectas.

En esta versión se evita representar la señal como una curva continua. Las gráficas se muestran
como secuencias discretas, porque el análisis se realiza bit a bit.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Simulación",
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

Comprender cómo el ruido afecta una señal digital discreta y cómo dicha alteración puede
provocar errores en la decisión del receptor.

**Objetivos específicos**

1. Representar una secuencia binaria mediante pulsos discretos.
2. Convertir bits en símbolos BPSK.
3. Interpretar el ruido como una variable aleatoria.
4. Modificar la media, desviación estándar y varianza del ruido.
5. Observar cómo el ruido altera las muestras recibidas.
6. Aplicar una regla de decisión por umbral.
7. Calcular la tasa de error de bit.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
Un sistema de comunicación digital transmite información desde un transmisor hacia un receptor.
En este caso, la información se representa mediante bits discretos. Para modelar una transmisión
simple, cada bit se transforma en un símbolo:

$$
0 \\rightarrow -1
$$

$$
1 \\rightarrow +1
$$

El canal se modela mediante:

$$
r = s + n
$$

donde:

- $r$ es el valor recibido;
- $s$ es el símbolo transmitido;
- $n$ es el ruido agregado por el canal.

El ruido se modela como una variable aleatoria gaussiana:

$$
n \\sim \\mathcal{N}(\\mu, \\sigma^2)
$$

donde:

- $\\mu$ es la media;
- $\\sigma$ es la desviación estándar;
- $\\sigma^2$ es la varianza.

La decisión del receptor se realiza con un umbral en cero:

$$
r \\geq 0 \\Rightarrow \\hat{b}=1
$$

$$
r < 0 \\Rightarrow \\hat{b}=0
$$

Si el ruido desplaza el valor recibido al lado contrario del umbral, el receptor decide
un bit incorrecto.

**Sobre la semilla**

La semilla es un número que permite repetir una misma simulación. Si se mantienen la misma
secuencia, la misma desviación estándar y la misma semilla, el ruido generado será el mismo.
Esto permite comparar resultados y verificar ejercicios de forma reproducible.
"""
        )

    with tabs[2]:
        st.header("Simulación interactiva")

        col_entrada, col_info = st.columns([1, 1])

        with col_entrada:
            modo = st.radio(
                "Modo de entrada",
                ["Secuencia manual", "Secuencia aleatoria"],
                key="g1_modo_entrada",
            )

            if modo == "Secuencia manual":
                bits = st.text_input(
                    "Secuencia binaria transmitida",
                    value="1011001",
                    help="Ingrese únicamente ceros y unos.",
                    key="g1_bits_manual",
                ).strip()
            else:
                cantidad_bits = st.selectbox(
                    "Cantidad de bits",
                    [8, 16, 32, 64],
                    index=1,
                    key="g1_cantidad_bits",
                )

                semilla_bits = st.number_input(
                    "Semilla para generar bits",
                    min_value=0,
                    max_value=999999,
                    value=10,
                    step=1,
                    key="g1_semilla_bits",
                )

                rng = np.random.default_rng(int(semilla_bits))
                bits_generados = rng.integers(0, 2, size=cantidad_bits)
                bits = "".join(str(bit) for bit in bits_generados)

                st.code(f"Bits generados: {bits}", language="text")

            media = st.number_input(
                "Media del ruido μ",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                key="g1_media_ruido",
            )

            sigma = st.slider(
                "Desviación estándar del ruido σ",
                min_value=0.0,
                max_value=2.0,
                value=0.30,
                step=0.05,
                key="g1_sigma_ruido",
            )

            semilla_ruido = st.number_input(
                "Semilla del ruido",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                key="g1_semilla_ruido",
            )

            ejecutar = st.button("Ejecutar simulación", width="stretch")

        with col_info:
            st.info(
                """
La desviación estándar controla la dispersión del ruido.  
La varianza se calcula como:

$$
\\sigma^2
$$

Si aumenta $\\sigma$, las muestras de ruido pueden alejarse más de la media y provocar
más cruces del umbral de decisión.
"""
            )

            st.metric("Varianza del ruido σ²", f"{sigma**2:.4f}")

        if not validar_bits(bits):
            st.error("La secuencia ingresada no es válida. Use únicamente 0 y 1.")
            return

        if ejecutar:
            bits_originales = np.array([int(bit) for bit in bits])
            simbolos = bits_a_simbolos_bpsk(bits)

            ruido = generar_ruido_gaussiano(
                cantidad=len(bits),
                media=media,
                sigma=sigma,
                semilla=int(semilla_ruido),
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

            secuencia_decidida = "".join(str(bit) for bit in bits_decididos)

            st.code(
                f"Bits transmitidos: {bits}\n"
                f"Bits decididos:    {secuencia_decidida}",
                language="text",
            )

            st.subheader("Tabla del proceso")
            st.dataframe(tabla, width="stretch", hide_index=True)

            st.subheader("1. Pulsos discretos de bits")
            graficar_pulsos_bits(bits)

            st.subheader("2. Símbolos BPSK discretos")
            graficar_simbolos_bpsk(bits, simbolos)

            st.subheader("3. Ruido por muestra")
            graficar_ruido_discreto(ruido)

            st.subheader("4. Símbolos transmitidos vs valores recibidos")
            graficar_senal_recibida(simbolos, recibida)

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
                    "No se produjeron errores. Ningún valor recibido cruzó el umbral incorrectamente."
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
                "media": media,
                "semilla_ruido": int(semilla_ruido),
            }

    with tabs[3]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección combina la interpretación de resultados con actividades guiadas.  
Primero revise la última simulación ejecutada y luego realice las pruebas propuestas.
"""
        )

        if "guia_01_resultados" in st.session_state:
            resultados = st.session_state["guia_01_resultados"]

            st.subheader("Resumen de la última simulación")

            st.table(
                pd.DataFrame(
                    [
                        {
                            "Secuencia": resultados["bits"],
                            "Media μ": resultados["media"],
                            "σ": resultados["sigma"],
                            "σ²": resultados["varianza"],
                            "Semilla": resultados["semilla_ruido"],
                            "Errores": resultados["errores"],
                            "BER": resultados["ber"],
                        }
                    ]
                )
            )
        else:
            st.info("Ejecute primero una simulación en la pestaña Simulación.")

        st.subheader("Actividades guiadas")

        st.markdown(
            """
Realice las siguientes pruebas:

1. Use la misma secuencia con $\\sigma = 0.10$.
2. Repita con $\\sigma = 0.50$.
3. Repita con $\\sigma = 1.00$.
4. Observe cómo cambia la dispersión del ruido.
5. Identifique si los valores recibidos cruzan el umbral.
6. Cambie la semilla del ruido y observe cómo cambia la realización del experimento.
7. Explique por qué el BER puede cambiar aunque la secuencia sea la misma.
"""
        )

        st.subheader("Preguntas de análisis")

        respuesta_1 = st.radio(
            "Pregunta 1: ¿Por qué se usan gráficas discretas en esta guía?",
            [
                "Porque los bits y las muestras se analizan por posiciones discretas.",
                "Porque el canal no tiene ruido.",
                "Porque el BER siempre es cero.",
                "Porque la señal no puede representarse numéricamente.",
            ],
            index=None,
            key="g1_pregunta_discreta",
        )

        if respuesta_1:
            if respuesta_1 == "Porque los bits y las muestras se analizan por posiciones discretas.":
                st.success("Correcto. El análisis se realiza bit a bit y muestra a muestra.")
            else:
                st.error("Revise la diferencia entre representación discreta y continua.")

        respuesta_2 = st.radio(
            "Pregunta 2: ¿Qué representa σ²?",
            [
                "La media del ruido.",
                "La varianza del ruido.",
                "El número de bits transmitidos.",
                "La palabra Hamming.",
            ],
            index=None,
            key="g1_pregunta_varianza",
        )

        if respuesta_2:
            if respuesta_2 == "La varianza del ruido.":
                st.success("Correcto. σ² representa la varianza del ruido.")
            else:
                st.error("Revise la relación entre desviación estándar y varianza.")

        respuesta_3 = st.radio(
            "Pregunta 3: ¿Cuándo ocurre un error de decisión?",
            [
                "Cuando el valor recibido cruza al lado incorrecto del umbral.",
                "Cuando el ruido tiene media cero.",
                "Cuando el bit transmitido es siempre 1.",
                "Cuando la semilla es fija.",
            ],
            index=None,
            key="g1_pregunta_error",
        )

        if respuesta_3:
            if respuesta_3 == "Cuando el valor recibido cruza al lado incorrecto del umbral.":
                st.success(
                    "Correcto. El receptor decide mal cuando el ruido desplaza la muestra al lado opuesto del umbral."
                )
            else:
                st.error("Revise el criterio de decisión por umbral.")

        respuesta_4 = st.radio(
            "Pregunta 4: ¿Qué permite la semilla dentro de la simulación?",
            [
                "Eliminar completamente el ruido.",
                "Repetir el mismo experimento aleatorio bajo los mismos parámetros.",
                "Reducir automáticamente el BER a cero.",
                "Cambiar el código Hamming.",
            ],
            index=None,
            key="g1_pregunta_semilla",
        )

        if respuesta_4:
            if respuesta_4 == "Repetir el mismo experimento aleatorio bajo los mismos parámetros.":
                st.success("Correcto. La semilla hace reproducible la simulación.")
            else:
                st.error("Revise el significado de la semilla en simulaciones aleatorias.")

    with tabs[4]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- una secuencia binaria se analiza de forma discreta;
- los bits pueden representarse como pulsos;
- los símbolos BPSK representan los bits mediante niveles +1 y -1;
- el ruido se modela como variable aleatoria;
- la desviación estándar controla la dispersión del ruido;
- la varianza corresponde a $\\sigma^2$;
- un error ocurre cuando el ruido desplaza una muestra al lado incorrecto del umbral;
- el BER permite cuantificar la proporción de errores;
- la semilla permite repetir un mismo experimento aleatorio.
"""
        )

    with tabs[5]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2012). *Data communications and networking* (5th ed.). McGraw-Hill.

Proakis, J. G., & Salehi, M. (2008). *Digital communications* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.
"""
        )