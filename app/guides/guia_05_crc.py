import math
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st


def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_generador(generador: str) -> bool:
    """
    Valida un polinomio generador binario.
    Debe iniciar y terminar en 1 para ser útil en CRC básico.
    """
    return (
        len(generador) >= 2
        and all(bit in "01" for bit in generador)
        and generador[0] == "1"
        and generador[-1] == "1"
    )


def division_modulo_2(dividendo: str, divisor: str) -> Tuple[str, pd.DataFrame]:
    """
    Realiza división módulo 2 para CRC y devuelve el residuo junto con una tabla de pasos.
    """
    trabajo = list(dividendo)
    divisor_bits = list(divisor)
    n = len(divisor_bits)
    pasos: List[Dict[str, object]] = []

    for i in range(len(dividendo) - n + 1):
        segmento_antes = "".join(trabajo[i : i + n])

        if trabajo[i] == "1":
            for j in range(n):
                trabajo[i + j] = str(int(trabajo[i + j]) ^ int(divisor_bits[j]))

            segmento_despues = "".join(trabajo[i : i + n])

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

    residuo = "".join(trabajo[-(n - 1) :])
    return residuo, pd.DataFrame(pasos)


def generar_crc(datos: str, generador: str) -> Tuple[str, str, pd.DataFrame]:
    """
    Genera el residuo CRC y la trama final.
    """
    ceros = "0" * (len(generador) - 1)
    dividendo = datos + ceros
    residuo, pasos = division_modulo_2(dividendo, generador)
    trama = datos + residuo
    return residuo, trama, pasos


def verificar_crc(trama: str, generador: str) -> Tuple[bool, str, pd.DataFrame]:
    """
    Verifica una trama con CRC. Si el residuo es cero, no se detecta error.
    """
    residuo, pasos = division_modulo_2(trama, generador)
    valido = all(bit == "0" for bit in residuo)
    return valido, residuo, pasos


def invertir_bit(bits: str, posicion: int) -> str:
    lista = list(bits)
    indice = posicion - 1
    lista[indice] = "1" if lista[indice] == "0" else "0"
    return "".join(lista)


def generar_bits_aleatorios(cantidad: int, semilla: int | None = None) -> str:
    rng = np.random.default_rng(semilla)
    bits = rng.integers(0, 2, size=cantidad)
    return "".join(str(bit) for bit in bits)


def dividir_en_tramas(bits: str, tamano_payload: int) -> List[str]:
    """
    Divide los datos en tramas de tamaño fijo.
    Si la última trama queda incompleta, se rellena con ceros.
    """
    tramas = []

    for i in range(0, len(bits), tamano_payload):
        bloque = bits[i : i + tamano_payload]

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
    Transmite una trama binaria usando BPSK + ruido gaussiano + decisión por umbral.
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


def simular_crc_estadistico(
    cantidad_bits_datos: int,
    tamano_payload: int,
    generador: str,
    sigma: float,
    semilla: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simula muchas tramas con CRC a través de canal AWGN.
    Calcula estadísticas de detección.
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

        trama_rx, simbolos, ruido, recibido_analogico = transmitir_awgn(
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

    ber = total_errores_bit / total_bits_codificados if total_bits_codificados else 0
    fer = tramas_con_error / len(tramas_datos) if tramas_datos else 0
    tasa_deteccion = tramas_detectadas / tramas_con_error if tramas_con_error else 0
    tasa_no_detectada = tramas_no_detectadas / tramas_con_error if tramas_con_error else 0

    resumen = {
        "Tramas evaluadas": len(tramas_datos),
        "Bits codificados evaluados": total_bits_codificados,
        "Errores de bit": total_errores_bit,
        "BER del canal": ber,
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


def render_guia_05() -> None:
    st.title("Guía 5: CRC y detección de errores remanentes")

    st.markdown(
        """
Esta guía estudia el Código de Redundancia Cíclica (CRC) como mecanismo de detección
de errores. A diferencia de Hamming, CRC no corrige bits alterados, pero permite
detectar errores que permanecen después de la transmisión o después de un proceso de
corrección incompleto.

La guía combina el procedimiento algebraico del CRC con simulaciones estadísticas
sobre muchas tramas transmitidas por un canal con ruido gaussiano.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "CRC paso a paso",
            "Señal y canal",
            "Estadística con muchas tramas",
            "Comparación",
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

Comprender el funcionamiento del CRC como técnica de detección de errores y analizar
su desempeño estadístico bajo condiciones de ruido.

**Objetivos específicos**

1. Representar una secuencia binaria mediante un polinomio sobre GF(2).
2. Calcular el residuo CRC mediante división módulo 2.
3. Construir una trama transmitida formada por datos y residuo CRC.
4. Verificar una trama recibida usando el residuo de la división.
5. Observar el efecto del ruido sobre una trama con CRC en el tiempo.
6. Evaluar muchas tramas para estimar BER, FER y tasa de detección.
7. Analizar errores detectados y errores no detectados.
8. Relacionar el CRC con la necesidad de detectar errores remanentes después de Hamming.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
El Código de Redundancia Cíclica (CRC) es una técnica de detección de errores basada
en aritmética binaria módulo 2. En este tipo de operación, la suma y la resta se realizan
mediante XOR.

El transmisor toma un mensaje binario y lo divide entre un polinomio generador. El
residuo de esa división se agrega al final del mensaje para formar la trama transmitida.

Si el mensaje es $M(x)$ y el polinomio generador es $G(x)$, la trama transmitida puede
expresarse como:

$$
T(x) = M(x)x^r + R(x)
$$

donde:

- $r$ es el grado del polinomio generador;
- $R(x)$ es el residuo CRC.

En el receptor se divide la trama recibida entre el mismo generador:

$$
Residuo\\left(\\frac{T(x)}{G(x)}\\right)
$$

Si el residuo es cero, no se detecta error. Si el residuo es diferente de cero, se detecta
una alteración en la trama.

El CRC no corrige errores. Su valor dentro del sistema propuesto está en detectar
errores remanentes, especialmente cuando Hamming no puede corregir adecuadamente
errores múltiples.

En esta guía se combina el análisis algebraico con simulaciones sobre muchas tramas,
usando ruido gaussiano y métricas estadísticas:

$$
BER = \\frac{\\text{bits erróneos}}{\\text{bits transmitidos}}
$$

$$
FER = \\frac{\\text{tramas con error}}{\\text{tramas transmitidas}}
$$

También se calcula la razón señal-ruido:

$$
SNR = \\frac{P_s}{P_n}
$$

y la potencia del ruido se relaciona con la varianza:

$$
P_n \\approx \\sigma^2
$$
"""
        )

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

            ejecutar = st.button("Calcular CRC", use_container_width=True)

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
            st.dataframe(pasos, use_container_width=True, hide_index=True)

            st.session_state["g5_trama_manual_resultado"] = trama
            st.session_state["g5_generador_manual_resultado"] = generador

    with tabs[3]:
        st.header("Señal, ruido y verificación CRC")

        st.markdown(
            """
En esta sección se transmite una trama con CRC a través de un canal con ruido gaussiano.
La trama se representa mediante símbolos BPSK, se suma ruido y el receptor toma una
decisión por umbral. Luego se aplica la verificación CRC.
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

            ejecutar_senal = st.button("Transmitir trama con CRC", use_container_width=True)

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

            df_senal = pd.DataFrame(
                {
                    "Posición": np.arange(1, len(trama_tx) + 1),
                    "Símbolo Tx": simbolos,
                    "Ruido": ruido,
                    "Valor recibido": recibido_analogico,
                    "Umbral": np.zeros(len(trama_tx)),
                }
            ).set_index("Posición")

            st.subheader("Señal en el tiempo")
            st.line_chart(df_senal[["Símbolo Tx", "Valor recibido", "Umbral"]])

            st.subheader("Ruido generado")
            st.line_chart(df_senal[["Ruido"]])

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
                [100, 1000, 10000, 50000],
                index=2,
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

            ejecutar_est = st.button("Ejecutar simulación estadística", use_container_width=True)

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
            st.session_state["g5_df_estadistico"] = df_resultados

    with tabs[5]:
        st.header("Comparación de escenarios")

        st.markdown(
            """
En esta sección se comparan varios niveles de ruido. Esto permite observar cómo cambian
el BER, el FER, la SNR y la capacidad de detección del CRC.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            bits_comp = st.selectbox(
                "Bits de datos por comparación",
                [1000, 10000, 50000],
                index=1,
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

            ejecutar_comp = st.button("Comparar escenarios", use_container_width=True)

        with col_info:
            st.info(
                """
La comparación estadística permite observar tendencias generales. Al aumentar σ, aumenta
la potencia del ruido, disminuye la SNR y se espera que aumenten BER y FER.
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
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.subheader("BER del canal vs σ")
            st.line_chart(df_comp[["σ", "BER del canal"]].set_index("σ"))

            st.subheader("FER vs σ")
            st.line_chart(df_comp[["σ", "FER"]].set_index("σ"))

            st.subheader("Tasa de errores no detectados vs σ")
            st.line_chart(df_comp[["σ", "Tasa de error no detectado"]].set_index("σ"))

            df_snr = df_comp.replace([np.inf, -np.inf], np.nan).dropna(subset=["SNR dB promedio"])
            if not df_snr.empty:
                st.subheader("BER del canal vs SNR dB")
                st.line_chart(
                    df_snr[["SNR dB promedio", "BER del canal"]]
                    .sort_values("SNR dB promedio")
                    .set_index("SNR dB promedio")
                )

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

    with tabs[6]:
        st.header("Dinámica de aprendizaje")

        st.markdown(
            """
Realice las siguientes actividades:

1. Calcule manualmente el CRC del mensaje `1101` usando el generador `1011`.
2. Compare sus pasos con la tabla de división módulo 2 de la app.
3. Transmita una trama con σ = 0.10 y observe si hay errores.
4. Repita con σ = 0.80 y compare el resultado.
5. Ejecute una simulación estadística con 10,000 bits.
6. Compare BER, FER y tasa de detección.
7. Explique por qué CRC detecta errores, pero no los corrige.
8. Relacione esta guía con la limitación de Hamming observada en la Guía 4.
"""
        )

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
- la presencia de errores no detectados muestra que ningún método simple es absoluto;
- CRC complementa a Hamming porque permite detectar errores remanentes o no corregibles.
"""
        )

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