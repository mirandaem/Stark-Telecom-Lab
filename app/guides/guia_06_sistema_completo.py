import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Utilidades generales
# ============================================================

def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_generador(generador: str) -> bool:
    return (
        len(generador) >= 2
        and all(bit in "01" for bit in generador)
        and generador[0] == "1"
        and generador[-1] == "1"
    )


def generar_bits_aleatorios(cantidad: int, semilla: int | None = None) -> str:
    rng = np.random.default_rng(semilla)
    bits = rng.integers(0, 2, size=cantidad)
    return "".join(str(bit) for bit in bits)


def dividir_en_bloques(bits: str, tamano: int) -> List[str]:
    return [bits[i:i + tamano] for i in range(0, len(bits), tamano)]


def rellenar_a_multiplo(bits: str, multiplo: int) -> Tuple[str, int]:
    residuo = len(bits) % multiplo

    if residuo == 0:
        return bits, 0

    padding = multiplo - residuo
    return bits + ("0" * padding), padding


def quitar_padding(bits: str, padding: int) -> str:
    if padding == 0:
        return bits
    return bits[:-padding]


def contar_errores(tx: str, rx: str) -> int:
    longitud = min(len(tx), len(rx))
    return sum(1 for a, b in zip(tx[:longitud], rx[:longitud]) if a != b)


def calcular_ber(tx: str, rx: str) -> float:
    if len(tx) == 0:
        return 0.0
    return contar_errores(tx, rx) / len(tx)


# ============================================================
# CRC
# ============================================================

def division_modulo_2(dividendo: str, divisor: str) -> Tuple[str, pd.DataFrame]:
    trabajo = list(dividendo)
    divisor_bits = list(divisor)
    n = len(divisor_bits)

    pasos = []

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
    ceros = "0" * (len(generador) - 1)
    dividendo = datos + ceros
    residuo, pasos = division_modulo_2(dividendo, generador)
    trama = datos + residuo
    return residuo, trama, pasos


def verificar_crc(trama: str, generador: str) -> Tuple[bool, str, pd.DataFrame]:
    residuo, pasos = division_modulo_2(trama, generador)
    valido = all(bit == "0" for bit in residuo)
    return valido, residuo, pasos


# ============================================================
# Hamming (7,4)
# Estructura: [P1 P2 D1 P4 D2 D3 D4]
# ============================================================

def hamming_encode_4(bits4: str) -> str:
    d1, d2, d3, d4 = [int(bit) for bit in bits4]

    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4

    codigo = [p1, p2, d1, p4, d2, d3, d4]
    return "".join(str(bit) for bit in codigo)


def hamming_encode_stream(bits: str) -> Tuple[str, int, pd.DataFrame]:
    bits_rellenados, padding = rellenar_a_multiplo(bits, 4)
    bloques = dividir_en_bloques(bits_rellenados, 4)

    codificados = []
    filas = []

    for i, bloque in enumerate(bloques, start=1):
        codigo = hamming_encode_4(bloque)
        codificados.append(codigo)

        filas.append(
            {
                "Bloque": i,
                "Datos 4 bits": bloque,
                "Hamming 7 bits": codigo,
            }
        )

    return "".join(codificados), padding, pd.DataFrame(filas)


def calcular_sindrome_7(bits7: str) -> Dict[str, int | str]:
    b = [int(bit) for bit in bits7]

    s1 = b[0] ^ b[2] ^ b[4] ^ b[6]
    s2 = b[1] ^ b[2] ^ b[5] ^ b[6]
    s4 = b[3] ^ b[4] ^ b[5] ^ b[6]

    posicion_error = s1 + (2 * s2) + (4 * s4)

    return {
        "s1": s1,
        "s2": s2,
        "s4": s4,
        "sindrome": f"{s4}{s2}{s1}",
        "posicion_error": posicion_error,
    }


def invertir_bit(bits: str, posicion: int) -> str:
    lista = list(bits)
    indice = posicion - 1
    lista[indice] = "1" if lista[indice] == "0" else "0"
    return "".join(lista)


def hamming_decode_7(bits7: str) -> Dict[str, object]:
    sindrome = calcular_sindrome_7(bits7)
    posicion_error = int(sindrome["posicion_error"])

    if posicion_error == 0:
        corregido = bits7
        corregido_por_hamming = False
    else:
        corregido = invertir_bit(bits7, posicion_error)
        corregido_por_hamming = True

    datos = corregido[2] + corregido[4] + corregido[5] + corregido[6]

    return {
        "recibido": bits7,
        "sindrome": sindrome["sindrome"],
        "posicion_error": posicion_error,
        "corregido": corregido,
        "datos": datos,
        "corrigio": corregido_por_hamming,
    }


def hamming_decode_stream(bits_codificados: str) -> Tuple[str, str, pd.DataFrame]:
    bloques = dividir_en_bloques(bits_codificados, 7)

    datos_decodificados = []
    bloques_corregidos = []
    filas = []

    for i, bloque in enumerate(bloques, start=1):
        if len(bloque) != 7:
            continue

        resultado = hamming_decode_7(bloque)

        datos_decodificados.append(str(resultado["datos"]))
        bloques_corregidos.append(str(resultado["corregido"]))

        filas.append(
            {
                "Bloque": i,
                "Recibido": resultado["recibido"],
                "Síndrome": resultado["sindrome"],
                "Posición detectada": resultado["posicion_error"],
                "Corregido": resultado["corregido"],
                "Datos recuperados": resultado["datos"],
                "Hamming corrigió": resultado["corrigio"],
            }
        )

    return "".join(datos_decodificados), "".join(bloques_corregidos), pd.DataFrame(filas)


# ============================================================
# Canal AWGN con BPSK
# ============================================================

def bits_a_simbolos_bpsk(bits: str) -> np.ndarray:
    bits_array = np.array([int(bit) for bit in bits])
    return np.where(bits_array == 1, 1.0, -1.0)


def transmitir_awgn(
    bits: str,
    sigma: float,
    semilla: int | None = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semilla)

    simbolos = bits_a_simbolos_bpsk(bits)
    ruido = rng.normal(loc=0.0, scale=sigma, size=len(simbolos))
    recibido_analogico = simbolos + ruido
    bits_decididos = np.where(recibido_analogico >= 0, 1, 0)

    bits_rx = "".join(str(bit) for bit in bits_decididos)

    return bits_rx, simbolos, ruido, recibido_analogico


def calcular_potencias(simbolos: np.ndarray, ruido: np.ndarray) -> Tuple[float, float, float, float]:
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
# Pipeline completo: CRC -> Hamming -> Canal -> Hamming Rx -> CRC
# ============================================================

def ejecutar_pipeline_completo(
    datos: str,
    generador_crc: str,
    sigma: float,
    semilla: int | None = None,
) -> Dict[str, object]:
    crc_tx, trama_crc_tx, pasos_crc_tx = generar_crc(datos, generador_crc)

    hamming_tx, padding, tabla_hamming_tx = hamming_encode_stream(trama_crc_tx)

    hamming_rx, simbolos, ruido, recibido_analogico = transmitir_awgn(
        hamming_tx,
        sigma=sigma,
        semilla=semilla,
    )

    datos_decodificados_padded, hamming_corregido, tabla_hamming_rx = hamming_decode_stream(hamming_rx)
    datos_decodificados = quitar_padding(datos_decodificados_padded, padding)

    crc_valido, residuo_rx, pasos_crc_rx = verificar_crc(datos_decodificados, generador_crc)

    longitud_crc = len(generador_crc) - 1
    datos_recuperados = datos_decodificados[:-longitud_crc] if longitud_crc > 0 else datos_decodificados

    errores_canal_codificado = contar_errores(hamming_tx, hamming_rx)
    ber_canal_codificado = calcular_ber(hamming_tx, hamming_rx)

    errores_post_hamming = contar_errores(trama_crc_tx, datos_decodificados)
    ber_post_hamming = calcular_ber(trama_crc_tx, datos_decodificados)

    errores_datos_final = contar_errores(datos, datos_recuperados)
    ber_datos_final = calcular_ber(datos, datos_recuperados)

    ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

    estado_final = "Aceptada" if crc_valido else "Rechazada por CRC"

    resumen = {
        "Datos originales": datos,
        "CRC Tx": crc_tx,
        "Trama con CRC": trama_crc_tx,
        "Padding Hamming": padding,
        "Hamming Tx": hamming_tx,
        "Hamming Rx": hamming_rx,
        "Hamming corregido": hamming_corregido,
        "Trama recuperada post-Hamming": datos_decodificados,
        "Datos recuperados": datos_recuperados,
        "Residuo CRC Rx": residuo_rx,
        "CRC válido": crc_valido,
        "Estado final": estado_final,
        "Errores canal codificado": errores_canal_codificado,
        "BER canal codificado": ber_canal_codificado,
        "Errores post-Hamming": errores_post_hamming,
        "BER post-Hamming": ber_post_hamming,
        "Errores datos finales": errores_datos_final,
        "BER datos finales": ber_datos_final,
        "Potencia señal": ps,
        "Potencia ruido": pn,
        "SNR": snr,
        "SNR dB": snr_db,
        "σ": sigma,
        "σ²": sigma**2,
    }

    return {
        "resumen": resumen,
        "tabla_hamming_tx": tabla_hamming_tx,
        "tabla_hamming_rx": tabla_hamming_rx,
        "pasos_crc_tx": pasos_crc_tx,
        "pasos_crc_rx": pasos_crc_rx,
        "simbolos": simbolos,
        "ruido": ruido,
        "recibido_analogico": recibido_analogico,
    }


# ============================================================
# Escenarios comparativos
# ============================================================

def escenario_sin_proteccion(
    datos: str,
    sigma: float,
    semilla: int | None = None,
) -> Dict[str, object]:
    rx, simbolos, ruido, _ = transmitir_awgn(datos, sigma, semilla)
    errores = contar_errores(datos, rx)
    ber = calcular_ber(datos, rx)
    ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

    return {
        "Escenario": "Sin protección",
        "Bits útiles": len(datos),
        "Bits transmitidos": len(datos),
        "Errores finales": errores,
        "BER final": ber,
        "Estado": "Sin detección ni corrección",
        "SNR dB": snr_db,
    }


def escenario_crc_solo(
    datos: str,
    generador: str,
    sigma: float,
    semilla: int | None = None,
) -> Dict[str, object]:
    crc_tx, trama_tx, _ = generar_crc(datos, generador)
    trama_rx, simbolos, ruido, _ = transmitir_awgn(trama_tx, sigma, semilla)

    valido, residuo, _ = verificar_crc(trama_rx, generador)

    crc_len = len(generador) - 1
    datos_rx = trama_rx[:-crc_len] if crc_len > 0 else trama_rx

    errores = contar_errores(datos, datos_rx)
    ber = calcular_ber(datos, datos_rx)

    ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

    return {
        "Escenario": "CRC solo",
        "Bits útiles": len(datos),
        "Bits transmitidos": len(trama_tx),
        "Errores finales": errores,
        "BER final": ber,
        "Estado": "Aceptada" if valido else "Rechazada por CRC",
        "SNR dB": snr_db,
    }


def escenario_hamming_solo(
    datos: str,
    sigma: float,
    semilla: int | None = None,
) -> Dict[str, object]:
    hamming_tx, padding, _ = hamming_encode_stream(datos)
    hamming_rx, simbolos, ruido, _ = transmitir_awgn(hamming_tx, sigma, semilla)

    datos_decodificados_padded, _, _ = hamming_decode_stream(hamming_rx)
    datos_rx = quitar_padding(datos_decodificados_padded, padding)

    errores = contar_errores(datos, datos_rx)
    ber = calcular_ber(datos, datos_rx)

    ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

    return {
        "Escenario": "Hamming solo",
        "Bits útiles": len(datos),
        "Bits transmitidos": len(hamming_tx),
        "Errores finales": errores,
        "BER final": ber,
        "Estado": "Corrige 1 bit por bloque",
        "SNR dB": snr_db,
    }


def escenario_hamming_crc(
    datos: str,
    generador: str,
    sigma: float,
    semilla: int | None = None,
) -> Dict[str, object]:
    resultado = ejecutar_pipeline_completo(datos, generador, sigma, semilla)
    resumen = resultado["resumen"]

    return {
        "Escenario": "Hamming + CRC",
        "Bits útiles": len(datos),
        "Bits transmitidos": len(resumen["Hamming Tx"]),
        "Errores finales": resumen["Errores datos finales"],
        "BER final": resumen["BER datos finales"],
        "Estado": resumen["Estado final"],
        "SNR dB": resumen["SNR dB"],
    }


def comparar_escenarios(
    datos: str,
    generador: str,
    sigma: float,
    semilla: int | None = None,
) -> pd.DataFrame:
    semillas = {
        "sin": None if semilla is None else semilla + 1,
        "crc": None if semilla is None else semilla + 2,
        "hamming": None if semilla is None else semilla + 3,
        "hamming_crc": None if semilla is None else semilla + 4,
    }

    filas = [
        escenario_sin_proteccion(datos, sigma, semillas["sin"]),
        escenario_crc_solo(datos, generador, sigma, semillas["crc"]),
        escenario_hamming_solo(datos, sigma, semillas["hamming"]),
        escenario_hamming_crc(datos, generador, sigma, semillas["hamming_crc"]),
    ]

    return pd.DataFrame(filas)


def simulacion_estadistica_final(
    cantidad_bits: int,
    generador: str,
    sigma: float,
    semilla: int | None = None,
) -> pd.DataFrame:
    datos = generar_bits_aleatorios(cantidad_bits, semilla)
    return comparar_escenarios(datos, generador, sigma, semilla)


def comparar_sigmas_sistema_completo(
    cantidad_bits: int,
    generador: str,
    valores_sigma: List[float],
    semilla: int | None = None,
) -> pd.DataFrame:
    filas = []

    for i, sigma in enumerate(valores_sigma):
        semilla_escenario = None if semilla is None else semilla + (1000 * i)
        datos = generar_bits_aleatorios(cantidad_bits, semilla_escenario)

        resultado = escenario_hamming_crc(
            datos=datos,
            generador=generador,
            sigma=sigma,
            semilla=semilla_escenario,
        )

        resultado["σ"] = sigma
        resultado["σ²"] = sigma**2

        filas.append(resultado)

    return pd.DataFrame(filas)


# ============================================================
# Interfaz Streamlit
# ============================================================

def render_guia_06() -> None:
    st.title("Guía 6: Sistema completo Hamming + CRC y comparación de desempeño")

    st.markdown(
        """
Esta guía integra los elementos desarrollados en las guías anteriores: ruido, BER,
SNR, Hamming, síndrome y CRC. El objetivo es evaluar el desempeño de un sistema
completo de transmisión digital que utiliza corrección de errores mediante Hamming
y detección final mediante CRC.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Flujo completo",
            "Señal y receptor",
            "Comparación de escenarios",
            "Estadística final",
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

Evaluar el desempeño de un sistema de comunicación digital que integra CRC, Hamming,
canal con ruido y verificación final mediante métricas estadísticas.

**Objetivos específicos**

1. Implementar el flujo completo: mensaje, CRC, Hamming, canal, corrección y verificación.
2. Observar el efecto del ruido sobre la trama codificada.
3. Calcular BER antes y después de la corrección.
4. Verificar la detección de errores remanentes mediante CRC.
5. Comparar escenarios sin protección, con CRC, con Hamming y con Hamming + CRC.
6. Analizar el impacto de la desviación estándar del ruido sobre el desempeño.
7. Relacionar la mejora obtenida con los conceptos de redundancia, corrección y detección.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En un sistema digital real, el control de errores no depende de una sola técnica. Es
frecuente combinar mecanismos de corrección y detección para mejorar la confiabilidad.

En esta guía se utiliza el siguiente flujo:

$$
Mensaje \\rightarrow CRC \\rightarrow Hamming \\rightarrow Canal \\rightarrow Hamming^{-1} \\rightarrow CRC^{-1}
$$

El CRC se aplica primero para agregar un residuo de detección al mensaje. Luego, la
secuencia resultante se divide en bloques de 4 bits y se codifica con Hamming (7,4).
Después de pasar por el canal con ruido, el receptor aplica el síndrome de Hamming para
corregir errores de un bit por bloque. Finalmente, el CRC verifica si la información
recuperada contiene errores remanentes.

El canal se modela mediante:

$$
r = s + n
$$

donde el ruido se considera gaussiano:

$$
n \\sim \\mathcal{N}(0, \\sigma^2)
$$

Para medir el desempeño se utilizan:

$$
BER = \\frac{N_e}{N_t}
$$

$$
SNR = \\frac{P_s}{P_n}
$$

El objetivo principal no es solo observar si la trama llega correctamente, sino comparar
cuantitativamente cómo mejora el sistema cuando se agrega redundancia para corrección
y detección.
"""
        )

    with tabs[2]:
        st.header("Flujo completo Hamming + CRC")

        st.markdown(
            """
En esta sección se ejecuta el flujo completo para un mensaje binario definido por el
usuario. La aplicación muestra cada etapa del transmisor, canal y receptor.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            datos = st.text_input(
                "Mensaje binario",
                value="101100111001",
                key="g6_datos_flujo",
            ).strip()

            generador = st.text_input(
                "Generador CRC",
                value="1011",
                key="g6_generador_flujo",
            ).strip()

            sigma = st.slider(
                "Desviación estándar del ruido σ",
                min_value=0.0,
                max_value=2.0,
                value=0.50,
                step=0.05,
                key="g6_sigma_flujo",
            )

            semilla = st.number_input(
                "Semilla",
                min_value=0,
                max_value=999999,
                value=300,
                step=1,
                key="g6_semilla_flujo",
            )

            ejecutar = st.button("Ejecutar sistema completo", use_container_width=True)

        with col_info:
            st.info(
                """
La secuencia se protege primero con CRC y luego con Hamming. Después del canal, Hamming
intenta corregir errores de un bit por bloque y CRC valida si quedan errores remanentes.
"""
            )
            st.metric("Varianza del ruido σ²", f"{sigma**2:.4f}")

        if not validar_bits(datos):
            st.error("El mensaje debe contener únicamente 0 y 1.")
        elif not validar_generador(generador):
            st.error("El generador CRC debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar:
            resultado = ejecutar_pipeline_completo(datos, generador, sigma, int(semilla))
            resumen = resultado["resumen"]

            st.subheader("Resumen del sistema")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BER canal codificado", f"{resumen['BER canal codificado']:.6f}")
            c2.metric("BER post-Hamming", f"{resumen['BER post-Hamming']:.6f}")
            c3.metric("BER datos finales", f"{resumen['BER datos finales']:.6f}")
            c4.metric("Estado final", resumen["Estado final"])

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Errores canal", resumen["Errores canal codificado"])
            c6.metric("Errores post-Hamming", resumen["Errores post-Hamming"])
            c7.metric("Errores datos finales", resumen["Errores datos finales"])
            c8.metric(
                "SNR dB",
                "∞" if math.isinf(resumen["SNR dB"]) else f"{resumen['SNR dB']:.2f}",
            )

            st.subheader("Secuencias principales")

            st.code(
                f"Datos originales:            {resumen['Datos originales']}\n"
                f"CRC Tx:                       {resumen['CRC Tx']}\n"
                f"Trama con CRC:                {resumen['Trama con CRC']}\n"
                f"Hamming Tx:                   {resumen['Hamming Tx']}\n"
                f"Hamming Rx:                   {resumen['Hamming Rx']}\n"
                f"Hamming corregido:            {resumen['Hamming corregido']}\n"
                f"Trama recuperada post-Hamming:{resumen['Trama recuperada post-Hamming']}\n"
                f"Datos recuperados:            {resumen['Datos recuperados']}\n"
                f"Residuo CRC Rx:               {resumen['Residuo CRC Rx']}",
                language="text",
            )

            st.subheader("Codificación Hamming en el transmisor")
            st.dataframe(resultado["tabla_hamming_tx"], use_container_width=True, hide_index=True)

            st.subheader("Decodificación Hamming en el receptor")
            st.dataframe(resultado["tabla_hamming_rx"], use_container_width=True, hide_index=True)

            st.subheader("Verificación CRC en el receptor")
            st.dataframe(resultado["pasos_crc_rx"], use_container_width=True, hide_index=True)

            if resumen["CRC válido"]:
                st.success("El CRC acepta la trama recuperada. No se detectan errores remanentes.")
            else:
                st.error("El CRC rechaza la trama. Se detectan errores remanentes después de Hamming.")

            st.session_state["g6_ultimo_resultado"] = resultado

    with tabs[3]:
        st.header("Señal codificada en el tiempo")

        st.markdown(
            """
Esta sección muestra cómo el ruido afecta la trama codificada con Hamming. Esta
visualización conecta la integración final con las primeras guías sobre señal, ruido
y decisión por umbral.
"""
        )

        if "g6_ultimo_resultado" not in st.session_state:
            st.info("Ejecute primero el flujo completo en la pestaña anterior.")
        else:
            resultado = st.session_state["g6_ultimo_resultado"]
            resumen = resultado["resumen"]

            simbolos = resultado["simbolos"]
            ruido = resultado["ruido"]
            recibido_analogico = resultado["recibido_analogico"]

            df_senal = pd.DataFrame(
                {
                    "Posición": np.arange(1, len(simbolos) + 1),
                    "Símbolo Tx": simbolos,
                    "Valor recibido": recibido_analogico,
                    "Umbral": np.zeros(len(simbolos)),
                    "Ruido": ruido,
                }
            ).set_index("Posición")

            st.subheader("Señal transmitida vs señal recibida")
            st.line_chart(df_senal[["Símbolo Tx", "Valor recibido", "Umbral"]])

            st.subheader("Ruido en el tiempo")
            st.line_chart(df_senal[["Ruido"]])

            st.subheader("Interpretación")

            st.markdown(
                f"""
- La trama Hamming transmitida tiene {len(resumen["Hamming Tx"])} bits codificados.
- El canal produjo {resumen["Errores canal codificado"]} errores sobre la trama codificada.
- Después de Hamming quedaron {resumen["Errores post-Hamming"]} errores en la trama con CRC.
- El CRC final determinó el estado: **{resumen["Estado final"]}**.
"""
            )

    with tabs[4]:
        st.header("Comparación de escenarios")

        st.markdown(
            """
En esta sección se compara el desempeño del sistema bajo cuatro escenarios:

1. Sin protección.
2. CRC solo.
3. Hamming solo.
4. Hamming + CRC.

Esto permite observar la mejora producida por la corrección y la detección de errores.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits = st.selectbox(
                "Cantidad de bits útiles",
                [100, 1000, 10000, 50000],
                index=2,
                key="g6_bits_comparacion",
            )

            generador_comp = st.text_input(
                "Generador CRC",
                value="1011",
                key="g6_generador_comp",
            ).strip()

            sigma_comp = st.slider(
                "Desviación estándar σ",
                min_value=0.0,
                max_value=2.0,
                value=0.50,
                step=0.05,
                key="g6_sigma_comp",
            )

            semilla_comp = st.number_input(
                "Semilla",
                min_value=0,
                max_value=999999,
                value=900,
                step=1,
                key="g6_semilla_comp",
            )

            ejecutar_comp = st.button("Comparar escenarios", use_container_width=True)

        with col_info:
            st.info(
                """
Esta comparación permite responder una pregunta central del proyecto: ¿cómo mejora
el desempeño cuando se agregan mecanismos de detección y corrección de errores?
"""
            )

        if not validar_generador(generador_comp):
            st.error("El generador CRC debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar_comp:
            datos_comp = generar_bits_aleatorios(cantidad_bits, int(semilla_comp))

            df_comp = comparar_escenarios(
                datos=datos_comp,
                generador=generador_comp,
                sigma=sigma_comp,
                semilla=int(semilla_comp),
            )

            st.subheader("Tabla comparativa")
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.subheader("BER final por escenario")
            st.bar_chart(df_comp[["Escenario", "BER final"]].set_index("Escenario"))

            st.subheader("Bits transmitidos por escenario")
            st.bar_chart(df_comp[["Escenario", "Bits transmitidos"]].set_index("Escenario"))

            st.markdown(
                """
**Lectura esperada**

- Sin protección, los errores del canal llegan directamente a los datos.
- CRC solo detecta errores, pero no los corrige.
- Hamming reduce errores cuando estos son corregibles por bloque.
- Hamming + CRC combina corrección y verificación final.
- La mejora en BER debe interpretarse junto con el costo de redundancia.
"""
            )

    with tabs[5]:
        st.header("Estadística final del sistema completo")

        st.markdown(
            """
En esta sección se evalúa el sistema Hamming + CRC bajo diferentes niveles de ruido.
El objetivo es observar cómo varían el BER final, el estado de aceptación/rechazo y el
comportamiento del sistema al aumentar σ.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            bits_est = st.selectbox(
                "Cantidad de bits útiles",
                [1000, 10000, 50000],
                index=1,
                key="g6_bits_est",
            )

            generador_est = st.text_input(
                "Generador CRC",
                value="1011",
                key="g6_generador_est",
            ).strip()

            sigmas_texto = st.text_input(
                "Valores de σ separados por coma",
                value="0.10, 0.30, 0.50, 0.80, 1.00",
                key="g6_sigmas_est",
            )

            semilla_est = st.number_input(
                "Semilla base",
                min_value=0,
                max_value=999999,
                value=1200,
                step=1,
                key="g6_semilla_est",
            )

            ejecutar_est = st.button("Ejecutar estadística final", use_container_width=True)

        with col_info:
            st.info(
                """
Este análisis resume el desempeño final del sistema completo ante distintos niveles de
ruido. Es la evidencia estadística principal de la guía.
"""
            )

        if not validar_generador(generador_est):
            st.error("El generador CRC debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar_est:
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

            df_est = comparar_sigmas_sistema_completo(
                cantidad_bits=bits_est,
                generador=generador_est,
                valores_sigma=valores_sigma,
                semilla=int(semilla_est),
            )

            st.subheader("Tabla estadística del sistema completo")
            st.dataframe(df_est, use_container_width=True, hide_index=True)

            st.subheader("BER final vs σ")
            st.line_chart(df_est[["σ", "BER final"]].set_index("σ"))

            st.subheader("Bits transmitidos vs σ")
            st.line_chart(df_est[["σ", "Bits transmitidos"]].set_index("σ"))

            st.subheader("Estados finales")
            conteo = df_est["Estado"].value_counts().reset_index()
            conteo.columns = ["Estado", "Cantidad"]
            st.bar_chart(conteo.set_index("Estado"))

            st.markdown(
                """
**Interpretación esperada**

- Al aumentar σ, el canal introduce más errores.
- Hamming puede corregir errores de un bit por bloque, pero no todos los patrones.
- CRC permite rechazar tramas con errores remanentes.
- La confiabilidad del sistema debe evaluarse considerando BER final, aceptación/rechazo
  y costo de redundancia.
"""
            )

    with tabs[6]:
        st.header("Dinámica de aprendizaje")

        st.markdown(
            """
Realice las siguientes actividades:

1. Ejecute el flujo completo con un mensaje corto y σ = 0.10.
2. Repita con σ = 0.80.
3. Observe el BER del canal codificado y el BER post-Hamming.
4. Verifique si el CRC acepta o rechaza la trama.
5. Compare los escenarios sin protección, CRC solo, Hamming solo y Hamming + CRC.
6. Ejecute la estadística final para varios valores de σ.
7. Explique en qué casos Hamming mejora el desempeño.
8. Explique por qué CRC es necesario aunque Hamming corrija errores.
"""
        )

        pregunta_1 = st.radio(
            "Pregunta 1: ¿Cuál es la función de Hamming en el sistema completo?",
            [
                "Detectar errores remanentes sin corregirlos.",
                "Corregir errores de un bit por bloque.",
                "Eliminar el ruido gaussiano.",
                "Reducir la potencia de la señal.",
            ],
            index=None,
            key="g6_pregunta_1",
        )

        if pregunta_1:
            if pregunta_1 == "Corregir errores de un bit por bloque.":
                st.success("Correcto. Hamming permite corregir errores simples por bloque.")
            else:
                st.error("Revise la función del código Hamming en el receptor.")

        pregunta_2 = st.radio(
            "Pregunta 2: ¿Cuál es la función de CRC al final del receptor?",
            [
                "Aumentar la desviación estándar del ruido.",
                "Detectar errores remanentes después de la corrección.",
                "Convertir bits en símbolos BPSK.",
                "Calcular la matriz generadora.",
            ],
            index=None,
            key="g6_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "Detectar errores remanentes después de la corrección.":
                st.success("Correcto. CRC verifica si la información recuperada conserva integridad.")
            else:
                st.error("Revise el papel del CRC en el flujo completo.")

        pregunta_3 = st.radio(
            "Pregunta 3: ¿Por qué se comparan escenarios?",
            [
                "Para ocultar los errores del canal.",
                "Para evaluar cuantitativamente la mejora producida por la redundancia.",
                "Para evitar usar BER.",
                "Para eliminar la necesidad de simulación.",
            ],
            index=None,
            key="g6_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "Para evaluar cuantitativamente la mejora producida por la redundancia.":
                st.success("Correcto. La comparación permite medir el impacto de Hamming y CRC.")
            else:
                st.error("Revise la importancia de las estadísticas comparativas.")

    with tabs[7]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- el sistema completo combina detección y corrección de errores;
- CRC agrega información de verificación;
- Hamming agrega redundancia para corregir errores simples;
- el canal con ruido altera la trama codificada;
- Hamming reduce errores cuando estos son corregibles;
- CRC detecta errores remanentes que no fueron corregidos;
- la comparación estadística permite evaluar la mejora real del sistema;
- mayor confiabilidad implica mayor redundancia y mayor cantidad de bits transmitidos.
"""
        )

    with tabs[8]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2012). *Data communications and networking* (5th ed.). McGraw-Hill.

Proakis, J. G., & Salehi, M. (2008). *Digital communications* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Lin, S., & Costello, D. J. (1983). *Error control coding: Fundamentals and applications*. Prentice-Hall.
"""
        )