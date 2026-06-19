import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Validaciones generales
# ============================================================

def validar_bits(bits: str) -> bool:
    """
    Valida que una cadena contenga únicamente bits 0 y 1.
    """
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_generador(generador: str) -> bool:
    """
    Valida un generador CRC binario.

    Un generador CRC válido debe:
    - tener al menos 2 bits;
    - contener únicamente 0 y 1;
    - iniciar en 1;
    - terminar en 1.
    """
    return (
        len(generador) >= 2
        and all(bit in "01" for bit in generador)
        and generador[0] == "1"
        and generador[-1] == "1"
    )


def limpiar_bits(bits: str) -> str:
    """
    Limpia espacios y saltos de línea de una secuencia binaria.
    """
    return bits.strip().replace(" ", "").replace("\n", "").replace("\t", "")


def generar_bits_aleatorios(cantidad: int, semilla: int | None = None) -> str:
    """
    Genera una cadena binaria aleatoria reproducible mediante semilla.
    """
    rng = np.random.default_rng(semilla)
    bits = rng.integers(0, 2, size=cantidad)
    return "".join(str(bit) for bit in bits)


def contar_errores_bits(tx: str, rx: str) -> int:
    """
    Cuenta errores entre dos cadenas binarias comparando hasta la longitud común.
    """
    longitud = min(len(tx), len(rx))

    if longitud == 0:
        return 0

    return sum(1 for a, b in zip(tx[:longitud], rx[:longitud]) if a != b)


def calcular_ber(tx: str, rx: str) -> float:
    """
    Calcula BER entre dos secuencias binarias.
    """
    longitud = min(len(tx), len(rx))

    if longitud == 0:
        return 0.0

    return contar_errores_bits(tx, rx) / longitud


def dividir_en_bloques(bits: str, tamano: int) -> List[str]:
    """
    Divide una secuencia en bloques de tamaño fijo.
    """
    return [bits[i:i + tamano] for i in range(0, len(bits), tamano)]


def rellenar_a_multiplo(bits: str, multiplo: int) -> Tuple[str, int]:
    """
    Rellena con ceros hasta que la longitud sea múltiplo del valor indicado.

    Devuelve:
    - bits con relleno;
    - cantidad de ceros agregados.
    """
    residuo = len(bits) % multiplo

    if residuo == 0:
        return bits, 0

    padding = multiplo - residuo
    return bits + ("0" * padding), padding


def quitar_padding(bits: str, padding: int) -> str:
    """
    Elimina bits de relleno agregados al final.
    """
    if padding <= 0:
        return bits

    return bits[:-padding]


# ============================================================
# Modelo de canal AWGN con BPSK
# ============================================================

def bits_a_simbolos_bpsk(bits: str) -> np.ndarray:
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
    Transmite una secuencia binaria por un canal AWGN usando BPSK.

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

    simbolos = bits_a_simbolos_bpsk(bits)
    ruido = rng.normal(loc=0.0, scale=sigma, size=len(simbolos))
    recibido = simbolos + ruido
    bits_rx_array = np.where(recibido >= 0, 1, 0)

    bits_rx = "".join(str(bit) for bit in bits_rx_array)

    return bits_rx, simbolos, ruido, recibido


def calcular_potencias(
    simbolos: np.ndarray,
    ruido: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Calcula potencia de señal, potencia de ruido, SNR lineal y SNR en dB.
    """
    if len(simbolos) == 0 or len(ruido) == 0:
        return 0.0, 0.0, math.inf, math.inf

    potencia_senal = float(np.mean(simbolos**2))
    potencia_ruido = float(np.mean(ruido**2))

    if potencia_ruido == 0:
        return potencia_senal, potencia_ruido, math.inf, math.inf

    snr = potencia_senal / potencia_ruido
    snr_db = 10 * math.log10(snr)

    return potencia_senal, potencia_ruido, snr, snr_db


# ============================================================
# Hamming (7,4)
# Estructura usada: [P1 P2 D1 P4 D2 D3 D4]
# ============================================================

def hamming_codificar_bloque_4(bits4: str) -> str:
    """
    Codifica 4 bits usando Hamming (7,4) con paridad par.

    Entrada:
    bits4 = D1 D2 D3 D4

    Salida:
    [P1 P2 D1 P4 D2 D3 D4]
    """
    d1, d2, d3, d4 = [int(bit) for bit in bits4]

    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4

    return f"{p1}{p2}{d1}{p4}{d2}{d3}{d4}"


def hamming_calcular_sindrome(bits7: str) -> Dict[str, int | str]:
    """
    Calcula el síndrome de una palabra Hamming (7,4).

    s1 revisa posiciones 1, 3, 5, 7
    s2 revisa posiciones 2, 3, 6, 7
    s4 revisa posiciones 4, 5, 6, 7

    S = (s4 s2 s1)_2
    """
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


def hamming_invertir_bit(bits: str, posicion: int) -> str:
    """
    Invierte el bit indicado por una posición que inicia en 1.
    """
    lista = list(bits)
    indice = posicion - 1
    lista[indice] = "1" if lista[indice] == "0" else "0"
    return "".join(lista)


def hamming_corregir_bloque_7(bits7: str) -> Tuple[str, str, int, bool]:
    """
    Corrige una palabra Hamming (7,4).

    Devuelve:
    - bloque corregido;
    - síndrome;
    - posición detectada;
    - si aplicó corrección.
    """
    sindrome = hamming_calcular_sindrome(bits7)
    posicion = int(sindrome["posicion_error"])

    if posicion == 0:
        return bits7, str(sindrome["sindrome"]), posicion, False

    return hamming_invertir_bit(bits7, posicion), str(sindrome["sindrome"]), posicion, True


def hamming_extraer_datos(bits7: str) -> str:
    """
    Extrae D1 D2 D3 D4 desde [P1 P2 D1 P4 D2 D3 D4].
    """
    return bits7[2] + bits7[4] + bits7[5] + bits7[6]


def hamming_codificar_stream(bits: str) -> Tuple[str, int, pd.DataFrame]:
    """
    Codifica una secuencia binaria completa usando Hamming (7,4) por bloques.

    Si la longitud no es múltiplo de 4, se agregan ceros de relleno.
    """
    bits_rellenados, padding = rellenar_a_multiplo(bits, 4)
    bloques = dividir_en_bloques(bits_rellenados, 4)

    codificados = []
    filas = []

    for i, bloque in enumerate(bloques, start=1):
        codigo = hamming_codificar_bloque_4(bloque)
        codificados.append(codigo)

        if i <= 25:
            filas.append(
                {
                    "Bloque": i,
                    "Datos": bloque,
                    "Hamming (7,4)": codigo,
                }
            )

    return "".join(codificados), padding, pd.DataFrame(filas)


def hamming_decodificar_stream(bits_codificados: str) -> Tuple[str, str, pd.DataFrame]:
    """
    Decodifica una secuencia formada por palabras Hamming de 7 bits.

    Devuelve:
    - datos decodificados con posible padding;
    - bloques corregidos concatenados;
    - tabla de primeros bloques.
    """
    if len(bits_codificados) % 7 != 0:
        raise ValueError("La secuencia Hamming recibida debe tener longitud múltiplo de 7.")

    bloques = dividir_en_bloques(bits_codificados, 7)

    datos = []
    corregidos = []
    filas = []

    for i, bloque in enumerate(bloques, start=1):
        corregido, sindrome, posicion, corrigio = hamming_corregir_bloque_7(bloque)
        datos_extraidos = hamming_extraer_datos(corregido)

        datos.append(datos_extraidos)
        corregidos.append(corregido)

        if i <= 25:
            filas.append(
                {
                    "Bloque": i,
                    "Recibido": bloque,
                    "Síndrome": sindrome,
                    "Posición detectada": posicion,
                    "Corrigió": "Sí" if corrigio else "No",
                    "Corregido": corregido,
                    "Datos recuperados": datos_extraidos,
                }
            )

    return "".join(datos), "".join(corregidos), pd.DataFrame(filas)


# ============================================================
# CRC
# ============================================================

def binario_a_polinomio(bits: str, variable: str = "x") -> str:
    """
    Convierte una cadena binaria en representación polinómica.

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


def crc_division_modulo_2(dividendo: str, divisor: str) -> Tuple[str, pd.DataFrame]:
    """
    División módulo 2 usada por CRC.

    La resta en módulo 2 se implementa mediante XOR.
    """
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
            operacion = f"{segmento_antes} XOR {divisor}"
        else:
            segmento_despues = segmento_antes
            operacion = "No se aplica XOR porque el bit líder es 0"

        if len(pasos) < 30:
            pasos.append(
                {
                    "Paso": len(pasos) + 1,
                    "Posición": i + 1,
                    "Segmento antes": segmento_antes,
                    "Operación": operacion,
                    "Segmento después": segmento_despues,
                    "Trama temporal": "".join(trabajo),
                }
            )

    residuo = "".join(trabajo[-(n - 1):])
    return residuo, pd.DataFrame(pasos)


def crc_generar(datos: str, generador: str) -> Tuple[str, str, pd.DataFrame]:
    """
    Genera CRC.

    R(x) = M(x)x^r mod G(x)
    T(x) = M(x)x^r + R(x)

    En binario:
    t = m || R
    """
    r = len(generador) - 1
    dividendo = datos + ("0" * r)
    residuo, pasos = crc_division_modulo_2(dividendo, generador)
    trama = datos + residuo

    return residuo, trama, pasos


def crc_verificar(trama: str, generador: str) -> Tuple[bool, str, pd.DataFrame]:
    """
    Verifica una trama con CRC.

    Residuo cero:
    - no se detecta error.

    Residuo distinto de cero:
    - se detecta error.
    """
    residuo, pasos = crc_division_modulo_2(trama, generador)
    valido = all(bit == "0" for bit in residuo)

    return valido, residuo, pasos


def crc_dividir_en_tramas(datos: str, tamano_payload: int) -> Tuple[List[str], int]:
    """
    Divide datos en tramas de tamaño fijo.

    Si la última trama queda incompleta, se rellena con ceros.
    """
    datos_rellenados, padding = rellenar_a_multiplo(datos, tamano_payload)
    tramas = dividir_en_bloques(datos_rellenados, tamano_payload)

    return tramas, padding


def crc_codificar_stream(
    datos: str,
    tamano_payload: int,
    generador: str,
) -> Tuple[str, int, int, pd.DataFrame]:
    """
    Codifica una secuencia de datos en varias tramas con CRC.

    Devuelve:
    - flujo completo con CRC;
    - padding aplicado a los datos;
    - longitud de cada trama codificada;
    - tabla de primeras tramas.
    """
    tramas, padding = crc_dividir_en_tramas(datos, tamano_payload)
    flujo = []
    filas = []

    r = len(generador) - 1
    longitud_trama = tamano_payload + r

    for i, payload in enumerate(tramas, start=1):
        residuo, trama_crc, _ = crc_generar(payload, generador)
        flujo.append(trama_crc)

        if i <= 25:
            filas.append(
                {
                    "Trama": i,
                    "Datos": payload,
                    "CRC": residuo,
                    "Trama con CRC": trama_crc,
                }
            )

    return "".join(flujo), padding, longitud_trama, pd.DataFrame(filas)


def crc_decodificar_stream(
    flujo_crc_rx: str,
    tamano_payload: int,
    generador: str,
    longitud_original_datos: int,
) -> Tuple[str, Dict[str, int | float], pd.DataFrame]:
    """
    Verifica un flujo formado por tramas CRC concatenadas.

    Devuelve:
    - datos recuperados sin padding;
    - métricas CRC;
    - tabla de primeras tramas.
    """
    r = len(generador) - 1
    longitud_trama = tamano_payload + r

    tramas = dividir_en_bloques(flujo_crc_rx, longitud_trama)

    datos_recuperados = []
    filas = []

    tramas_evaluadas = 0
    tramas_detectadas = 0
    tramas_validas = 0

    for i, trama in enumerate(tramas, start=1):
        if len(trama) != longitud_trama:
            continue

        tramas_evaluadas += 1

        valido, residuo_rx, _ = crc_verificar(trama, generador)

        if valido:
            tramas_validas += 1
        else:
            tramas_detectadas += 1

        payload_rx = trama[:tamano_payload]
        datos_recuperados.append(payload_rx)

        if i <= 25:
            filas.append(
                {
                    "Trama": i,
                    "Trama Rx": trama,
                    "Payload Rx": payload_rx,
                    "Residuo verificación": residuo_rx,
                    "CRC válido": "Sí" if valido else "No",
                    "Estado": "No se detecta error" if valido else "Error detectado",
                }
            )

    datos_padded_rx = "".join(datos_recuperados)
    datos_rx = datos_padded_rx[:longitud_original_datos]

    metricas = {
        "Tramas CRC evaluadas": tramas_evaluadas,
        "Tramas válidas CRC": tramas_validas,
        "Tramas detectadas por CRC": tramas_detectadas,
        "Tasa de detección sobre tramas evaluadas": (
            tramas_detectadas / tramas_evaluadas if tramas_evaluadas else 0.0
        ),
    }

    return datos_rx, metricas, pd.DataFrame(filas)


# ============================================================
# Sistema completo
# ============================================================

def simular_sin_proteccion(
    datos: str,
    sigma: float,
    semilla: int,
) -> Tuple[Dict[str, float | int | str], Dict[str, object]]:
    """
    Escenario 1: transmisión sin protección.
    """
    tx = datos

    rx, simbolos, ruido, recibido = transmitir_awgn(tx, sigma=sigma, semilla=semilla)

    ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

    ber_canal = calcular_ber(tx, rx)
    ber_final = calcular_ber(datos, rx[:len(datos)])

    resumen = {
        "Escenario": "Sin protección",
        "Bits de datos originales": len(datos),
        "Bits transmitidos por canal": len(tx),
        "Bits de redundancia": 0,
        "Factor de expansión": len(tx) / len(datos) if len(datos) else 0,
        "BER canal": ber_canal,
        "BER post-Hamming": np.nan,
        "BER final datos": ber_final,
        "Errores finales": contar_errores_bits(datos, rx[:len(datos)]),
        "Tramas detectadas por CRC": 0,
        "SNR dB": snr_db,
        "σ": sigma,
        "σ²": sigma**2,
    }

    detalle = {
        "tx": tx,
        "rx": rx,
        "simbolos": simbolos,
        "ruido": ruido,
        "recibido": recibido,
        "datos_recuperados": rx[:len(datos)],
    }

    return resumen, detalle


def simular_crc_solo(
    datos: str,
    tamano_payload: int,
    generador: str,
    sigma: float,
    semilla: int,
) -> Tuple[Dict[str, float | int | str], Dict[str, object]]:
    """
    Escenario 2: CRC sin Hamming.

    CRC detecta, pero no corrige.
    """
    flujo_crc_tx, padding_crc, longitud_trama_crc, tabla_tx = crc_codificar_stream(
        datos,
        tamano_payload=tamano_payload,
        generador=generador,
    )

    flujo_crc_rx, simbolos, ruido, recibido = transmitir_awgn(
        flujo_crc_tx,
        sigma=sigma,
        semilla=semilla,
    )

    datos_rx, metricas_crc, tabla_rx = crc_decodificar_stream(
        flujo_crc_rx,
        tamano_payload=tamano_payload,
        generador=generador,
        longitud_original_datos=len(datos),
    )

    ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

    ber_canal = calcular_ber(flujo_crc_tx, flujo_crc_rx)
    ber_final = calcular_ber(datos, datos_rx)

    resumen = {
        "Escenario": "CRC solo",
        "Bits de datos originales": len(datos),
        "Bits transmitidos por canal": len(flujo_crc_tx),
        "Bits de redundancia": len(flujo_crc_tx) - len(datos),
        "Factor de expansión": len(flujo_crc_tx) / len(datos) if len(datos) else 0,
        "BER canal": ber_canal,
        "BER post-Hamming": np.nan,
        "BER final datos": ber_final,
        "Errores finales": contar_errores_bits(datos, datos_rx),
        "Tramas detectadas por CRC": int(metricas_crc["Tramas detectadas por CRC"]),
        "SNR dB": snr_db,
        "σ": sigma,
        "σ²": sigma**2,
    }

    detalle = {
        "tx": flujo_crc_tx,
        "rx": flujo_crc_rx,
        "simbolos": simbolos,
        "ruido": ruido,
        "recibido": recibido,
        "datos_recuperados": datos_rx,
        "tabla_crc_tx": tabla_tx,
        "tabla_crc_rx": tabla_rx,
        "metricas_crc": metricas_crc,
        "padding_crc": padding_crc,
        "longitud_trama_crc": longitud_trama_crc,
    }

    return resumen, detalle


def simular_hamming_solo(
    datos: str,
    sigma: float,
    semilla: int,
) -> Tuple[Dict[str, float | int | str], Dict[str, object]]:
    """
    Escenario 3: Hamming sin CRC.

    Hamming corrige un error por bloque, pero no verifica con CRC.
    """
    hamming_tx, padding_hamming, tabla_hamming_tx = hamming_codificar_stream(datos)

    hamming_rx, simbolos, ruido, recibido = transmitir_awgn(
        hamming_tx,
        sigma=sigma,
        semilla=semilla,
    )

    datos_decodificados_padded, hamming_corregido, tabla_hamming_rx = hamming_decodificar_stream(
        hamming_rx
    )

    datos_decodificados = quitar_padding(datos_decodificados_padded, padding_hamming)
    datos_rx = datos_decodificados[:len(datos)]

    ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

    ber_canal = calcular_ber(hamming_tx, hamming_rx)
    ber_post_hamming = calcular_ber(hamming_tx, hamming_corregido)
    ber_final = calcular_ber(datos, datos_rx)

    resumen = {
        "Escenario": "Hamming solo",
        "Bits de datos originales": len(datos),
        "Bits transmitidos por canal": len(hamming_tx),
        "Bits de redundancia": len(hamming_tx) - len(datos),
        "Factor de expansión": len(hamming_tx) / len(datos) if len(datos) else 0,
        "BER canal": ber_canal,
        "BER post-Hamming": ber_post_hamming,
        "BER final datos": ber_final,
        "Errores finales": contar_errores_bits(datos, datos_rx),
        "Tramas detectadas por CRC": 0,
        "SNR dB": snr_db,
        "σ": sigma,
        "σ²": sigma**2,
    }

    detalle = {
        "tx": hamming_tx,
        "rx": hamming_rx,
        "simbolos": simbolos,
        "ruido": ruido,
        "recibido": recibido,
        "hamming_corregido": hamming_corregido,
        "datos_recuperados": datos_rx,
        "tabla_hamming_tx": tabla_hamming_tx,
        "tabla_hamming_rx": tabla_hamming_rx,
        "padding_hamming": padding_hamming,
    }

    return resumen, detalle


def simular_hamming_crc(
    datos: str,
    tamano_payload: int,
    generador: str,
    sigma: float,
    semilla: int,
) -> Tuple[Dict[str, float | int | str], Dict[str, object]]:
    """
    Escenario 4: sistema integrado CRC + Hamming.

    Flujo:
    datos -> CRC -> Hamming -> canal -> Hamming Rx -> CRC Rx
    """
    flujo_crc_tx, padding_crc, longitud_trama_crc, tabla_crc_tx = crc_codificar_stream(
        datos,
        tamano_payload=tamano_payload,
        generador=generador,
    )

    hamming_tx, padding_hamming, tabla_hamming_tx = hamming_codificar_stream(flujo_crc_tx)

    hamming_rx, simbolos, ruido, recibido = transmitir_awgn(
        hamming_tx,
        sigma=sigma,
        semilla=semilla,
    )

    crc_decodificado_padded, hamming_corregido, tabla_hamming_rx = hamming_decodificar_stream(
        hamming_rx
    )

    flujo_crc_rx = quitar_padding(crc_decodificado_padded, padding_hamming)
    flujo_crc_rx = flujo_crc_rx[:len(flujo_crc_tx)]

    datos_rx, metricas_crc, tabla_crc_rx = crc_decodificar_stream(
        flujo_crc_rx,
        tamano_payload=tamano_payload,
        generador=generador,
        longitud_original_datos=len(datos),
    )

    ps, pn, snr, snr_db = calcular_potencias(simbolos, ruido)

    ber_canal = calcular_ber(hamming_tx, hamming_rx)
    ber_post_hamming = calcular_ber(flujo_crc_tx, flujo_crc_rx)
    ber_final = calcular_ber(datos, datos_rx)

    resumen = {
        "Escenario": "Hamming + CRC",
        "Bits de datos originales": len(datos),
        "Bits transmitidos por canal": len(hamming_tx),
        "Bits de redundancia": len(hamming_tx) - len(datos),
        "Factor de expansión": len(hamming_tx) / len(datos) if len(datos) else 0,
        "BER canal": ber_canal,
        "BER post-Hamming": ber_post_hamming,
        "BER final datos": ber_final,
        "Errores finales": contar_errores_bits(datos, datos_rx),
        "Tramas detectadas por CRC": int(metricas_crc["Tramas detectadas por CRC"]),
        "SNR dB": snr_db,
        "σ": sigma,
        "σ²": sigma**2,
    }

    detalle = {
        "tx": hamming_tx,
        "rx": hamming_rx,
        "simbolos": simbolos,
        "ruido": ruido,
        "recibido": recibido,
        "hamming_corregido": hamming_corregido,
        "flujo_crc_tx": flujo_crc_tx,
        "flujo_crc_rx": flujo_crc_rx,
        "datos_recuperados": datos_rx,
        "tabla_crc_tx": tabla_crc_tx,
        "tabla_crc_rx": tabla_crc_rx,
        "tabla_hamming_tx": tabla_hamming_tx,
        "tabla_hamming_rx": tabla_hamming_rx,
        "metricas_crc": metricas_crc,
        "padding_crc": padding_crc,
        "padding_hamming": padding_hamming,
        "longitud_trama_crc": longitud_trama_crc,
    }

    return resumen, detalle


def simular_todos_los_escenarios(
    datos: str,
    tamano_payload: int,
    generador: str,
    sigma: float,
    semilla: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Ejecuta los cuatro escenarios:

    1. Sin protección
    2. CRC solo
    3. Hamming solo
    4. Hamming + CRC
    """
    resultados = []
    detalles = {}

    resumen_sin, detalle_sin = simular_sin_proteccion(datos, sigma, semilla + 1)
    resultados.append(resumen_sin)
    detalles["Sin protección"] = detalle_sin

    resumen_crc, detalle_crc = simular_crc_solo(
        datos,
        tamano_payload=tamano_payload,
        generador=generador,
        sigma=sigma,
        semilla=semilla + 2,
    )
    resultados.append(resumen_crc)
    detalles["CRC solo"] = detalle_crc

    resumen_hamming, detalle_hamming = simular_hamming_solo(
        datos,
        sigma=sigma,
        semilla=semilla + 3,
    )
    resultados.append(resumen_hamming)
    detalles["Hamming solo"] = detalle_hamming

    resumen_integrado, detalle_integrado = simular_hamming_crc(
        datos,
        tamano_payload=tamano_payload,
        generador=generador,
        sigma=sigma,
        semilla=semilla + 4,
    )
    resultados.append(resumen_integrado)
    detalles["Hamming + CRC"] = detalle_integrado

    df = pd.DataFrame(resultados)

    return df, detalles


def comparar_por_sigma(
    cantidad_bits: int,
    tamano_payload: int,
    generador: str,
    valores_sigma: List[float],
    semilla: int,
) -> pd.DataFrame:
    """
    Ejecuta el sistema completo Hamming + CRC para varios valores de sigma.
    """
    datos = generar_bits_aleatorios(cantidad_bits, semilla=semilla)

    filas = []

    for i, sigma in enumerate(valores_sigma):
        resumen, _ = simular_hamming_crc(
            datos,
            tamano_payload=tamano_payload,
            generador=generador,
            sigma=sigma,
            semilla=semilla + (1000 * (i + 1)),
        )

        filas.append(resumen)

    return pd.DataFrame(filas)


# ============================================================
# Tablas teóricas
# ============================================================

def construir_tabla_flujo_sistema() -> pd.DataFrame:
    """
    Tabla del flujo completo del sistema integrado.
    """
    return pd.DataFrame(
        {
            "Etapa": [
                "1. Datos originales",
                "2. CRC en transmisor",
                "3. Hamming en transmisor",
                "4. Canal con ruido",
                "5. Hamming en receptor",
                "6. CRC en receptor",
                "7. Datos finales",
            ],
            "Operación": [
                "Se genera o ingresa la secuencia de bits",
                "Se calcula R(x) y se forma t = m || R",
                "Se agregan bits de paridad Hamming por bloques de 4 bits",
                "Se transmite usando BPSK y ruido AWGN",
                "Se calcula síndrome y se corrige un error por bloque",
                "Se verifica el residuo CRC de cada trama",
                "Se comparan los datos recuperados con los originales",
            ],
            "Propósito": [
                "Definir la información útil",
                "Agregar detección de errores",
                "Agregar corrección de errores simples",
                "Modelar errores de transmisión",
                "Reducir errores antes de verificar CRC",
                "Detectar errores remanentes",
                "Medir desempeño final",
            ],
        }
    )


def construir_tabla_comparacion_teorica() -> pd.DataFrame:
    """
    Tabla conceptual de escenarios.
    """
    return pd.DataFrame(
        {
            "Escenario": [
                "Sin protección",
                "CRC solo",
                "Hamming solo",
                "Hamming + CRC",
            ],
            "Qué agrega": [
                "Nada",
                "Residuo CRC",
                "Bits de paridad Hamming",
                "CRC + paridad Hamming",
            ],
            "Qué puede hacer": [
                "No detecta ni corrige",
                "Detecta errores",
                "Corrige un error por bloque",
                "Corrige errores simples y detecta remanentes",
            ],
            "Limitación": [
                "Los errores llegan directamente a los datos",
                "No corrige",
                "Puede fallar ante errores múltiples",
                "Aumenta la redundancia y no garantiza detección absoluta",
            ],
        }
    )


def construir_tabla_metricas() -> pd.DataFrame:
    """
    Tabla de métricas usadas en la Guía 6.
    """
    return pd.DataFrame(
        {
            "Métrica": [
                "BER canal",
                "BER post-Hamming",
                "BER final datos",
                "SNR",
                "Factor de expansión",
                "Tramas detectadas por CRC",
            ],
            "Qué mide": [
                "Errores producidos directamente por el canal",
                "Errores que quedan después de corrección Hamming",
                "Errores finales en los datos útiles recuperados",
                "Relación entre potencia de señal y potencia de ruido",
                "Cuántos bits se transmiten respecto a los datos originales",
                "Cantidad de tramas marcadas como alteradas por CRC",
            ],
            "Interpretación": [
                "Mide el daño del canal",
                "Mide la mejora por Hamming",
                "Mide la calidad final para el usuario",
                "A mayor SNR, se espera menor BER",
                "Mayor protección implica mayor redundancia",
                "Mayor detección no significa corrección",
            ],
        }
    )


# ============================================================
# Gráficas
# ============================================================

def graficar_muestras_discretas(detalle: Dict[str, object], max_muestras: int = 40) -> None:
    """
    Grafica bits/símbolos y valores recibidos para una cantidad limitada de muestras.
    """
    tx = str(detalle["tx"])[:max_muestras]
    rx = str(detalle["rx"])[:max_muestras]

    simbolos = np.array(detalle["simbolos"][:max_muestras])
    ruido = np.array(detalle["ruido"][:max_muestras])
    recibido = np.array(detalle["recibido"][:max_muestras])

    posiciones = np.arange(1, len(tx) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(posiciones, simbolos, basefmt=" ", label="Símbolo transmitido")
    ax.scatter(posiciones, recibido, marker="x", label="Valor recibido")
    ax.axhline(0, linestyle="--", linewidth=1, label="Umbral de decisión")
    ax.set_title("Símbolos transmitidos y valores recibidos por muestra")
    ax.set_xlabel("Índice de muestra")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig_ruido, ax_ruido = plt.subplots(figsize=(10, 3))
    ax_ruido.stem(posiciones, ruido, basefmt=" ")
    ax_ruido.axhline(0, linestyle="--", linewidth=1)
    ax_ruido.set_title("Ruido gaussiano por muestra")
    ax_ruido.set_xlabel("Índice de muestra")
    ax_ruido.set_ylabel("Ruido")
    ax_ruido.grid(True)
    st.pyplot(fig_ruido)
    plt.close(fig_ruido)

    tabla = pd.DataFrame(
        {
            "Muestra": posiciones,
            "Bit Tx": list(tx),
            "Bit Rx": list(rx),
            "Estado": ["Correcto" if a == b else "Error" for a, b in zip(tx, rx)],
        }
    )

    st.dataframe(tabla, use_container_width=True, hide_index=True)


def graficar_comparacion_ber(df: pd.DataFrame) -> None:
    """
    Grafica BER final por escenario en escala logarítmica.

    Si en algún escenario no se observan errores finales, se representa como
    BER < 1/N para evitar graficar cero en escala logarítmica.
    """
    df_plot = df.copy()

    df_plot["BER final datos"] = pd.to_numeric(
        df_plot["BER final datos"],
        errors="coerce",
    )
    df_plot["Bits de datos originales"] = pd.to_numeric(
        df_plot["Bits de datos originales"],
        errors="coerce",
    )

    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
    df_plot = df_plot.dropna(
        subset=["Escenario", "BER final datos", "Bits de datos originales"]
    )
    df_plot = df_plot[df_plot["Bits de datos originales"] > 0]

    if df_plot.empty:
        st.info("No hay datos válidos para graficar BER por escenario.")
        return

    df_plot["Límite experimental 1/N"] = 1 / df_plot["Bits de datos originales"]
    df_plot["BER para gráfica"] = df_plot.apply(
        lambda fila: fila["BER final datos"]
        if fila["BER final datos"] > 0
        else fila["Límite experimental 1/N"],
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df_plot["Escenario"], df_plot["BER para gráfica"])
    ax.set_yscale("log")
    ax.set_title("BER final de datos por escenario")
    ax.set_xlabel("Escenario")
    ax.set_ylabel("BER en escala logarítmica")
    ax.grid(True, which="both", axis="y")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    plt.close(fig)

    if (df_plot["BER final datos"] == 0).any():
        st.caption(
            "Cuando no se observan errores finales, la barra representa el límite experimental BER < 1/N."
        )


def graficar_expansion(df: pd.DataFrame) -> None:
    """
    Grafica factor de expansión por escenario.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["Escenario"], df["Factor de expansión"])
    ax.set_title("Factor de expansión por escenario")
    ax.set_xlabel("Escenario")
    ax.set_ylabel("Bits transmitidos / bits originales")
    ax.grid(True, axis="y")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    plt.close(fig)


def graficar_ber_vs_sigma(df: pd.DataFrame) -> None:
    """
    Grafica BER final vs sigma para el sistema integrado.

    Cada punto representa una simulación independiente para un valor específico de σ.
    El eje vertical se mantiene en escala logarítmica y no se unen los puntos con línea
    continua, para evitar interpretar la gráfica como una medición continua.
    """
    df_plot = df.copy()

    df_plot["BER final datos"] = pd.to_numeric(
        df_plot["BER final datos"],
        errors="coerce",
    )
    df_plot["Bits de datos originales"] = pd.to_numeric(
        df_plot["Bits de datos originales"],
        errors="coerce",
    )
    df_plot["σ"] = pd.to_numeric(
        df_plot["σ"],
        errors="coerce",
    )

    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
    df_plot = df_plot.dropna(subset=["σ", "BER final datos", "Bits de datos originales"])
    df_plot = df_plot[df_plot["Bits de datos originales"] > 0]

    if df_plot.empty:
        st.info("No hay datos válidos para graficar BER vs σ.")
        return

    df_plot = df_plot.sort_values("σ")
    df_plot["Límite experimental 1/N"] = 1 / df_plot["Bits de datos originales"]

    df_con_eventos = df_plot[df_plot["BER final datos"] > 0]
    df_sin_eventos = df_plot[df_plot["BER final datos"] == 0]

    fig, ax = plt.subplots(figsize=(8, 4))

    if not df_con_eventos.empty:
        ax.scatter(
            df_con_eventos["σ"],
            df_con_eventos["BER final datos"],
            marker="o",
            s=85,
            label="BER final observado",
            zorder=3,
        )

    if not df_sin_eventos.empty:
        ax.scatter(
            df_sin_eventos["σ"],
            df_sin_eventos["Límite experimental 1/N"],
            marker="v",
            s=95,
            label="0 errores observados: BER < 1/N",
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_title("Sistema Hamming + CRC: BER final vs σ")
    ax.set_xlabel("Desviación estándar del ruido σ")
    ax.set_ylabel("BER final en escala logarítmica")
    ax.grid(True, which="major", linestyle="-", linewidth=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Cada punto representa una simulación independiente para un valor específico de σ. "
        "El eje vertical está en escala logarítmica para visualizar valores pequeños de BER."
    )


def graficar_ber_vs_snr(df: pd.DataFrame) -> None:
    """
    Grafica BER final vs SNR dB para el sistema integrado.

    Cada punto representa una simulación independiente. El eje vertical es logarítmico,
    pero los puntos no se unen con línea continua.
    """
    df_plot = df.copy()

    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
    df_plot["BER final datos"] = pd.to_numeric(
        df_plot["BER final datos"],
        errors="coerce",
    )
    df_plot["Bits de datos originales"] = pd.to_numeric(
        df_plot["Bits de datos originales"],
        errors="coerce",
    )
    df_plot["SNR dB"] = pd.to_numeric(
        df_plot["SNR dB"],
        errors="coerce",
    )

    df_plot = df_plot.dropna(
        subset=["SNR dB", "BER final datos", "Bits de datos originales"]
    )
    df_plot = df_plot[df_plot["Bits de datos originales"] > 0]

    if df_plot.empty:
        st.info("No hay valores finitos de SNR para graficar.")
        return

    df_plot = df_plot.sort_values("SNR dB")
    df_plot["Límite experimental 1/N"] = 1 / df_plot["Bits de datos originales"]

    df_con_eventos = df_plot[df_plot["BER final datos"] > 0]
    df_sin_eventos = df_plot[df_plot["BER final datos"] == 0]

    fig, ax = plt.subplots(figsize=(8, 4))

    if not df_con_eventos.empty:
        ax.scatter(
            df_con_eventos["SNR dB"],
            df_con_eventos["BER final datos"],
            marker="o",
            s=85,
            label="BER final observado",
            zorder=3,
        )

    if not df_sin_eventos.empty:
        ax.scatter(
            df_sin_eventos["SNR dB"],
            df_sin_eventos["Límite experimental 1/N"],
            marker="v",
            s=95,
            label="0 errores observados: BER < 1/N",
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_title("Sistema Hamming + CRC: BER final vs SNR dB")
    ax.set_xlabel("SNR dB")
    ax.set_ylabel("BER final en escala logarítmica")
    ax.grid(True, which="major", linestyle="-", linewidth=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Cada punto representa una simulación independiente para un valor específico de SNR. "
        "El eje vertical está en escala logarítmica para visualizar valores pequeños de BER."
    )


# ============================================================
# Interfaz Streamlit
# ============================================================

def render_guia_06() -> None:
    st.title("Guía 6: Sistema completo Hamming + CRC")

    st.markdown(
        """
Esta guía integra los elementos estudiados en las guías anteriores: canal con ruido,
BER, SNR, codificación Hamming, cálculo del síndrome y verificación CRC.

El objetivo es analizar un sistema digital completo que no solo transmite bits por un
canal ruidoso, sino que también aplica mecanismos de control de errores para mejorar
la confiabilidad. El flujo general usado en esta guía es:

Datos → CRC → Hamming → Canal con ruido → Hamming Rx → CRC Rx → Datos recuperados

Hamming se usa como mecanismo de corrección de errores simples, mientras que CRC se usa
como mecanismo de detección de errores remanentes. Esta combinación permite estudiar
la relación entre redundancia, corrección, detección y desempeño estadístico del sistema
(Hamming, 1950; Lin & Costello, 2004; Forouzan, 2013; Stallings, 2015).
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Flujo del sistema",
            "Simulación interactiva",
            "Comparación estadística",
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

Integrar Hamming (7,4), CRC y un canal con ruido gaussiano para evaluar el desempeño
de un sistema digital con mecanismos de detección y corrección de errores.

**Objetivos específicos**

1. Comprender el flujo completo de transmisión y recepción.
2. Aplicar CRC para agregar capacidad de detección de errores.
3. Aplicar Hamming (7,4) para corregir errores simples.
4. Simular un canal AWGN usando modulación BPSK normalizada.
5. Calcular BER del canal, BER post-Hamming y BER final.
6. Comparar escenarios con y sin protección.
7. Analizar el costo de redundancia agregado por CRC y Hamming.
8. Interpretar el efecto de la SNR y de la varianza del ruido sobre el BER.
9. Evaluar el sistema con muchos bits para obtener resultados estadísticos.
10. Relacionar los resultados de la app con la teoría de control de errores.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
En comunicaciones digitales, el objetivo no es únicamente enviar bits, sino recuperar
la información con la menor cantidad posible de errores. Cuando una señal digital atraviesa
un canal físico, puede ser afectada por ruido, interferencias y distorsiones. Para estudiar
ese fenómeno de forma controlada, en esta app se usa un canal AWGN, es decir, ruido blanco
gaussiano aditivo (Forouzan, 2013; Proakis & Salehi, 2008).

El modelo de canal utilizado es:

$$
r = s + n
$$

donde:

- $s$ es el símbolo transmitido;
- $n$ es el ruido gaussiano;
- $r$ es el valor recibido.

Para representar bits se usa BPSK normalizado:

$$
0 \\rightarrow -1
$$

$$
1 \\rightarrow +1
$$

El receptor decide usando un umbral en cero:

- si $r \\geq 0$, decide 1;
- si $r < 0$, decide 0.

Este modelo permite observar cómo el ruido puede desplazar una muestra hacia el lado
incorrecto del umbral y producir errores de bit.
"""
        )

        st.subheader("Hamming como corrección de errores")

        st.markdown(
            """
Hamming (7,4) es un código de bloque que toma 4 bits de datos y agrega 3 bits de paridad
para formar una palabra de 7 bits. En esta app se usa la estructura:

[P1 P2 D1 P4 D2 D3 D4]

En el receptor, las comprobaciones de paridad forman un síndrome. Si el síndrome es
distinto de cero, su valor decimal indica la posición del error bajo la hipótesis de
que ocurrió un solo error en el bloque (Hamming, 1950; Lin & Costello, 2004).

Por tanto, Hamming (7,4):

- corrige un error por bloque;
- mejora la confiabilidad cuando los errores son aislados;
- puede fallar ante errores múltiples dentro del mismo bloque.

Esta limitación es importante porque justifica la integración de CRC.
"""
        )

        st.subheader("CRC como detección de errores remanentes")

        st.markdown(
            """
CRC es una técnica de detección de errores basada en división módulo 2. El transmisor
calcula un residuo $R(x)$ a partir de un mensaje $M(x)$ y un generador $G(x)$:

$$
R(x) = M(x)x^r \\bmod G(x)
$$

Luego construye la trama:

$$
T(x) = M(x)x^r + R(x)
$$

En forma binaria:

$$
t = m || R
$$

donde $||$ significa concatenación.

En el receptor, la trama recibida se divide entre el mismo generador. Si el residuo es
distinto de cero, se detecta error. Si el residuo es cero, se dice que no se detecta
error, pero no se garantiza de forma absoluta que no haya existido alteración
(Forouzan, 2013; Stallings, 2015; Tanenbaum & Wetherall, 2011).
"""
        )

        st.subheader("Sistema integrado Hamming + CRC")

        st.markdown(
            """
En esta guía, CRC se aplica primero y Hamming después:

Datos → CRC → Hamming

Esto significa que Hamming protege tanto los datos como los bits CRC. Luego, en el receptor,
se invierte el proceso:

Hamming Rx → CRC Rx

La razón de este orden es la siguiente:

1. CRC agrega una verificación de integridad a la información.
2. Hamming protege la trama resultante frente a errores simples del canal.
3. Al recibir, Hamming intenta corregir errores de un bit por bloque.
4. Después, CRC verifica si todavía quedan errores remanentes.

Esta arquitectura es didácticamente útil porque permite observar dos funciones distintas:

- Hamming corrige.
- CRC detecta.

Ambos mecanismos agregan redundancia, por lo que mejoran la confiabilidad pero aumentan
la cantidad de bits transmitidos (Lin & Costello, 2004; Stallings, 2015).
"""
        )

        st.subheader("Métricas de desempeño")

        st.markdown(
            """
Para evaluar el sistema se usan varias métricas.

La tasa de error de bit, o BER, se define como:

$$
BER = \\frac{\\text{bits erróneos}}{\\text{bits transmitidos}}
$$

La razón señal-ruido se define como:

$$
SNR = \\frac{P_s}{P_n}
$$

donde:

- $P_s$ es la potencia promedio de la señal;
- $P_n$ es la potencia promedio del ruido.

En dB:

$$
SNR_{dB} = 10 \\log_{10}(SNR)
$$

También se analiza el factor de expansión:

$$
\\text{Factor de expansión} =
\\frac{\\text{bits transmitidos por el canal}}{\\text{bits de datos originales}}
$$

Este factor permite observar el costo de redundancia. Un sistema con mayor protección
transmite más bits, pero puede reducir la cantidad de errores finales (Proakis & Salehi,
2008; Lin & Costello, 2004).
"""
        )

        st.dataframe(
            construir_tabla_metricas(),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        st.header("Flujo del sistema integrado")

        st.markdown(
            """
La siguiente tabla resume el flujo completo implementado en la app. Esta guía une los
conceptos de las guías anteriores en una sola cadena de transmisión y recepción.
"""
        )

        st.dataframe(
            construir_tabla_flujo_sistema(),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Comparación conceptual de escenarios")

        st.markdown(
            """
Para comprender el aporte de cada mecanismo, la app compara cuatro escenarios:

1. Sin protección.
2. CRC solo.
3. Hamming solo.
4. Hamming + CRC.

Esta comparación permite observar no solo el BER final, sino también el costo de
redundancia.
"""
        )

        st.dataframe(
            construir_tabla_comparacion_teorica(),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            """
La combinación Hamming + CRC no significa que el sistema sea perfecto. Significa que
se agregan dos niveles de protección: primero corrección de errores simples mediante
Hamming y luego detección de errores remanentes mediante CRC.
"""
        )

    with tabs[3]:
        st.header("Simulación interactiva del sistema completo")

        st.markdown(
            """
En esta sección se ejecuta una simulación completa. El estudiante puede usar un mensaje
manual o generar bits aleatorios. Luego se comparan cuatro escenarios de protección.

Para evitar errores de memoria o tiempos excesivos, las tablas muestran únicamente las
primeras tramas o bloques, aunque las métricas se calculan sobre todos los bits simulados.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            modo = st.radio(
                "Modo de entrada",
                ["Mensaje manual", "Bits aleatorios"],
                key="g6_modo_entrada",
            )

            if modo == "Mensaje manual":
                datos = st.text_area(
                    "Datos binarios",
                    value="1011001110001111",
                    key="g6_datos_manual",
                )
                datos = limpiar_bits(datos)
            else:
                cantidad_bits = st.selectbox(
                    "Cantidad de bits",
                    [32, 100, 1000, 5000, 10000, 50000],
                    index=3,
                    key="g6_cantidad_bits",
                )

                semilla_datos = st.number_input(
                    "Semilla para datos",
                    min_value=0,
                    max_value=999999,
                    value=600,
                    step=1,
                    key="g6_semilla_datos",
                )

                datos = generar_bits_aleatorios(
                    int(cantidad_bits),
                    semilla=int(semilla_datos),
                )

                st.code(f"Primeros bits generados: {datos[:120]}", language="text")

            tamano_payload = st.selectbox(
                "Tamaño de payload CRC por trama",
                [4, 8, 16, 32],
                index=2,
                key="g6_payload",
            )

            generador_crc = st.text_input(
                "Generador CRC",
                value="1011",
                key="g6_generador_crc",
            ).strip()

            sigma = st.slider(
                "Desviación estándar del ruido σ",
                min_value=0.0,
                max_value=2.0,
                value=0.50,
                step=0.05,
                key="g6_sigma",
            )

            semilla_canal = st.number_input(
                "Semilla del canal",
                min_value=0,
                max_value=999999,
                value=900,
                step=1,
                key="g6_semilla_canal",
            )

            ejecutar = st.button("Ejecutar sistema completo")

        with col_info:
            st.info(
                """
Recomendación:

- Use pocos bits para observar tablas y secuencias.
- Use muchos bits para obtener métricas estadísticas más estables.

La semilla permite repetir el mismo experimento.
"""
            )

            st.metric("Varianza del ruido σ²", f"{sigma**2:.4f}")

        if not validar_bits(datos):
            st.error("Los datos deben contener únicamente 0 y 1.")
        elif not validar_generador(generador_crc):
            st.error("El generador CRC debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar:
            df_resultados, detalles = simular_todos_los_escenarios(
                datos=datos,
                tamano_payload=int(tamano_payload),
                generador=generador_crc,
                sigma=float(sigma),
                semilla=int(semilla_canal),
            )

            st.subheader("Tabla comparativa de escenarios")

            columnas = [
                "Escenario",
                "Bits de datos originales",
                "Bits transmitidos por canal",
                "Bits de redundancia",
                "Factor de expansión",
                "BER canal",
                "BER post-Hamming",
                "BER final datos",
                "Errores finales",
                "Tramas detectadas por CRC",
                "SNR dB",
                "σ",
                "σ²",
            ]

            for columna in columnas:
                if columna not in df_resultados.columns:
                    df_resultados[columna] = np.nan

            st.dataframe(
                df_resultados[columnas],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("BER final por escenario")
            graficar_comparacion_ber(df_resultados)

            st.subheader("Costo de redundancia")
            graficar_expansion(df_resultados)

            st.subheader("Detalle del sistema Hamming + CRC")

            detalle_integrado = detalles["Hamming + CRC"]

            st.markdown(
                """
Las siguientes tablas muestran solo las primeras tramas o bloques para facilitar la
lectura. Las métricas anteriores sí se calculan usando toda la secuencia simulada.
"""
            )

            st.markdown("**Primeras tramas CRC generadas en transmisión**")
            st.dataframe(
                detalle_integrado["tabla_crc_tx"],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Primeros bloques Hamming recibidos y corregidos**")
            st.dataframe(
                detalle_integrado["tabla_hamming_rx"],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Primeras tramas verificadas por CRC en recepción**")
            st.dataframe(
                detalle_integrado["tabla_crc_rx"],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Muestras discretas del canal en Hamming + CRC")

            graficar_muestras_discretas(detalle_integrado, max_muestras=40)

            st.session_state["g6_df_resultados"] = df_resultados
            st.session_state["g6_detalles"] = detalles
            st.session_state["g6_datos_originales_resultado"] = datos
            st.session_state["g6_sigma_resultado"] = sigma
            st.session_state["g6_generador_resultado"] = generador_crc
            st.session_state["g6_payload_resultado"] = tamano_payload

    with tabs[4]:
        st.header("Comparación estadística del sistema Hamming + CRC")

        st.markdown(
            """
En esta sección se evalúa únicamente el sistema integrado Hamming + CRC para distintos
valores de ruido. El objetivo es observar cómo cambia el BER final cuando aumenta σ y
cuando cambia la SNR.

Se espera que:

- Al aumentar σ, aumente la potencia de ruido.
- Al aumentar la potencia de ruido, disminuya la SNR.
- Al disminuir la SNR, aumente la probabilidad de error.
- Al aumentar los errores, Hamming y CRC pueden verse más exigidos.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_bits_comp = st.selectbox(
                "Cantidad de bits para comparación",
                [1000, 5000, 10000, 50000],
                index=2,
                key="g6_bits_comp",
            )

            payload_comp = st.selectbox(
                "Payload CRC",
                [4, 8, 16, 32],
                index=2,
                key="g6_payload_comp",
            )

            gen_comp = st.text_input(
                "Generador CRC",
                value="1011",
                key="g6_gen_comp",
            ).strip()

            sigmas_texto = st.text_input(
                "Valores de σ separados por coma",
                value="0.10, 0.30, 0.50, 0.80, 1.00",
                key="g6_sigmas_comp",
            )

            semilla_comp = st.number_input(
                "Semilla base",
                min_value=0,
                max_value=999999,
                value=1500,
                step=1,
                key="g6_semilla_comp",
            )

            ejecutar_comp = st.button("Comparar Hamming + CRC contra σ")

        with col_info:
            st.info(
                """
Esta comparación es útil para justificar el uso de gráficas BER vs σ y BER vs SNR dB.

La escala logarítmica ayuda cuando los valores de BER son pequeños.
"""
            )

        if not validar_generador(gen_comp):
            st.error("El generador CRC debe ser válido.")
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

            df_comp = comparar_por_sigma(
                cantidad_bits=int(cantidad_bits_comp),
                tamano_payload=int(payload_comp),
                generador=gen_comp,
                valores_sigma=valores_sigma,
                semilla=int(semilla_comp),
            )

            st.subheader("Tabla de comparación")

            st.dataframe(
                df_comp[
                    [
                        "σ",
                        "σ²",
                        "SNR dB",
                        "BER canal",
                        "BER post-Hamming",
                        "BER final datos",
                        "Errores finales",
                        "Tramas detectadas por CRC",
                        "Bits transmitidos por canal",
                        "Factor de expansión",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("BER final vs σ")
            graficar_ber_vs_sigma(df_comp)

            st.subheader("BER final vs SNR dB")
            graficar_ber_vs_snr(df_comp)

            st.session_state["g6_comparacion_sigma"] = df_comp

    with tabs[5]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección reúne los resultados de la última simulación y plantea preguntas de
interpretación. El objetivo es que el estudiante conecte la teoría con las métricas
obtenidas.
"""
        )

        if "g6_df_resultados" in st.session_state:
            st.subheader("Última comparación de escenarios")

            df_resultados = st.session_state["g6_df_resultados"]

            st.dataframe(df_resultados, use_container_width=True, hide_index=True)

            mejor = df_resultados.loc[df_resultados["BER final datos"].idxmin()]
            mayor_redundancia = df_resultados.loc[df_resultados["Factor de expansión"].idxmax()]

            st.markdown(
                f"""
**Lectura automática**

- El menor BER final observado fue **{mejor["BER final datos"]:.6f}** en el escenario **{mejor["Escenario"]}**.
- El mayor factor de expansión fue **{mayor_redundancia["Factor de expansión"]:.4f}** en el escenario **{mayor_redundancia["Escenario"]}**.

Esto muestra el compromiso central del control de errores: para mejorar confiabilidad,
normalmente se transmite más redundancia.
"""
            )
        else:
            st.info("Ejecute primero una simulación interactiva.")

        if "g6_comparacion_sigma" in st.session_state:
            st.subheader("Última comparación contra σ")

            df_comp = st.session_state["g6_comparacion_sigma"]

            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            menor_sigma = df_comp.loc[df_comp["σ"].idxmin()]
            mayor_sigma = df_comp.loc[df_comp["σ"].idxmax()]

            st.markdown(
                f"""
**Efecto del ruido**

- Con $\\sigma = {menor_sigma["σ"]:.2f}$, el BER final fue **{menor_sigma["BER final datos"]:.6f}**.
- Con $\\sigma = {mayor_sigma["σ"]:.2f}$, el BER final fue **{mayor_sigma["BER final datos"]:.6f}**.

En general, se espera que al aumentar σ aumente la probabilidad de error, aunque los
resultados pueden variar por la semilla y por la cantidad de bits simulados.
"""
            )
        else:
            st.info("Ejecute una comparación estadística contra σ para ver este análisis.")

        st.subheader("Actividades guiadas")

        st.markdown(
            """
Realice las siguientes actividades:

1. Ejecute la simulación con 32 bits y observe las tablas de tramas y bloques.
2. Repita con 10,000 bits y observe cómo las métricas se vuelven más estadísticas.
3. Compare el escenario sin protección con Hamming + CRC.
4. Identifique cuántos bits adicionales se transmiten por usar redundancia.
5. Observe si Hamming reduce el BER después del canal.
6. Observe si CRC detecta tramas alteradas.
7. Cambie σ de 0.10 a 1.00 y observe el cambio en BER.
8. Explique por qué un sistema con más redundancia puede tener menor BER final.
9. Explique por qué CRC no reemplaza a Hamming.
10. Explique por qué Hamming no reemplaza a CRC.
"""
        )

        st.subheader("Preguntas de análisis")

        pregunta_1 = st.radio(
            "Pregunta 1: ¿Cuál es el orden usado en el transmisor del sistema integrado?",
            [
                "Datos → Hamming → CRC → Canal",
                "Datos → CRC → Hamming → Canal",
                "Canal → CRC → Hamming → Datos",
                "Datos → Canal → CRC → Hamming",
            ],
            index=None,
            key="g6_pregunta_1",
        )

        if pregunta_1:
            if pregunta_1 == "Datos → CRC → Hamming → Canal":
                st.success("Correcto. Primero se agrega CRC y luego Hamming protege la trama resultante.")
            else:
                st.error("Revise el flujo del sistema integrado.")

        pregunta_2 = st.radio(
            "Pregunta 2: ¿Qué función cumple Hamming en el sistema integrado?",
            [
                "Detectar únicamente errores remanentes.",
                "Corregir errores simples por bloque.",
                "Eliminar el ruido gaussiano.",
                "Reducir la cantidad de bits transmitidos.",
            ],
            index=None,
            key="g6_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "Corregir errores simples por bloque.":
                st.success("Correcto. Hamming (7,4) corrige un error por bloque.")
            else:
                st.error("Revise el papel de Hamming dentro del sistema.")

        pregunta_3 = st.radio(
            "Pregunta 3: ¿Qué función cumple CRC en el sistema integrado?",
            [
                "Corregir todos los errores múltiples.",
                "Detectar errores remanentes.",
                "Cambiar la modulación BPSK.",
                "Aumentar la SNR física del canal.",
            ],
            index=None,
            key="g6_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "Detectar errores remanentes.":
                st.success("Correcto. CRC verifica si quedan errores después de la corrección.")
            else:
                st.error("Revise el papel de CRC como detector.")

        pregunta_4 = st.radio(
            "Pregunta 4: ¿Qué representa el factor de expansión?",
            [
                "La relación entre bits transmitidos y bits originales.",
                "La cantidad de ruido del canal.",
                "La cantidad de unos del mensaje.",
                "El valor del síndrome.",
            ],
            index=None,
            key="g6_pregunta_4",
        )

        if pregunta_4:
            if pregunta_4 == "La relación entre bits transmitidos y bits originales.":
                st.success("Correcto. Mide el costo de redundancia.")
            else:
                st.error("Revise la definición de factor de expansión.")

        pregunta_5 = st.radio(
            "Pregunta 5: ¿Por qué se usa BER final de datos?",
            [
                "Porque mide los errores que realmente quedan en la información útil recuperada.",
                "Porque siempre es cero.",
                "Porque solo mide los bits CRC.",
                "Porque no depende del canal.",
            ],
            index=None,
            key="g6_pregunta_5",
        )

        if pregunta_5:
            if pregunta_5 == "Porque mide los errores que realmente quedan en la información útil recuperada.":
                st.success("Correcto. Es la métrica más cercana a la calidad final percibida por el usuario.")
            else:
                st.error("Revise la interpretación del BER final.")

    with tabs[6]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- Un sistema digital puede modelarse como una cadena de transmisión y recepción.
- El canal AWGN permite estudiar el efecto del ruido sobre los bits recibidos.
- BPSK permite representar bits como símbolos positivos y negativos.
- Hamming (7,4) agrega bits de paridad para corregir errores simples.
- CRC agrega un residuo para detectar errores remanentes.
- El flujo integrado usado es: Datos → CRC → Hamming → Canal → Hamming Rx → CRC Rx.
- Hamming y CRC cumplen funciones diferentes dentro del sistema.
- Hamming corrige errores simples, pero puede fallar ante errores múltiples.
- CRC detecta errores remanentes, pero no corrige por sí mismo.
- Combinar Hamming y CRC mejora la confiabilidad respecto a usar el canal sin protección.
- La mejora de confiabilidad tiene un costo: se transmiten más bits.
- El BER del canal mide el daño producido directamente por el canal.
- El BER post-Hamming mide la mejora obtenida después de la corrección.
- El BER final de datos mide los errores que permanecen en la información útil recuperada.
- Al aumentar σ, normalmente aumenta el BER.
- Al aumentar la SNR, normalmente disminuye el BER.
- Las conclusiones estadísticas son más confiables cuando se analizan muchos bits.

La teoría aplicada en esta guía se fundamenta en comunicaciones digitales, codificación
de canal, control de errores, códigos Hamming, CRC, BER y SNR (Hamming, 1950; Lin &
Costello, 2004; Proakis & Salehi, 2008; Forouzan, 2013; Stallings, 2015).
"""
        )

    with tabs[7]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2013). *Data communications and networking* (5th ed.). McGraw-Hill Education.

Hamming, R. W. (1950). Error detecting and error correcting codes. *The Bell System Technical Journal, 29*(2), 147–160. https://doi.org/10.1002/j.1538-7305.1950.tb00463.x

Lin, S., & Costello, D. J. (2004). *Error control coding* (2nd ed.). Pearson.

Proakis, J. G., & Salehi, M. (2008). *Digital communications* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Tanenbaum, A. S., & Wetherall, D. J. (2011). *Computer networks* (5th ed.). Pearson.
"""
        )