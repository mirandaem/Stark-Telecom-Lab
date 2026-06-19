import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Utilidades generales
# ============================================================

def mostrar_dataframe(df: pd.DataFrame, hide_index: bool = True) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=hide_index)


def limpiar_bits(bits: str) -> str:
    return bits.strip().replace(" ", "").replace("\n", "").replace("\t", "")


def validar_bits(bits: str) -> bool:
    return len(bits) > 0 and all(bit in "01" for bit in bits)


def validar_generador_crc(generador: str) -> bool:
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


def truncar_bits(bits: str, max_len: int = 120) -> str:
    if len(bits) <= max_len:
        return bits

    return bits[:max_len] + f"... ({len(bits)} bits)"


def contar_errores_bits(bits_tx: str, bits_rx: str) -> int:
    longitud = min(len(bits_tx), len(bits_rx))

    if longitud == 0:
        return 0

    return sum(1 for a, b in zip(bits_tx[:longitud], bits_rx[:longitud]) if a != b)


def calcular_ber(bits_tx: str, bits_rx: str) -> float:
    longitud = min(len(bits_tx), len(bits_rx))

    if longitud == 0:
        return 0.0

    return contar_errores_bits(bits_tx, bits_rx) / longitud


def bits_a_bpsk(bits: str) -> np.ndarray:
    bits_array = np.fromiter((int(bit) for bit in bits), dtype=int)
    return np.where(bits_array == 1, 1.0, -1.0)


def transmitir_awgn(
    bits: str,
    sigma: float,
    semilla: int | None = None,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if len(bits) == 0:
        return "", np.array([]), np.array([]), np.array([])

    rng = np.random.default_rng(semilla)

    simbolos = bits_a_bpsk(bits)
    ruido = rng.normal(loc=0.0, scale=sigma, size=len(simbolos))
    recibido = simbolos + ruido
    bits_rx_array = np.where(recibido >= 0, 1, 0)

    bits_rx = "".join(str(bit) for bit in bits_rx_array)

    return bits_rx, simbolos, ruido, recibido


def calcular_snr(simbolos: np.ndarray, ruido: np.ndarray) -> Tuple[float, float, float, float]:
    if len(simbolos) == 0 or len(ruido) == 0:
        return 0.0, 0.0, math.inf, math.inf

    potencia_senal = float(np.mean(simbolos**2))
    potencia_ruido = float(np.mean(ruido**2))

    if potencia_ruido == 0:
        return potencia_senal, potencia_ruido, math.inf, math.inf

    snr = potencia_senal / potencia_ruido
    snr_db = 10 * math.log10(snr)

    return potencia_senal, potencia_ruido, snr, snr_db


def binario_a_polinomio(bits: str, variable: str = "x") -> str:
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


# ============================================================
# Tablas teóricas
# ============================================================

def construir_tabla_conceptos_crc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Concepto": [
                "Mensaje M(x)",
                "Generador G(x)",
                "Grado r",
                "Residuo R(x)",
                "Trama T(x)",
                "División módulo 2",
                "Verificación CRC",
            ],
            "Descripción": [
                "Secuencia de datos que se desea proteger.",
                "Polinomio binario compartido entre transmisor y receptor.",
                "Número de bits CRC que se agregan; r = grado de G(x).",
                "Resultado de dividir M(x)x^r entre G(x).",
                "Secuencia transmitida formada por datos y residuo CRC.",
                "División binaria donde la resta se implementa mediante XOR.",
                "El receptor divide T(x) recibido entre G(x); residuo cero implica que no se detecta error.",
            ],
        }
    )


def construir_tabla_metricas_crc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Métrica": [
                "BER del canal",
                "FER",
                "Tasa de detección condicional",
                "Tasa de error no detectado",
                "Tasa no detectada condicional",
                "SNR dB",
            ],
            "Expresión": [
                "Errores de bit / bits evaluados",
                "Tramas con error / tramas evaluadas",
                "Tramas erróneas detectadas / tramas con error",
                "Errores no detectados / tramas evaluadas",
                "Errores no detectados / tramas con error",
                "10 log10(Ps/Pn)",
            ],
            "Interpretación": [
                "Daño producido directamente por el canal.",
                "Proporción de tramas alteradas.",
                "Capacidad del CRC para detectar tramas realmente alteradas.",
                "Proporción global de tramas alteradas que pasaron como válidas.",
                "Probabilidad condicional de no detección dado que hubo error.",
                "Relación señal-ruido expresada en decibeles.",
            ],
            "Unidad o naturaleza": [
                "Adimensional",
                "Adimensional",
                "Adimensional",
                "Adimensional",
                "Adimensional",
                "dB",
            ],
        }
    )


def construir_tabla_division_polinomial_crc(
    datos: str,
    generador: str,
    residuo: str,
    trama: str,
) -> pd.DataFrame:
    r = len(generador) - 1
    datos_desplazados = datos + ("0" * r)

    return pd.DataFrame(
        {
            "Elemento": [
                "Mensaje M(x)",
                "Generador G(x)",
                "Grado r",
                "Mensaje desplazado M(x)x^r",
                "Residuo R(x)",
                "Trama T(x)",
            ],
            "Representación binaria": [
                datos,
                generador,
                str(r),
                datos_desplazados,
                residuo,
                trama,
            ],
            "Representación polinomial": [
                binario_a_polinomio(datos),
                binario_a_polinomio(generador),
                f"r = {r}",
                binario_a_polinomio(datos_desplazados),
                binario_a_polinomio(residuo),
                binario_a_polinomio(trama),
            ],
            "Interpretación": [
                "Información original que se desea proteger.",
                "Polinomio compartido por transmisor y receptor.",
                "Cantidad de bits CRC que se agregan a cada trama.",
                "Equivale a agregar r ceros al final del mensaje antes de dividir.",
                "Resultado de dividir M(x)x^r entre G(x) usando aritmética módulo 2.",
                "Secuencia final transmitida, formada por datos y residuo CRC.",
            ],
        }
    )


def construir_tabla_eleccion_generador_crc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Criterio para elegir G(x)": [
                "Debe iniciar en 1",
                "Debe terminar en 1",
                "Debe tener grado r definido",
                "Mayor grado agrega más redundancia",
                "Debe ser conocido por transmisor y receptor",
                "Debe seleccionarse según el tipo de error esperado",
                "Puede mejorar la detección de errores en ráfaga",
            ],
            "Razón técnica": [
                "Garantiza que el término de mayor grado exista.",
                "Evita que el generador sea divisible por x y asegura un término independiente.",
                "El grado r determina cuántos bits CRC se añaden.",
                "Más bits CRC aumentan la capacidad de detección, pero también el costo de transmisión.",
                "Ambos extremos deben usar el mismo polinomio para que la verificación sea válida.",
                "No todos los polinomios detectan igual todos los patrones de error.",
                "Los CRC se diseñan para detectar eficientemente errores agrupados en tramas.",
            ],
            "Ejemplo": [
                "1011 representa x³ + x + 1",
                "1001 termina en 1",
                "1011 tiene grado r = 3",
                "Un generador de 4 bits agrega 3 bits CRC",
                "Si Tx usa 1011, Rx también debe verificar con 1011",
                "Un sistema con ráfagas requiere un G(x) más robusto",
                "CRC es muy usado en redes y almacenamiento por esta propiedad",
            ],
        }
    )


def construir_tabla_generadores_crc_comunes() -> pd.DataFrame:
    generadores = ["1011", "10011", "11001", "100001"]

    return pd.DataFrame(
        {
            "Generador binario": generadores,
            "G(x)": [binario_a_polinomio(g) for g in generadores],
            "Grado r": [len(g) - 1 for g in generadores],
            "Bits CRC agregados": [len(g) - 1 for g in generadores],
            "Uso didáctico": [
                "Adecuado para ejemplos cortos y división manual.",
                "Permite observar mayor redundancia que 1011.",
                "Ejemplo alternativo de generador de grado 4.",
                "Permite visualizar cómo aumenta el residuo CRC.",
            ],
        }
    )


# ============================================================
# Lógica CRC y división módulo 2
# ============================================================

def crc_division_modulo_2(
    dividendo: str,
    divisor: str,
    registrar_pasos: bool = True,
    max_pasos: int = 40,
) -> Tuple[str, pd.DataFrame]:
    trabajo = list(dividendo)
    divisor_bits = list(divisor)
    longitud_divisor = len(divisor_bits)

    pasos = []

    for i in range(len(dividendo) - longitud_divisor + 1):
        segmento_antes = "".join(trabajo[i:i + longitud_divisor])

        if trabajo[i] == "1":
            for j in range(longitud_divisor):
                trabajo[i + j] = str(int(trabajo[i + j]) ^ int(divisor_bits[j]))

            segmento_despues = "".join(trabajo[i:i + longitud_divisor])
            operacion = f"{segmento_antes} XOR {divisor}"
        else:
            segmento_despues = segmento_antes
            operacion = "No se aplica XOR porque el bit líder es 0"

        if registrar_pasos and len(pasos) < max_pasos:
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

    residuo = "".join(trabajo[-(longitud_divisor - 1):])

    return residuo, pd.DataFrame(pasos)


def crc_generar(datos: str, generador: str) -> Tuple[str, str, pd.DataFrame]:
    r = len(generador) - 1
    dividendo = datos + ("0" * r)

    residuo, pasos = crc_division_modulo_2(
        dividendo=dividendo,
        divisor=generador,
        registrar_pasos=True,
    )

    trama = datos + residuo

    return residuo, trama, pasos


def crc_verificar(trama: str, generador: str) -> Tuple[bool, str, pd.DataFrame]:
    residuo, pasos = crc_division_modulo_2(
        dividendo=trama,
        divisor=generador,
        registrar_pasos=True,
    )

    valido = all(bit == "0" for bit in residuo)

    return valido, residuo, pasos


# ============================================================
# Codificación y evaluación exacta por tramas
# ============================================================

def codificar_tramas_crc(
    datos: str,
    tamano_payload: int,
    generador: str,
    max_tabla: int = 30,
) -> Tuple[str, int, int, pd.DataFrame]:
    datos_rellenados, padding = rellenar_a_multiplo(datos, tamano_payload)
    bloques = dividir_en_bloques(datos_rellenados, tamano_payload)

    r = len(generador) - 1
    longitud_trama = tamano_payload + r

    flujo = []
    filas = []

    for i, bloque in enumerate(bloques, start=1):
        residuo, trama, _ = crc_generar(bloque, generador)
        flujo.append(trama)

        if i <= max_tabla:
            filas.append(
                {
                    "Trama": i,
                    "Datos": bloque,
                    "Residuo CRC": residuo,
                    "Trama transmitida": trama,
                }
            )

    return "".join(flujo), padding, longitud_trama, pd.DataFrame(filas)


def evaluar_tramas_crc(
    flujo_tx: str,
    flujo_rx: str,
    longitud_trama: int,
    tamano_payload: int,
    generador: str,
    max_tabla: int = 30,
) -> Tuple[Dict[str, float | int], pd.DataFrame]:
    tramas_tx = dividir_en_bloques(flujo_tx, longitud_trama)
    tramas_rx = dividir_en_bloques(flujo_rx, longitud_trama)

    tramas_evaluadas = 0
    tramas_con_error = 0
    tramas_detectadas = 0
    tramas_detectadas_con_error = 0
    errores_no_detectados = 0
    falsas_alarmas = 0

    filas = []

    for i, (tx, rx) in enumerate(zip(tramas_tx, tramas_rx), start=1):
        if len(tx) != longitud_trama or len(rx) != longitud_trama:
            continue

        tramas_evaluadas += 1

        errores_trama = contar_errores_bits(tx, rx)
        tiene_error = errores_trama > 0

        valido, residuo_rx, _ = crc_verificar(rx, generador)
        detectado_por_crc = not valido

        if tiene_error:
            tramas_con_error += 1

        if detectado_por_crc:
            tramas_detectadas += 1

        if tiene_error and detectado_por_crc:
            tramas_detectadas_con_error += 1

        if tiene_error and valido:
            errores_no_detectados += 1

        if (not tiene_error) and detectado_por_crc:
            falsas_alarmas += 1

        if i <= max_tabla:
            if tiene_error and detectado_por_crc:
                estado = "Error detectado"
            elif tiene_error and valido:
                estado = "Error no detectado"
            elif (not tiene_error) and detectado_por_crc:
                estado = "Falsa alarma"
            else:
                estado = "Sin error"

            filas.append(
                {
                    "Trama": i,
                    "Trama Tx": tx,
                    "Trama Rx": rx,
                    "Payload Rx": rx[:tamano_payload],
                    "Errores en trama [bits]": errores_trama,
                    "Residuo verificación": residuo_rx,
                    "CRC válido": "Sí" if valido else "No",
                    "Estado": estado,
                }
            )

    fer = tramas_con_error / tramas_evaluadas if tramas_evaluadas else 0.0

    tasa_no_detectada_global = (
        errores_no_detectados / tramas_evaluadas if tramas_evaluadas else 0.0
    )

    tasa_no_detectada_condicional = (
        errores_no_detectados / tramas_con_error if tramas_con_error else 0.0
    )

    tasa_deteccion_condicional = (
        tramas_detectadas_con_error / tramas_con_error if tramas_con_error else 0.0
    )

    metricas = {
        "Tramas evaluadas": tramas_evaluadas,
        "Tramas con error": tramas_con_error,
        "Tramas detectadas por CRC": tramas_detectadas,
        "Tramas detectadas con error real": tramas_detectadas_con_error,
        "Errores no detectados": errores_no_detectados,
        "Falsas alarmas": falsas_alarmas,
        "FER": fer,
        "Tasa de detección condicional": tasa_deteccion_condicional,
        "Tasa de error no detectado": tasa_no_detectada_global,
        "Tasa no detectada condicional": tasa_no_detectada_condicional,
    }

    return metricas, pd.DataFrame(filas)


def simular_crc_awgn(
    datos: str,
    tamano_payload: int,
    generador: str,
    sigma: float,
    semilla_canal: int,
    max_tabla: int = 30,
) -> Tuple[Dict[str, float | int], Dict[str, object]]:
    flujo_tx, padding, longitud_trama, tabla_tx = codificar_tramas_crc(
        datos=datos,
        tamano_payload=tamano_payload,
        generador=generador,
        max_tabla=max_tabla,
    )

    flujo_rx, simbolos, ruido, recibido = transmitir_awgn(
        bits=flujo_tx,
        sigma=sigma,
        semilla=semilla_canal,
    )

    metricas_crc, tabla_rx = evaluar_tramas_crc(
        flujo_tx=flujo_tx,
        flujo_rx=flujo_rx,
        longitud_trama=longitud_trama,
        tamano_payload=tamano_payload,
        generador=generador,
        max_tabla=max_tabla,
    )

    potencia_senal, potencia_ruido, snr, snr_db = calcular_snr(simbolos, ruido)

    bits_evaluados = min(len(flujo_tx), len(flujo_rx))
    errores_bit_canal = contar_errores_bits(flujo_tx, flujo_rx)
    ber_canal = calcular_ber(flujo_tx, flujo_rx)

    resumen = {
        "σ": sigma,
        "σ²": sigma**2,
        "SNR": snr,
        "SNR dB": snr_db,
        "Potencia señal": potencia_senal,
        "Potencia ruido": potencia_ruido,
        "Bits de datos originales": len(datos),
        "Bits transmitidos": len(flujo_tx),
        "Bits evaluados": bits_evaluados,
        "Bits de redundancia CRC": len(flujo_tx) - len(datos),
        "Padding aplicado": padding,
        "Longitud de trama": longitud_trama,
        "Errores de bit del canal": errores_bit_canal,
        "BER del canal": ber_canal,
        **metricas_crc,
    }

    detalle = {
        "flujo_tx": flujo_tx,
        "flujo_rx": flujo_rx,
        "simbolos": simbolos,
        "ruido": ruido,
        "recibido": recibido,
        "tabla_tx": tabla_tx,
        "tabla_rx": tabla_rx,
        "padding": padding,
        "longitud_trama": longitud_trama,
    }

    return resumen, detalle


# ============================================================
# Comparación estadística rápida
# ============================================================

def generar_valores_sigma_desde_texto(texto: str) -> List[float]:
    valores = []

    for parte in texto.split(","):
        parte = parte.strip()

        if parte:
            valores.append(float(parte))

    return valores


def probabilidad_error_bpsk_awgn(sigma: float) -> float:
    if sigma <= 0:
        return 0.0

    return 0.5 * math.erfc(1 / (math.sqrt(2) * sigma))


def simular_barrido_crc(
    cantidad_tramas: int,
    tamano_payload: int,
    generador: str,
    valores_sigma: List[float],
    semilla_datos: int,
    semilla_canal_base: int,
) -> pd.DataFrame:
    r = len(generador) - 1
    longitud_trama = tamano_payload + r

    bits_datos_originales = cantidad_tramas * tamano_payload
    bits_transmitidos = cantidad_tramas * longitud_trama
    bits_redundancia_crc = cantidad_tramas * r

    potencia_senal = 1.0

    filas = []

    for i, sigma in enumerate(valores_sigma):
        sigma = float(sigma)

        rng = np.random.default_rng(
            semilla_datos + semilla_canal_base + (1000 * (i + 1))
        )

        prob_error_bit = probabilidad_error_bpsk_awgn(sigma)

        errores_bit_canal = int(
            rng.binomial(
                n=bits_transmitidos,
                p=prob_error_bit,
            )
        )

        ber_canal = (
            errores_bit_canal / bits_transmitidos
            if bits_transmitidos > 0
            else 0.0
        )

        prob_trama_con_error = 1 - ((1 - prob_error_bit) ** longitud_trama)

        tramas_con_error = int(
            rng.binomial(
                n=cantidad_tramas,
                p=prob_trama_con_error,
            )
        )

        fer = (
            tramas_con_error / cantidad_tramas
            if cantidad_tramas > 0
            else 0.0
        )

        prob_no_detectado_condicional = 1 / (2**r)

        errores_no_detectados = int(
            rng.binomial(
                n=tramas_con_error,
                p=prob_no_detectado_condicional,
            )
        )

        tramas_detectadas_con_error = tramas_con_error - errores_no_detectados
        tramas_detectadas_por_crc = tramas_detectadas_con_error
        falsas_alarmas = 0

        tasa_deteccion_condicional = (
            tramas_detectadas_con_error / tramas_con_error
            if tramas_con_error > 0
            else 0.0
        )

        tasa_error_no_detectado = (
            errores_no_detectados / cantidad_tramas
            if cantidad_tramas > 0
            else 0.0
        )

        tasa_no_detectada_condicional = (
            errores_no_detectados / tramas_con_error
            if tramas_con_error > 0
            else 0.0
        )

        potencia_ruido = sigma**2

        if potencia_ruido == 0:
            snr = math.inf
            snr_db = math.inf
        else:
            snr = potencia_senal / potencia_ruido
            snr_db = 10 * math.log10(snr)

        filas.append(
            {
                "σ": sigma,
                "σ²": sigma**2,
                "SNR": snr,
                "SNR dB": snr_db,
                "Potencia señal": potencia_senal,
                "Potencia ruido": potencia_ruido,
                "Bits de datos originales": bits_datos_originales,
                "Bits transmitidos": bits_transmitidos,
                "Bits evaluados": bits_transmitidos,
                "Bits de redundancia CRC": bits_redundancia_crc,
                "Padding aplicado": 0,
                "Longitud de trama": longitud_trama,
                "Errores de bit del canal": errores_bit_canal,
                "BER del canal": ber_canal,
                "Tramas evaluadas": cantidad_tramas,
                "Tramas con error": tramas_con_error,
                "Tramas detectadas por CRC": tramas_detectadas_por_crc,
                "Tramas detectadas con error real": tramas_detectadas_con_error,
                "Errores no detectados": errores_no_detectados,
                "Falsas alarmas": falsas_alarmas,
                "FER": fer,
                "Tasa de detección condicional": tasa_deteccion_condicional,
                "Tasa de error no detectado": tasa_error_no_detectado,
                "Tasa no detectada condicional": tasa_no_detectada_condicional,
            }
        )

    return pd.DataFrame(filas)


# ============================================================
# Gráficas
# ============================================================

def preparar_metrica_logaritmica(
    df: pd.DataFrame,
    columna_metrica: str,
    columna_n: str,
    columna_salida: str,
    columna_limite: str = "Límite experimental 1/N",
) -> pd.DataFrame:
    df_plot = df.copy()

    if columna_metrica not in df_plot.columns:
        raise ValueError(f"No existe la columna de métrica: {columna_metrica}")

    if columna_n not in df_plot.columns:
        raise ValueError(f"No existe la columna de tamaño muestral: {columna_n}")

    df_plot[columna_n] = pd.to_numeric(df_plot[columna_n], errors="coerce")
    df_plot[columna_metrica] = pd.to_numeric(df_plot[columna_metrica], errors="coerce")

    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)
    df_plot = df_plot.dropna(subset=[columna_metrica, columna_n])
    df_plot = df_plot[df_plot[columna_n] > 0]

    if df_plot.empty:
        return df_plot

    df_plot[columna_limite] = 1 / df_plot[columna_n]

    df_plot[columna_salida] = df_plot.apply(
        lambda fila: fila[columna_metrica]
        if fila[columna_metrica] > 0
        else fila[columna_limite],
        axis=1,
    )

    return df_plot


def graficar_metrica_logaritmica_con_limite(
    df: pd.DataFrame,
    columna_x: str,
    columna_metrica: str,
    columna_n: str,
    titulo: str,
    etiqueta_x: str,
    etiqueta_y: str,
    etiqueta_metrica: str,
    ordenar_ascendente: bool = True,
) -> None:
    if df.empty:
        st.info("No hay datos disponibles para graficar.")
        return

    if columna_x not in df.columns:
        st.warning(f"No se encontró la columna '{columna_x}' para el eje X.")
        return

    try:
        df_plot = preparar_metrica_logaritmica(
            df=df,
            columna_metrica=columna_metrica,
            columna_n=columna_n,
            columna_salida=f"{columna_metrica} para gráfica",
        )
    except ValueError as error:
        st.warning(str(error))
        return

    if df_plot.empty:
        st.info("No hay datos válidos para graficar.")
        return

    df_plot[columna_x] = pd.to_numeric(df_plot[columna_x], errors="coerce")
    df_plot = df_plot.dropna(subset=[columna_x])

    if df_plot.empty:
        st.info("No hay datos numéricos válidos para graficar.")
        return

    df_plot = df_plot.sort_values(columna_x, ascending=ordenar_ascendente)

    df_con_eventos = df_plot[df_plot[columna_metrica] > 0]
    df_sin_eventos = df_plot[df_plot[columna_metrica] == 0]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if not df_con_eventos.empty:
        ax.scatter(
            df_con_eventos[columna_x],
            df_con_eventos[columna_metrica],
            marker="o",
            s=85,
            label=etiqueta_metrica,
            zorder=3,
        )

    if not df_sin_eventos.empty:
        ax.scatter(
            df_sin_eventos[columna_x],
            df_sin_eventos["Límite experimental 1/N"],
            marker="v",
            s=95,
            label="0 eventos observados: métrica < 1/N",
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_title(titulo)
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel(etiqueta_y)

    valores_y = []

    if not df_con_eventos.empty:
        valores_y.extend(df_con_eventos[columna_metrica].tolist())

    if not df_sin_eventos.empty:
        valores_y.extend(df_sin_eventos["Límite experimental 1/N"].tolist())

    valores_y = [
        float(valor)
        for valor in valores_y
        if pd.notna(valor) and np.isfinite(valor) and float(valor) > 0
    ]

    if valores_y:
        y_min = max(min(valores_y) / 5, 1e-12)
        y_max = min(max(valores_y) * 5, 2.0)

        if y_min < y_max:
            ax.set_ylim(y_min, y_max)

    ax.grid(True, which="major", linestyle="-", linewidth=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Cada punto representa una evaluación estadística para un valor específico del parámetro. "
        "El eje vertical se mantiene en escala logarítmica para visualizar métricas pequeñas como BER, FER o tasa de error."
    )

    if not df_sin_eventos.empty:
        st.info(
            """
Los marcadores triangulares indican que no se observó ese evento durante la simulación.
Esto no significa que la probabilidad real sea exactamente cero. Se representa como:

**métrica < 1/N**

donde N es la cantidad de bits, tramas o pruebas evaluadas.
"""
        )


def graficar_fer_vs_sigma(df: pd.DataFrame) -> None:
    graficar_metrica_logaritmica_con_limite(
        df=df,
        columna_x="σ",
        columna_metrica="FER",
        columna_n="Tramas evaluadas",
        titulo="FER vs σ",
        etiqueta_x="Desviación estándar del ruido σ [amplitud normalizada]",
        etiqueta_y="FER [adimensional, escala logarítmica]",
        etiqueta_metrica="FER observada",
        ordenar_ascendente=True,
    )


def graficar_tasa_no_detectada_vs_sigma(df: pd.DataFrame) -> None:
    graficar_metrica_logaritmica_con_limite(
        df=df,
        columna_x="σ",
        columna_metrica="Tasa de error no detectado",
        columna_n="Tramas evaluadas",
        titulo="Tasa de error no detectado vs σ",
        etiqueta_x="Desviación estándar del ruido σ [amplitud normalizada]",
        etiqueta_y="Tasa de error no detectado [adimensional, escala logarítmica]",
        etiqueta_metrica="Tasa de error no detectado observada",
        ordenar_ascendente=True,
    )


def graficar_ber_canal_vs_sigma(df: pd.DataFrame) -> None:
    graficar_metrica_logaritmica_con_limite(
        df=df,
        columna_x="σ",
        columna_metrica="BER del canal",
        columna_n="Bits evaluados",
        titulo="BER del canal vs σ",
        etiqueta_x="Desviación estándar del ruido σ [amplitud normalizada]",
        etiqueta_y="BER del canal [adimensional, escala logarítmica]",
        etiqueta_metrica="BER del canal observada",
        ordenar_ascendente=True,
    )


def graficar_ber_canal_vs_snr(df: pd.DataFrame) -> None:
    graficar_metrica_logaritmica_con_limite(
        df=df,
        columna_x="SNR dB",
        columna_metrica="BER del canal",
        columna_n="Bits evaluados",
        titulo="BER del canal vs SNR",
        etiqueta_x="SNR [dB]",
        etiqueta_y="BER del canal [adimensional, escala logarítmica]",
        etiqueta_metrica="BER del canal observada",
        ordenar_ascendente=True,
    )


def graficar_muestras_awgn(detalle: Dict[str, object], max_muestras: int = 40) -> None:
    flujo_tx = str(detalle["flujo_tx"])[:max_muestras]
    flujo_rx = str(detalle["flujo_rx"])[:max_muestras]

    simbolos = np.array(detalle["simbolos"][:max_muestras])
    recibido = np.array(detalle["recibido"][:max_muestras])
    ruido = np.array(detalle["ruido"][:max_muestras])

    x = np.arange(1, len(flujo_tx) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(x, simbolos, basefmt=" ", label="Símbolo transmitido")
    ax.scatter(x, recibido, marker="x", label="Valor recibido")
    ax.axhline(0, linestyle="--", linewidth=1, label="Umbral")
    ax.set_title("Muestras discretas transmitidas y recibidas")
    ax.set_xlabel("Índice de muestra [n]")
    ax.set_ylabel("Amplitud normalizada")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig_ruido, ax_ruido = plt.subplots(figsize=(10, 3.5))
    ax_ruido.scatter(x, ruido, marker="o", label="Ruido por muestra")
    ax_ruido.axhline(0, linestyle="--", linewidth=1)
    ax_ruido.set_title("Ruido AWGN por muestra")
    ax_ruido.set_xlabel("Índice de muestra [n]")
    ax_ruido.set_ylabel("Ruido n[n] [amplitud normalizada]")
    ax_ruido.grid(True)
    ax_ruido.legend()
    st.pyplot(fig_ruido)
    plt.close(fig_ruido)

    tabla = pd.DataFrame(
        {
            "Muestra [n]": x,
            "Bit Tx": list(flujo_tx),
            "Bit Rx": list(flujo_rx),
            "Estado": [
                "Correcto" if a == b else "Error"
                for a, b in zip(flujo_tx, flujo_rx)
            ],
        }
    )

    mostrar_dataframe(tabla)


# ============================================================
# Texto explicativo
# ============================================================

def mostrar_explicacion_metricas_crc() -> None:
    st.markdown(
        """
### Interpretación de métricas estadísticas

En esta sección se evalúa el comportamiento del CRC bajo diferentes niveles de ruido. La simulación interactiva permite observar el proceso exacto de codificación, transmisión y verificación de tramas. La comparación estadística utiliza un modelo estadístico aproximado para analizar tendencias con muchas tramas sin hacer que la aplicación tarde demasiado en responder.

Las métricas se interpretan de la siguiente manera:

- **BER del canal:** mide la proporción de bits alterados directamente por el canal.
- **FER:** mide la proporción de tramas que presentan al menos un error.
- **Tasa de detección condicional:** indica qué proporción de las tramas realmente alteradas fueron detectadas por CRC.
- **Tasa de error no detectado:** mide la proporción global de tramas alteradas que no fueron identificadas por la verificación CRC.
- **Tasa no detectada condicional:** mide la proporción de errores no detectados considerando únicamente las tramas que sí fueron alteradas.

BER, FER y las tasas son magnitudes **adimensionales**, porque representan razones entre cantidades de bits, tramas o eventos.

Cuando una métrica toma el valor cero en una simulación, esto significa que no se observó ese evento en la cantidad de pruebas realizadas. No debe interpretarse como una probabilidad real absolutamente nula. Por esta razón, en las gráficas logarítmicas se utiliza el criterio:

$$
\\text{métrica} < \\frac{1}{N}
$$

donde $N$ representa el número de bits, tramas o pruebas evaluadas.
"""
    )


# ============================================================
# Interfaz principal
# ============================================================

def render_guia_05() -> None:
    st.title("Guía 5: Detección de errores mediante CRC")

    st.markdown(
        """
El Código de Redundancia Cíclica, conocido como CRC, es una técnica de detección de errores ampliamente utilizada en sistemas de comunicación digital y redes de datos. Su propósito es agregar redundancia calculada matemáticamente a una secuencia de bits para que el receptor pueda verificar si la trama recibida presenta alteraciones.

En esta guía se estudia el CRC desde tres perspectivas: el cálculo manual del residuo mediante división polinomial, la verificación de tramas recibidas y el análisis estadístico de su comportamiento bajo un canal con ruido AWGN.
"""
    )

    tabs = st.tabs(
        [
            "Objetivos",
            "Teoría",
            "Cálculo paso a paso",
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

Comprender el funcionamiento del Código de Redundancia Cíclica como técnica de detección de errores en sistemas de comunicación digital.

**Objetivos específicos**

1. Representar secuencias binarias como polinomios sobre GF(2).
2. Comprender la función del polinomio generador $G(x)$.
3. Explicar cómo se elige un generador CRC válido.
4. Calcular el residuo CRC mediante división polinomial módulo 2.
5. Construir la trama transmitida a partir de datos y residuo.
6. Verificar tramas recibidas usando el mismo generador.
7. Analizar la diferencia entre detectar y corregir errores.
8. Simular tramas CRC sobre un canal AWGN.
9. Calcular BER del canal, FER y tasa de error no detectado.
10. Interpretar métricas estadísticas usando escala logarítmica.
11. Identificar correctamente las unidades o naturaleza de las magnitudes graficadas.
"""
        )

    with tabs[1]:
        st.header("Fundamentación teórica")

        st.markdown(
            """
### 1. Redundancia y detección de errores

En los sistemas de comunicación digital, la información transmitida puede alterarse por ruido, interferencia o limitaciones del canal. Una forma de enfrentar este problema es agregar redundancia a los datos transmitidos. La redundancia no forma parte del mensaje original, pero permite al receptor verificar si la información pudo haber sido modificada.

CRC pertenece al grupo de técnicas de **detección de errores**. Esto significa que puede indicar que una trama probablemente está alterada, pero no identifica necesariamente la posición exacta del error ni corrige la trama por sí mismo.

### 2. Representación polinómica

En CRC, una secuencia binaria se interpreta como un polinomio cuyos coeficientes son bits. Por ejemplo, la secuencia 1011 representa:

$$
x^3 + x + 1
$$

El bit más significativo corresponde al término de mayor grado. Esta representación permite usar división polinómica sobre el campo binario GF(2).

### 3. Teoría del polinomio generador G(x)

El polinomio generador $G(x)$ es el elemento central del CRC. Este polinomio define la regla matemática con la que el transmisor calcula el residuo y con la que el receptor verifica la trama recibida.

En forma binaria, cada bit del generador representa un coeficiente del polinomio. Por ejemplo:

$$
G(x) = 1011
$$

equivale a:

$$
G(x) = x^3 + x + 1
$$

El bit más significativo representa el término de mayor grado y el bit menos significativo representa el término independiente.

Para que un generador CRC sea válido en esta guía, debe cumplir tres condiciones básicas:

- Debe estar formado únicamente por bits 0 y 1.
- Debe iniciar en 1.
- Debe terminar en 1.

Si el generador tiene longitud $L$, entonces su grado es:

$$
r = L - 1
$$

Ese grado $r$ determina cuántos bits CRC se agregan al mensaje original.
"""
        )

        st.subheader("Criterios para elegir G(x)")
        mostrar_dataframe(construir_tabla_eleccion_generador_crc())

        st.subheader("Ejemplos de generadores CRC")
        mostrar_dataframe(construir_tabla_generadores_crc_comunes())

        st.subheader("Desarrollo de la división polinomial")

        st.markdown(
            """
El proceso de generación CRC se basa en una división polinomial sobre el campo binario $GF(2)$. En este campo solo existen los valores 0 y 1, y las operaciones se realizan módulo 2.

Por esa razón, la resta polinomial se implementa mediante XOR:

$$
0 \\oplus 0 = 0
$$

$$
0 \\oplus 1 = 1
$$

$$
1 \\oplus 0 = 1
$$

$$
1 \\oplus 1 = 0
$$

Para calcular el CRC se siguen estos pasos:

1. Se toma el mensaje original $M(x)$.
2. Se identifica el grado $r$ del generador $G(x)$.
3. Se multiplica el mensaje por $x^r$, lo que en binario equivale a agregar $r$ ceros al final.
4. Se divide $M(x)x^r$ entre $G(x)$ usando división módulo 2.
5. El residuo de esa división es $R(x)$.
6. La trama transmitida se forma concatenando el mensaje original con el residuo.

Matemáticamente:

$$
R(x) = M(x)x^r \\bmod G(x)
$$

y la trama transmitida es:

$$
T(x) = M(x)x^r + R(x)
$$

En forma binaria:

$$
t = m || R
$$

donde $||$ representa concatenación.
"""
        )

        st.subheader("Conceptos principales")
        mostrar_dataframe(construir_tabla_conceptos_crc())

        st.subheader("Métricas usadas")
        mostrar_dataframe(construir_tabla_metricas_crc())

    with tabs[2]:
        st.header("Cálculo paso a paso del CRC")

        st.markdown(
            """
En esta sección se calcula el residuo CRC de una trama corta para observar el proceso de división módulo 2. La operación fundamental es XOR, ya que en GF(2) la resta y la suma se comportan como una suma módulo 2.
"""
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            datos_paso = limpiar_bits(
                st.text_input(
                    "Datos binarios M",
                    value="1011001",
                    key="g5_datos_paso",
                )
            )

            generador_paso = limpiar_bits(
                st.text_input(
                    "Generador CRC G",
                    value="1011",
                    key="g5_generador_paso",
                )
            )

            ejecutar_paso = st.button(
                "Calcular CRC paso a paso",
                key="g5_calcular_paso",
            )

        with col2:
            st.info(
                """
Use mensajes cortos para observar mejor la división.

Ejemplo:

Datos: 1011001  
Generador: 1011
"""
            )

        if ejecutar_paso:
            if not validar_bits(datos_paso):
                st.error("Los datos deben contener únicamente 0 y 1.")
            elif not validar_generador_crc(generador_paso):
                st.error("El generador debe ser binario, iniciar en 1 y terminar en 1.")
            else:
                r = len(generador_paso) - 1
                residuo, trama, pasos = crc_generar(datos_paso, generador_paso)
                valido, residuo_verificacion, _ = crc_verificar(trama, generador_paso)

                st.subheader("Representación polinómica")
                st.markdown(f"**M(x):** {binario_a_polinomio(datos_paso)}")
                st.markdown(f"**G(x):** {binario_a_polinomio(generador_paso)}")
                st.markdown(f"**Grado del generador:** r = {r}")
                st.markdown(f"**Bits CRC agregados:** {r} bits")

                st.subheader("Desarrollo polinomial del CRC")

                st.markdown(
                    """
La siguiente tabla muestra cómo se interpreta cada elemento del proceso CRC desde la forma binaria hasta la forma polinomial.
"""
                )

                mostrar_dataframe(
                    construir_tabla_division_polinomial_crc(
                        datos=datos_paso,
                        generador=generador_paso,
                        residuo=residuo,
                        trama=trama,
                    )
                )

                st.subheader("Resultado del cálculo")

                st.code(
                    f"Datos:           {datos_paso}\n"
                    f"Ceros agregados: {'0' * r}\n"
                    f"Residuo CRC:     {residuo}\n"
                    f"Trama final:     {trama}",
                    language="text",
                )

                st.subheader("Pasos de división módulo 2")
                mostrar_dataframe(pasos)

                st.subheader("Verificación de la trama transmitida")

                st.code(
                    f"Trama verificada: {trama}\n"
                    f"Residuo:          {residuo_verificacion}\n"
                    f"CRC válido:       {'Sí' if valido else 'No'}",
                    language="text",
                )

    with tabs[3]:
        st.header("Simulación interactiva CRC sobre canal AWGN")

        st.markdown(
            """
En esta sección se divide una secuencia de datos en tramas, se calcula el CRC de cada trama y luego se transmite el flujo completo por un canal AWGN. El receptor verifica cada trama con el mismo generador.
"""
        )

        col_param, col_info = st.columns([1, 1])

        with col_param:
            modo_entrada = st.radio(
                "Modo de entrada",
                ["Mensaje manual", "Bits aleatorios"],
                key="g5_modo_entrada",
            )

            if modo_entrada == "Mensaje manual":
                datos = limpiar_bits(
                    st.text_area(
                        "Datos binarios",
                        value="1011001110001111010101010011",
                        key="g5_datos_manual",
                    )
                )
            else:
                cantidad_bits = st.selectbox(
                    "Cantidad de bits [bits]",
                    [32, 64, 128, 1000, 5000, 10000, 50000],
                    index=3,
                    key="g5_cantidad_bits",
                )

                semilla_datos = st.number_input(
                    "Semilla para datos",
                    min_value=0,
                    max_value=999999,
                    value=505,
                    step=1,
                    key="g5_semilla_datos",
                )

                datos = generar_bits_aleatorios(
                    int(cantidad_bits),
                    semilla=int(semilla_datos),
                )

                st.code(truncar_bits(datos), language="text")

            tamano_payload = st.selectbox(
                "Tamaño del payload por trama [bits]",
                [4, 8, 16, 32, 64],
                index=2,
                key="g5_tamano_payload",
            )

            generador = limpiar_bits(
                st.text_input(
                    "Generador CRC",
                    value="1011",
                    key="g5_generador_crc",
                )
            )

            sigma = st.slider(
                "Desviación estándar del ruido σ [amplitud normalizada]",
                min_value=0.0,
                max_value=1.5,
                value=0.35,
                step=0.05,
                key="g5_sigma",
            )

            semilla_canal = st.number_input(
                "Semilla del canal",
                min_value=0,
                max_value=999999,
                value=707,
                step=1,
                key="g5_semilla_canal",
            )

            max_tabla = st.slider(
                "Cantidad de tramas a mostrar en tablas",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                key="g5_max_tabla",
            )

            ejecutar = st.button("Ejecutar simulación CRC", key="g5_ejecutar_simulacion")

        with col_info:
            st.info(
                "Use pocos bits para observar el proceso por tramas. "
                "Use muchos bits para obtener métricas estadísticas más representativas."
            )

            st.metric(
                "Varianza del ruido σ² [amplitud normalizada²]",
                f"{sigma**2:.6f}",
            )

        if not validar_bits(datos):
            st.error("Los datos deben contener únicamente 0 y 1.")
        elif not validar_generador_crc(generador):
            st.error("El generador CRC debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar:
            resumen, detalle = simular_crc_awgn(
                datos=datos,
                tamano_payload=int(tamano_payload),
                generador=generador,
                sigma=float(sigma),
                semilla_canal=int(semilla_canal),
                max_tabla=int(max_tabla),
            )

            st.subheader("Métricas principales")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("BER del canal [adimensional]", f"{resumen['BER del canal']:.6f}")
            c2.metric("FER [adimensional]", f"{resumen['FER']:.6f}")
            c3.metric("Errores no detectados", f"{int(resumen['Errores no detectados'])}")
            c4.metric(
                "SNR [dB]",
                "∞" if math.isinf(float(resumen["SNR dB"])) else f"{resumen['SNR dB']:.3f}",
            )

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Tramas evaluadas", f"{int(resumen['Tramas evaluadas'])}")
            c6.metric("Tramas con error", f"{int(resumen['Tramas con error'])}")
            c7.metric("Detectadas por CRC", f"{int(resumen['Tramas detectadas por CRC'])}")
            c8.metric(
                "Tasa no detectada [adimensional]",
                f"{resumen['Tasa de error no detectado']:.6f}",
            )

            st.subheader("Resumen numérico")
            mostrar_dataframe(pd.DataFrame([resumen]))

            st.subheader("Primeras tramas transmitidas")
            mostrar_dataframe(detalle["tabla_tx"])

            st.subheader("Primeras tramas verificadas en recepción")
            mostrar_dataframe(detalle["tabla_rx"])

            st.subheader("Muestras del canal")
            graficar_muestras_awgn(detalle, max_muestras=40)

            st.session_state["g5_resultado_resumen"] = resumen
            st.session_state["g5_resultado_detalle"] = detalle

    with tabs[4]:
        st.header("Comparación estadística del CRC")

        mostrar_explicacion_metricas_crc()

        col_param, col_info = st.columns([1, 1])

        with col_param:
            cantidad_tramas = st.selectbox(
                "Cantidad de tramas para comparación",
                [100, 500, 1000, 5000, 10000],
                index=3,
                key="g5_cantidad_tramas_comparacion_v8",
            )

            tamano_payload_comp = st.selectbox(
                "Tamaño de payload [bits]",
                [8, 16, 32, 64],
                index=1,
                key="g5_payload_comparacion_v8",
            )

            generador_comp = limpiar_bits(
                st.text_input(
                    "Generador CRC",
                    value="1011",
                    key="g5_generador_comparacion_v8",
                )
            )

            sigmas_texto = st.text_input(
                "Valores de σ separados por coma [amplitud normalizada]",
                value="0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00",
                key="g5_sigmas_comparacion_v8",
            )

            semilla_datos_comp = st.number_input(
                "Semilla de datos",
                min_value=0,
                max_value=999999,
                value=900,
                step=1,
                key="g5_semilla_datos_comparacion_v8",
            )

            semilla_canal_comp = st.number_input(
                "Semilla base del canal",
                min_value=0,
                max_value=999999,
                value=1200,
                step=1,
                key="g5_semilla_canal_comparacion_v8",
            )

            ejecutar_comp = st.button(
                "Ejecutar comparación estadística",
                key="g5_ejecutar_comparacion_v8",
            )

        with col_info:
            st.info(
                "Esta comparación evalúa muchas tramas. "
                "Las gráficas usan escala logarítmica y puntos discretos, sin líneas continuas."
            )

        if not validar_generador_crc(generador_comp):
            st.error("El generador CRC debe ser binario, iniciar en 1 y terminar en 1.")
        elif ejecutar_comp:
            try:
                valores_sigma = generar_valores_sigma_desde_texto(sigmas_texto)
            except ValueError:
                st.error("Ingrese valores numéricos válidos separados por coma.")
                return

            if len(valores_sigma) == 0:
                st.error("Debe ingresar al menos un valor de σ.")
                return

            if any(valor < 0 for valor in valores_sigma):
                st.error("Los valores de σ no pueden ser negativos.")
                return

            with st.spinner("Ejecutando comparación estadística..."):
                df_comp = simular_barrido_crc(
                    cantidad_tramas=int(cantidad_tramas),
                    tamano_payload=int(tamano_payload_comp),
                    generador=generador_comp,
                    valores_sigma=valores_sigma,
                    semilla_datos=int(semilla_datos_comp),
                    semilla_canal_base=int(semilla_canal_comp),
                )

            st.subheader("Tabla de comparación")

            columnas = [
                "σ",
                "σ²",
                "SNR dB",
                "Bits evaluados",
                "BER del canal",
                "Tramas evaluadas",
                "Tramas con error",
                "FER",
                "Tramas detectadas por CRC",
                "Errores no detectados",
                "Tasa de detección condicional",
                "Tasa de error no detectado",
                "Tasa no detectada condicional",
            ]

            columnas_existentes = [col for col in columnas if col in df_comp.columns]
            mostrar_dataframe(df_comp[columnas_existentes])

            st.subheader("FER vs σ")
            graficar_fer_vs_sigma(df_comp)

            st.subheader("Tasa de error no detectado vs σ")
            graficar_tasa_no_detectada_vs_sigma(df_comp)

            st.subheader("BER del canal vs σ")
            graficar_ber_canal_vs_sigma(df_comp)

            st.subheader("BER del canal vs SNR")
            graficar_ber_canal_vs_snr(df_comp)

            st.session_state["g5_df_comparacion_v8"] = df_comp

    with tabs[5]:
        st.header("Análisis y dinámica")

        st.markdown(
            """
Esta sección conecta los resultados de simulación con la interpretación teórica del CRC. El objetivo es que el estudiante distinga entre el error producido por el canal, la capacidad de detección del CRC y el costo de redundancia agregado por el polinomio generador.
"""
        )

        if "g5_resultado_resumen" in st.session_state:
            st.subheader("Última simulación interactiva")

            resumen = st.session_state["g5_resultado_resumen"]
            mostrar_dataframe(pd.DataFrame([resumen]))

            st.markdown(
                f"""
En la última simulación se evaluaron **{int(resumen['Tramas evaluadas'])} tramas**. De ellas, **{int(resumen['Tramas con error'])}** presentaron al menos un error de bit.

El canal produjo una BER de **{resumen['BER del canal']:.6f}** [adimensional] y una FER de **{resumen['FER']:.6f}** [adimensional]. El CRC detectó **{int(resumen['Tramas detectadas por CRC'])}** tramas como alteradas.

Los errores no detectados fueron **{int(resumen['Errores no detectados'])}**. Si este valor es cero, debe interpretarse como ausencia de eventos observados en la simulación, no como una garantía absoluta de que el CRC nunca falle.
"""
            )
        else:
            st.info("Ejecute primero una simulación interactiva.")

        if "g5_df_comparacion_v8" in st.session_state:
            st.subheader("Última comparación estadística")

            df_comp = st.session_state["g5_df_comparacion_v8"]
            mostrar_dataframe(df_comp)

            df_ordenado = df_comp.sort_values("σ")
            menor_sigma = df_ordenado.iloc[0]
            mayor_sigma = df_ordenado.iloc[-1]

            st.markdown(
                f"""
Con el menor valor de ruido evaluado, $\\sigma = {menor_sigma['σ']:.2f}$ [amplitud normalizada], la FER fue **{menor_sigma['FER']:.6f}** [adimensional] y la tasa de error no detectado fue **{menor_sigma['Tasa de error no detectado']:.6f}** [adimensional].

Con el mayor valor de ruido evaluado, $\\sigma = {mayor_sigma['σ']:.2f}$ [amplitud normalizada], la FER fue **{mayor_sigma['FER']:.6f}** [adimensional] y la tasa de error no detectado fue **{mayor_sigma['Tasa de error no detectado']:.6f}** [adimensional].

La tendencia esperada es que al aumentar $\\sigma$ aumente la dispersión del ruido, lo cual incrementa los errores del canal.
"""
            )
        else:
            st.info("Ejecute primero una comparación estadística.")

        st.subheader("Actividades guiadas")

        st.markdown(
            """
1. Calcule manualmente el CRC de una trama corta.
2. Identifique el polinomio $M(x)$ asociado al mensaje.
3. Identifique el polinomio generador $G(x)$.
4. Determine el grado $r$ del generador.
5. Agregue $r$ ceros al mensaje para formar $M(x)x^r$.
6. Realice la división módulo 2 usando XOR.
7. Verifique que el residuo obtenido coincida con la app.
8. Ejecute la simulación con pocos bits y observe las tablas de tramas.
9. Ejecute la comparación estadística con 5000 o más tramas.
10. Observe cómo cambia la FER al aumentar σ.
11. Observe cómo cambia la BER del canal al aumentar o disminuir la SNR.
12. Explique por qué cero errores no detectados no significa probabilidad real igual a cero.
13. Explique por qué CRC detecta errores, pero no los corrige.
"""
        )

        st.subheader("Preguntas de análisis")

        pregunta_1 = st.radio(
            "Pregunta 1: ¿Qué operación reemplaza a la resta en la división módulo 2?",
            [
                "Suma decimal",
                "XOR",
                "Multiplicación decimal",
                "Raíz cuadrada",
            ],
            index=None,
            key="g5_pregunta_1",
        )

        if pregunta_1:
            if pregunta_1 == "XOR":
                st.success("Correcto. En GF(2), la división CRC usa XOR como operación fundamental.")
            else:
                st.error("Revise la explicación de división módulo 2.")

        pregunta_2 = st.radio(
            "Pregunta 2: Si el residuo de verificación CRC es distinto de cero, ¿qué se concluye?",
            [
                "No se detecta error.",
                "Se detecta error.",
                "El CRC corrige automáticamente la trama.",
                "La SNR es infinita.",
            ],
            index=None,
            key="g5_pregunta_2",
        )

        if pregunta_2:
            if pregunta_2 == "Se detecta error.":
                st.success("Correcto. Residuo distinto de cero indica error detectado.")
            else:
                st.error("Revise el proceso de verificación CRC.")

        pregunta_3 = st.radio(
            "Pregunta 3: ¿CRC corrige errores por sí mismo?",
            [
                "Sí, siempre corrige todos los errores.",
                "No, CRC detecta errores pero no los corrige por sí mismo.",
                "Sí, pero solo si σ = 0.",
                "Solo corrige errores en el primer bit.",
            ],
            index=None,
            key="g5_pregunta_3",
        )

        if pregunta_3:
            if pregunta_3 == "No, CRC detecta errores pero no los corrige por sí mismo.":
                st.success("Correcto. CRC es un mecanismo de detección, no de corrección.")
            else:
                st.error("Revise la diferencia entre detección y corrección.")

        pregunta_4 = st.radio(
            "Pregunta 4: Si no se observan errores no detectados en N tramas, ¿qué interpretación es más adecuada?",
            [
                "La probabilidad real de error no detectado es exactamente cero.",
                "No se observaron errores no detectados; puede representarse como tasa < 1/N.",
                "El generador CRC deja de funcionar.",
                "El canal no tiene ruido.",
            ],
            index=None,
            key="g5_pregunta_4",
        )

        if pregunta_4:
            if pregunta_4 == "No se observaron errores no detectados; puede representarse como tasa < 1/N.":
                st.success("Correcto. Es una interpretación estadística más adecuada.")
            else:
                st.error("Revise la interpretación de métricas en escala logarítmica.")

        pregunta_5 = st.radio(
            "Pregunta 5: ¿Qué determina el grado r del generador G(x)?",
            [
                "La cantidad de bits CRC agregados.",
                "La cantidad de errores corregidos automáticamente.",
                "La amplitud del ruido.",
                "La SNR del canal.",
            ],
            index=None,
            key="g5_pregunta_5",
        )

        if pregunta_5:
            if pregunta_5 == "La cantidad de bits CRC agregados.":
                st.success("Correcto. Si G(x) tiene longitud L, entonces r = L - 1.")
            else:
                st.error("Revise la teoría del polinomio generador.")

    with tabs[6]:
        st.header("Conclusiones")

        st.markdown(
            """
Al finalizar esta guía, el estudiante debe concluir que:

- CRC es una técnica de detección de errores basada en redundancia.
- Las secuencias binarias pueden representarse como polinomios sobre GF(2).
- El polinomio generador $G(x)$ define la regla de cálculo y verificación del CRC.
- El generador debe iniciar en 1, terminar en 1 y ser conocido por transmisor y receptor.
- El grado $r$ del generador determina cuántos bits CRC se agregan.
- El residuo CRC se obtiene mediante división polinomial módulo 2.
- La operación fundamental de la división CRC es XOR.
- El transmisor envía los datos concatenados con el residuo.
- El receptor divide la trama recibida entre el mismo generador.
- Un residuo cero significa que no se detecta error.
- Un residuo distinto de cero significa que se detecta error.
- CRC no corrige errores por sí mismo.
- La capacidad de detección depende del generador y del patrón de error.
- BER mide errores de bit producidos por el canal.
- FER mide tramas alteradas.
- BER, FER y las tasas son magnitudes adimensionales.
- La SNR se expresa en dB.
- La desviación estándar del ruido σ se interpreta en amplitud normalizada.
- En gráficas logarítmicas, cero eventos observados debe interpretarse como un límite experimental.
- Evaluar muchas tramas permite obtener conclusiones estadísticas más confiables.

CRC es especialmente útil cuando se combina con otros mecanismos, como Hamming, ya que Hamming puede corregir errores simples y CRC puede detectar errores remanentes.
"""
        )

    with tabs[7]:
        st.header("Referencias")

        st.markdown(
            """
Forouzan, B. A. (2013). *Data communications and networking* (5th ed.). McGraw-Hill Education.

Lin, S., & Costello, D. J. (2004). *Error control coding* (2nd ed.). Pearson.

Peterson, W. W., & Brown, D. T. (1961). Cyclic codes for error detection. *Proceedings of the IRE, 49*(1), 228–235.

Proakis, J. G., & Salehi, M. (2008). *Digital communications* (5th ed.). McGraw-Hill.

Stallings, W. (2015). *Data and computer communications* (10th ed.). Pearson.

Tanenbaum, A. S., & Wetherall, D. J. (2011). *Computer networks* (5th ed.). Pearson.

Wicker, S. B., & Bhargava, V. K. (1999). *Reed-Solomon codes and their applications*. IEEE Press.
"""
        )