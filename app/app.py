"""Streamlit app for the full CRC + Hamming educational pipeline."""
from __future__ import annotations

import os
import sys

import streamlit as st

# Ensure the project root is importable when Streamlit runs from /app.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_pipeline


st.set_page_config(page_title="Stark Telecom Lab", layout="wide")
st.title("Stark Telecom Lab")
st.caption("Simulador educativo del flujo completo: CRC → Hamming → Canal → Receptor")

with st.sidebar:
    st.header("Configuración")
    input_bits = st.text_input("Bits de entrada", value="1011")
    generator = st.text_input("Polinomio CRC", value="1011")
    channel_type = st.selectbox(
        "Canal",
        options=["ideal", "manual", "bsc", "awgn"],
        format_func=lambda x: {
            "ideal": "Ideal (sin errores)",
            "manual": "Error manual",
            "bsc": "BSC (probabilidad de bit)",
            "awgn": "AWGN educativo (BPSK + decisión dura)",
        }[x],
    )

    manual_error_position = None
    bsc_p = 0.0
    awgn_sigma = 0.2

    if channel_type == "manual":
        manual_error_position = st.number_input(
            "Posición del error sobre la trama transmitida", min_value=1, value=3, step=1
        )
    elif channel_type == "bsc":
        bsc_p = st.slider("Probabilidad de error", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    elif channel_type == "awgn":
        awgn_sigma = st.slider("Sigma del ruido", min_value=0.0, max_value=2.0, value=0.2, step=0.05)

    seed = st.number_input("Semilla aleatoria", min_value=0, value=7, step=1)
    run = st.button("Ejecutar simulación", type="primary")

if run:
    try:
        result = run_pipeline(
            input_bits,
            generator=generator,
            channel_type=channel_type,
            manual_error_position=int(manual_error_position) if manual_error_position else None,
            bsc_p=bsc_p,
            awgn_sigma=awgn_sigma,
            seed=int(seed),
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Transmisor")
            st.write(f"**Mensaje original:** `{result.original_bits}`")
            st.write(f"**CRC añadido:** `{result.crc_codeword}`")
            st.write(f"**Bits de relleno:** `{result.padding_bits}`")
            st.write(f"**Secuencia para Hamming:** `{result.padded_crc_bits}`")
            st.write(f"**Bloques Hamming TX:** {result.hamming_blocks_tx}")
            st.write(f"**Trama transmitida:** `{result.transmitted_frame}`")

        with col2:
            st.subheader("Canal y receptor")
            st.write(f"**Trama recibida:** `{result.received_frame}`")
            st.write(f"**Síndromes:** {result.syndromes}")
            st.write(f"**Bloques corregidos:** {result.corrected_blocks}")
            st.write(f"**Bits decodificados (con padding):** `{result.decoded_padded_bits}`")
            st.write(f"**Bits decodificados (sin padding):** `{result.decoded_crc_bits}`")
            st.write(f"**CRC válido:** `{result.crc_valid}`")
            st.write(f"**Mensaje recuperado:** `{result.recovered_bits}`")

        st.subheader("Métricas")
        m1, m2 = st.columns(2)
        m1.metric("BER del canal", f"{result.ber_channel:.4f}")
        m2.metric("BER extremo a extremo", f"{result.ber_end_to_end:.4f}")

        with st.expander("Ver resultado completo"):
            st.json(result.to_dict())

    except ValueError as exc:
        st.error(str(exc))
else:
    st.info("Configura los parámetros y pulsa 'Ejecutar simulación'.")
