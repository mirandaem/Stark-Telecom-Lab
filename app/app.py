import streamlit as st

from guides.guia_01_ruido_tiempo import render_guia_01
from guides.guia_02_ber_snr import render_guia_02
from guides.guia_03_hamming_tx import render_guia_03
from guides.guia_04_hamming_rx import render_guia_04

st.set_page_config(
    page_title="Stark Telecom Lab",
    page_icon="ST",
    layout="wide",
)

st.sidebar.title("Stark Telecom Lab")

st.sidebar.markdown(
    """
Plataforma interactiva para el estudio de detección y corrección básica de errores
en sistemas de telecomunicaciones digitales.
"""
)

guia = st.sidebar.radio(
    "Seleccione una guía",
    [
        "Guía 1 - Canal y ruido en el tiempo",
        "Guía 2 - BER, SNR y estadísticas",
        "Guía 3 - Hamming Tx",
        "Guía 4 - Hamming Rx",
        "Guía 5 - CRC",
        "Guía 6 - Sistema completo",
    ],
)

if guia == "Guía 1 - Canal y ruido en el tiempo":
    render_guia_01()
elif guia == "Guía 2 - BER, SNR y estadísticas":
    render_guia_02()
elif guia == "Guía 3 - Hamming Tx":
    render_guia_03()
elif guia == "Guía 4 - Hamming Rx":
    render_guia_04()
else:
    st.title(guia)
    st.info(
        "Este módulo será desarrollado en la siguiente etapa del proyecto. "
        "La plataforma mantendrá la estructura de objetivos, teoría, simulación, "
        "dinámica, análisis, conclusiones y referencias."
    )