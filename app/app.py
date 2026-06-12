import streamlit as st

from guides.guia_01_ruido_tiempo import render_guia_01
from guides.guia_02_ber_snr import render_guia_02
from guides.guia_03_hamming_tx import render_guia_03
from guides.guia_04_hamming_rx import render_guia_04
from guides.guia_05_crc import render_guia_05
from guides.guia_06_sistema_completo import render_guia_06

st.set_page_config(
    page_title="Stark Telecom Lab",
    page_icon="ST",
    layout="wide",
)

def aplicar_estilo_academico() -> None:
    st.markdown(
        """
        <style>
        /* Fondo general */
        .stApp {
            background-color: #F7F9FB;
            color: #1C1C1C;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0B1F3A;
        }

        section[data-testid="stSidebar"] * {
            color: #F7F9FB !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 0.35rem 0.5rem;
            margin-bottom: 0.25rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: rgba(255, 255, 255, 0.16);
        }

        /* Títulos */
        h1 {
            color: #0B1F3A;
            border-bottom: 3px solid #556B2F;
            padding-bottom: 0.4rem;
        }

        h2, h3 {
            color: #1F4E79;
        }

        /* Pestañas fijas arriba */
        div[data-testid="stTabs"] div[role="tablist"] {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: #F7F9FB;
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #D8E3EC;
        }

        div[data-testid="stTabs"] div[role="tablist"] button {
            white-space: nowrap;
            color: #0B1F3A;
            border-radius: 8px 8px 0 0;
        }

        div[data-testid="stTabs"] div[role="tablist"] button[aria-selected="true"] {
            background-color: #EAF3FA;
            color: #0B1F3A;
            border-bottom: 3px solid #556B2F;
            font-weight: 700;
        }

        /* Botones */
        .stButton > button {
            background-color: #1F4E79;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.45rem 1rem;
            font-weight: 600;
        }

        .stButton > button:hover {
            background-color: #0B1F3A;
            color: white;
            border: none;
        }

        /* Métricas */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border-left: 5px solid #556B2F;
            padding: 0.8rem;
            border-radius: 10px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
        }

        div[data-testid="stMetricLabel"] {
            color: #1F4E79;
            font-weight: 700;
        }

        /* Cajas informativas */
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        /* Dataframes y tablas */
        div[data-testid="stDataFrame"] {
            border: 1px solid #D8E3EC;
            border-radius: 10px;
        }

        /* Código */
        code {
            color: #0B1F3A;
        }

        pre {
            border-left: 5px solid #556B2F;
            border-radius: 8px;
        }

        /* Separadores */
        hr {
            border: none;
            border-top: 1px solid #D8E3EC;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def fijar_pestanas_superiores() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] div[role="tablist"] {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: var(--background-color);
            padding-top: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        }

        div[data-testid="stTabs"] div[role="tablist"] button {
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
aplicar_estilo_academico()
fijar_pestanas_superiores()

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
elif guia == "Guía 5 - CRC":
    render_guia_05()
elif guia == "Guía 6 - Sistema completo":
    render_guia_06()
else:
    st.title(guia)
    st.info(
        "Este módulo será desarrollado en la siguiente etapa del proyecto. "
        "La plataforma mantendrá la estructura de objetivos, teoría, simulación, "
        "dinámica, análisis, conclusiones y referencias."
    )