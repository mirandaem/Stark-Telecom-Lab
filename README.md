# Stark Telecom Lab

Proyecto base para un simulador educativo de transmisión digital con énfasis en:

- Codificación Hamming (7,4)
- CRC
- Canal BSC y AWGN educativo
- Decodificación y verificación en el receptor
- Guías de laboratorio en Jupyter Notebook

## Estructura

```text
stark-telecom-lab/
├── app/
├── src/
├── notebooks/
├── data/
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows use .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Ejecutar la app

```bash
streamlit run app/app.py
```

## Ejecutar pruebas

```bash
pytest
```

## Próximos pasos sugeridos

1. Añadir CRC al flujo completo del transmisor y receptor.
2. Crear notebooks por guía de laboratorio.
3. Agregar métricas BER y visualizaciones.
4. Integrar simulación BPSK + AWGN con gráficas.

## App disponible en: 
https://stark-telecom-lab.streamlit.app/
