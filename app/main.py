from fastapi import FastAPI, Query
import pandas as pd
import numpy as np
import joblib
from typing import Optional

from fastapi.middleware.cors import CORSMiddleware


# =====================================================
# 1️⃣ Crear app FastAPI
# =====================================================
app = FastAPI(
    title="API Mantenimiento Preventivo",
    description="Modelo Random Forest para selección óptima de mantenimiento diario",
    version="1.0.0"
)

# =====================================================
# 2️⃣ Habilitar CORS (antes de cualquier endpoint)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para pruebas locales, acepta cualquier origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 2️⃣ Cargar modelo UNA sola vez al iniciar
# =====================================================
pipeline = joblib.load("app/modelo_rf_mantenimiento.pkl")

# =====================================================
# 3️⃣ Feature Engineering (idéntico al entrenamiento)
# =====================================================
def crear_features(df: pd.DataFrame):

    df = df.copy()
    df['FECHA'] = pd.to_datetime(df['FECHA'])

    # Variables calendario
    df['dia_semana'] = df['FECHA'].dt.weekday
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    df['semana_anio'] = df['FECHA'].dt.isocalendar().week.astype(int)
    df['mes'] = df['FECHA'].dt.month

    # Histórico mantenimiento
    df['fecha_mant'] = df['FECHA'].where(
        df['Seleccionado_para_Mantenimiento'] == 1
    )

    df['fecha_mant_ffill'] = (
        df.groupby('Número interno unidad')['fecha_mant'].ffill()
    )

    df['fecha_penultimo'] = (
        df.groupby('Número interno unidad')['fecha_mant']
        .shift(1).ffill()
    )

    df['dias_desde_ultimo_mantenimiento_raw'] = (
        df['FECHA'] - df['fecha_mant_ffill']
    ).dt.days.fillna(0).clip(lower=0)

    df['dias_desde_penultimo'] = (
        df.groupby('Número interno unidad')
        ['dias_desde_ultimo_mantenimiento_raw']
        .shift(1)
        .fillna(0)
    )

    # Estadísticas históricas
    df['intervalo_real'] = (
        df.groupby('Número interno unidad')['fecha_mant']
        .diff().dt.days
    )

    df['prom_intervalo_hist'] = (
        df.groupby('Número interno unidad')['intervalo_real']
        .expanding().mean()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df['std_intervalo_hist'] = (
        df.groupby('Número interno unidad')['intervalo_real']
        .expanding().std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df['mnts_ult_30d'] = (
        df.groupby('Número interno unidad')
        ['Seleccionado_para_Mantenimiento']
        .rolling(window=30, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df['mnts_ult_60d'] = (
        df.groupby('Número interno unidad')
        ['Seleccionado_para_Mantenimiento']
        .rolling(window=60, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df['ratio_reciente_vs_hist'] = (
        df['mnts_ult_30d'] / (df['prom_intervalo_hist'] + 1)
    )

    features = [
        'tipologia_num',
        'dia_semana',
        'es_fin_semana',
        'semana_anio',
        'mes',
        'dias_desde_penultimo',
        'prom_intervalo_hist',
        'std_intervalo_hist',
        'mnts_ult_30d',
        'mnts_ult_60d',
        'ratio_reciente_vs_hist'
    ]

    return df, features


# =====================================================
# 4️⃣ Función de predicción diaria
# =====================================================
def predecir_mantenimiento_diario(
    df_historico: pd.DataFrame,
    fecha_prediccion: str,
    cupos: dict
):

    fecha_prediccion = pd.to_datetime(fecha_prediccion)

    # Usar solo histórico hasta la fecha seleccionada
    df_hasta_hoy = df_historico[
        df_historico['FECHA'] <= fecha_prediccion
    ]

    df_features, feature_cols = crear_features(df_hasta_hoy)

    # Tomar última fila por vehículo
    df_ultimos = (
        df_features
        .groupby('Número interno unidad')
        .last()
        .reset_index()
    )

    X_pred = df_ultimos[feature_cols]

    # Predicción
    df_ultimos['prob_mantenimiento'] = (
        pipeline.predict_proba(X_pred)[:, 1]
    )

    df_ultimos['tipologia'] = df_ultimos['tipologia_num']

    # Aplicar cupos
    seleccion = []

    for tipo, k in cupos.items():
        df_tipo = df_ultimos[df_ultimos['tipologia'] == tipo]
        top_k = (
            df_tipo
            .sort_values('prob_mantenimiento', ascending=False)
            .head(k)
        )
        seleccion.append(top_k)

    df_final = pd.concat(seleccion).reset_index(drop=True)
    df_final['FECHA'] = fecha_prediccion

    return df_final


# =====================================================
# 5️⃣ Endpoint DEMO para portafolio
# =====================================================
@app.get("/demo")
def demo(
    fecha: Optional[str] = Query(
        None,
        description="Fecha para predicción (YYYY-MM-DD)"
    ),
    cupo_tipo_1: int = 35,
    cupo_tipo_2: int = 5,
    cupo_tipo_3: int = 10
):
    """
    Endpoint interactivo para portafolio.
    Usa dataset demo interno.
    """

    # Dataset fijo demo
    df_historico = pd.read_csv(
        "app/data/mantenimiento_vehiculos_2024.csv"
    )
    df_historico['FECHA'] = pd.to_datetime(
        df_historico['FECHA']
    )

    # Si no envían fecha → usar última disponible
    if fecha is None:
        fecha_prediccion = df_historico['FECHA'].max()
    else:
        fecha_prediccion = fecha

    cupos = {
        1: cupo_tipo_1,
        2: cupo_tipo_2,
        3: cupo_tipo_3
    }

    resultado = predecir_mantenimiento_diario(
        df_historico=df_historico,
        fecha_prediccion=fecha_prediccion,
        cupos=cupos
    )

    # Limpiar valores no JSON-compatibles
    resultado = resultado.replace(
        [np.inf, -np.inf], 0
    ).fillna(0)

    # Respuesta limpia para frontend
    return resultado[
        [
            'Número interno unidad',
            'tipologia',
            'prob_mantenimiento',
            'FECHA'
        ]
    ].to_dict(orient="records")
