import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Clayey Soil AI Predictor",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Clayey Soil Engineering Properties Predictor")

st.write(
    "Enter the laboratory/index properties of the soil."
)


# ============================================================
# LOAD TRAINED RANDOM FOREST MODELS
# ============================================================

MODEL_FILE = "Clayey_Soil_RF_Models.joblib"


@st.cache_resource
def load_models():

    package = joblib.load(MODEL_FILE)

    return package


try:

    package = load_models()

    models = package["models"]

    INPUTS = package["inputs"]

    OUTPUTS = package["outputs"]

except Exception as e:

    st.error(
        f"Unable to load the trained model: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Soil Input Properties")


LL = st.sidebar.number_input(
    "Liquid Limit, LL (%)",
    min_value=0.0,
    max_value=200.0,
    value=50.0,
    step=0.1
)


PL = st.sidebar.number_input(
    "Plastic Limit, PL (%)",
    min_value=0.0,
    max_value=200.0,
    value=25.0,
    step=0.1
)


PI = st.sidebar.number_input(
    "Plasticity Index, PI (%)",
    min_value=0.0,
    max_value=200.0,
    value=25.0,
    step=0.1
)


w = st.sidebar.number_input(
    "Natural Water Content, w (%)",
    min_value=0.0,
    max_value=200.0,
    value=18.0,
    step=0.1
)


Gs = st.sidebar.number_input(
    "Specific Gravity, Gs",
    min_value=1.0,
    max_value=4.0,
    value=2.67,
    step=0.01
)


# ============================================================
# INPUT DATAFRAME
# ============================================================

input_values = {

    "LL (%)": LL,

    "PL (%)": PL,

    "PI (%)": PI,

    "w (%)": w,

    "Gs (-)": Gs
}


X = pd.DataFrame(
    [input_values]
)


# ============================================================
# DISPLAY INPUT PROPERTIES
# ============================================================

st.subheader("Input Soil Properties")

display_df = pd.DataFrame({

    "LL": [LL],

    "PL": [PL],

    "PI": [PI],

    "w": [w],

    "Gs": [Gs]

})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Engineering Properties",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # Ensure exact feature order used during training
        # ----------------------------------------------------

        X_model = X[INPUTS]


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = {}


        for target in OUTPUTS:

            model = models[target]

            prediction = model.predict(
                X_model
            )[0]

            predictions[target] = float(
                prediction
            )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.success(
            "Prediction completed successfully!"
        )


        st.subheader(
            "🎯 Predicted Engineering Properties"
        )


        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # CBR
        # ----------------------------------------------------

        with col1:

            st.metric(
                "CBR Unsoaked (%)",
                f"{predictions['CBR_unsoaked (%)']:.2f}"
            )

            st.metric(
                "CBR Soaked (%)",
                f"{predictions['CBR_soaked (%)']:.2f}"
            )


        # ----------------------------------------------------
        # COMPACTION
        # ----------------------------------------------------

        with col2:

            st.metric(
                "OMC (%)",
                f"{predictions['OMC (%)']:.2f}"
            )

            st.metric(
                "MDD (kg/m³)",
                f"{predictions['MDD (kg/m³)']:.2f}"
            )


        # ----------------------------------------------------
        # SHEAR STRENGTH
        # ----------------------------------------------------

        with col3:

            st.metric(
                "Cohesion, c (kPa)",
                f"{predictions['c (kPa)']:.2f}"
            )

            st.metric(
                "Phi (°)",
                f"{predictions['phi (deg)']:.2f}"
            )


        # ----------------------------------------------------
        # UCS
        # ----------------------------------------------------

        st.metric(
            "UCS (kPa)",
            f"{predictions['UCS (kPa)']:.2f}"
        )


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        results_table = pd.DataFrame({

            "Engineering Property": [

                "CBR Unsoaked",

                "CBR Soaked",

                "OMC",

                "MDD",

                "Cohesion (c)",

                "Phi",

                "UCS"

            ],

            "Predicted Value": [

                predictions["CBR_unsoaked (%)"],

                predictions["CBR_soaked (%)"],

                predictions["OMC (%)"],

                predictions["MDD (kg/m³)"],

                predictions["c (kPa)"],

                predictions["phi (deg)"],

                predictions["UCS (kPa)"]

            ],

            "Unit": [

                "%",

                "%",

                "%",

                "kg/m³",

                "kPa",

                "degree",

                "kPa"

            ]

        })


        st.subheader(
            "📊 Prediction Summary"
        )


        st.dataframe(
            results_table,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "ℹ️ Model Information"
)

st.write(
    "The application uses Random Forest regression models "
    "trained using clayey-soil index properties."
)

st.write(
    "**Input variables:**"
)

st.write(
    ", ".join(INPUTS)
)

st.write(
    "**Predicted engineering properties:**"
)

st.write(
    ", ".join(OUTPUTS)
)
