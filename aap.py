import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


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

st.markdown(
    """
    ### Random Forest Based Prediction

    Enter the soil index properties below to predict:

    - CBR Unsoaked
    - CBR Soaked
    - OMC
    - MDD
    - Cohesion (C)
    - Angle of Internal Friction (φ)
    - UCS
    """
)


# ============================================================
# MODEL FILE
# ============================================================

MODEL_FILE = "Clayey_Soil_RF_Models.joblib"


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        "Model file not found. "
        "Please upload Clayey_Soil_RF_Models.joblib "
        "to the same GitHub repository as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_FILE)


try:

    model_package = load_model()

    models = model_package["models"]

    input_features = model_package["inputs"]

    output_features = model_package["outputs"]

except Exception as e:

    st.error(
        f"Error loading model: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Soil Input Parameters")

st.sidebar.markdown(
    "Enter the laboratory/index properties of the soil."
)


# ============================================================
# INPUTS
# ============================================================

LL = st.sidebar.number_input(
    "Liquid Limit, LL (%)",
    min_value=0.0,
    max_value=200.0,
    value=52.0,
    step=0.1
)


PL = st.sidebar.number_input(
    "Plastic Limit, PL (%)",
    min_value=0.0,
    max_value=150.0,
    value=28.0,
    step=0.1
)


PI = st.sidebar.number_input(
    "Plasticity Index, PI (%)",
    min_value=0.0,
    max_value=150.0,
    value=24.0,
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
    value=2.68,
    step=0.01
)


# ============================================================
# INPUT TABLE
# ============================================================

input_data = pd.DataFrame({

    "LL": [LL],

    "PL": [PL],

    "PI": [PI],

    "w": [w],

    "Gs": [Gs]

})


st.subheader("Input Soil Properties")

st.dataframe(
    input_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Engineering Properties",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        predictions = {}

        for output in output_features:

            model = models[output]

            prediction = model.predict(
                input_data[input_features]
            )[0]

            predictions[output] = prediction


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.subheader(
            "🎯 Predicted Engineering Properties"
        )


        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # CBR UNSOAKED
        # ----------------------------------------------------

        with col1:

            st.metric(
                "CBR Unsoaked (%)",
                f"{predictions['CBR_Unsoaked']:.2f}"
            )


        # ----------------------------------------------------
        # CBR SOAKED
        # ----------------------------------------------------

        with col2:

            st.metric(
                "CBR Soaked (%)",
                f"{predictions['CBR_Soaked']:.2f}"
            )


        # ----------------------------------------------------
        # OMC
        # ----------------------------------------------------

        with col3:

            st.metric(
                "OMC (%)",
                f"{predictions['OMC']:.2f}"
            )


        col4, col5, col6, col7 = st.columns(4)


        # ----------------------------------------------------
        # MDD
        # ----------------------------------------------------

        with col4:

            st.metric(
                "MDD (kg/m³)",
                f"{predictions['MDD']:.2f}"
            )


        # ----------------------------------------------------
        # COHESION
        # ----------------------------------------------------

        with col5:

            st.metric(
                "Cohesion C (kPa)",
                f"{predictions['C']:.2f}"
            )


        # ----------------------------------------------------
        # PHI
        # ----------------------------------------------------

        with col6:

            st.metric(
                "Phi (°)",
                f"{predictions['Phi']:.2f}"
            )


        # ----------------------------------------------------
        # UCS
        # ----------------------------------------------------

        with col7:

            st.metric(
                "UCS (kPa)",
                f"{predictions['UCS']:.2f}"
            )


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        st.subheader(
            "📊 Prediction Summary"
        )


        results = pd.DataFrame({

            "Engineering Property": [

                "CBR Unsoaked",

                "CBR Soaked",

                "OMC",

                "MDD",

                "Cohesion",

                "Phi",

                "UCS"

            ],

            "Predicted Value": [

                predictions["CBR_Unsoaked"],

                predictions["CBR_Soaked"],

                predictions["OMC"],

                predictions["MDD"],

                predictions["C"],

                predictions["Phi"],

                predictions["UCS"]

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


        st.dataframe(

            results.style.format({

                "Predicted Value":
                    "{:.3f}"

            }),

            use_container_width=True,

            hide_index=True

        )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        csv_data = results.to_csv(
            index=False
        )


        st.download_button(

            label="⬇️ Download Prediction Results",

            data=csv_data,

            file_name="soil_prediction_results.csv",

            mime="text/csv",

            use_container_width=True

        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

st.subheader("ℹ️ Model Information")

st.write(
    "Model: Random Forest Regressor"
)

st.write(
    "Input variables: LL, PL, PI, Natural Water Content and Gs"
)

st.write(
    "Predicted outputs: CBR, OMC, MDD, C, Phi and UCS"
)

st.caption(
    "This application provides model-based predictions and "
    "should be used together with appropriate laboratory "
    "testing and engineering judgment."
)
