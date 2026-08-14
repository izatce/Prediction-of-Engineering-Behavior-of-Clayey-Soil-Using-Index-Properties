import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


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
    "Random Forest-based prediction with Explainable AI (SHAP)"
)

st.info(
    "Enter the laboratory/index properties of the soil "
    "to predict engineering properties and interpret the "
    "model prediction using SHAP."
)


# ============================================================
# LOAD TRAINED MODEL
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
# SIDEBAR
# ============================================================

st.sidebar.header("🧪 Soil Input Properties")


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
# CREATE INPUT DATAFRAME
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


# Make absolutely sure the order is the same
# as used during Random Forest training

X_model = X[INPUTS]


# ============================================================
# DISPLAY INPUTS
# ============================================================

st.subheader("📋 Input Soil Properties")


display_df = pd.DataFrame({

    "LL (%)": [LL],

    "PL (%)": [PL],

    "PI (%)": [PI],

    "w (%)": [w],

    "Gs": [Gs]

})


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Engineering Properties",
    use_container_width=True
):

    try:

        predictions = {}


        # ====================================================
        # MAKE PREDICTIONS
        # ====================================================

        for target in OUTPUTS:

            model = models[target]

            prediction = model.predict(
                X_model
            )[0]

            predictions[target] = float(
                prediction
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "Prediction completed successfully!"
        )


        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.subheader(
            "🎯 Predicted Engineering Properties"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "CBR Unsoaked (%)",
                f"{predictions['CBR_unsoaked (%)']:.2f}"
            )

            st.metric(
                "CBR Soaked (%)",
                f"{predictions['CBR_soaked (%)']:.2f}"
            )


        with col2:

            st.metric(
                "OMC (%)",
                f"{predictions['OMC (%)']:.2f}"
            )

            st.metric(
                "MDD (kg/m³)",
                f"{predictions['MDD (kg/m³)']:.2f}"
            )


        with col3:

            st.metric(
                "Cohesion, c (kPa)",
                f"{predictions['c (kPa)']:.2f}"
            )

            st.metric(
                "Phi (°)",
                f"{predictions['phi (deg)']:.2f}"
            )


        st.metric(
            "UCS (kPa)",
            f"{predictions['UCS (kPa)']:.2f}"
        )


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        st.subheader(
            "📊 Prediction Summary"
        )


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


        st.dataframe(
            results_table,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SHAP INTERPRETATION
        # ====================================================

        st.divider()

        st.header(
            "🔍 Explainable AI — SHAP Interpretation"
        )


        st.write(
            "SHAP (SHapley Additive exPlanations) explains "
            "how each input soil property contributes to "
            "the prediction of each engineering property."
        )


        st.info(
            "Positive SHAP contribution pushes the prediction "
            "higher, while negative SHAP contribution pushes "
            "the prediction lower relative to the model's "
            "baseline prediction."
        )


        # ====================================================
        # SHAP FOR EACH OUTPUT
        # ====================================================

        for target in OUTPUTS:

            st.subheader(
                f"🔬 SHAP Explanation: {target}"
            )


            model = models[target]


            try:

                # ------------------------------------------------
                # TreeExplainer for Random Forest
                # ------------------------------------------------

                explainer = shap.TreeExplainer(
                    model
                )


                shap_values = explainer.shap_values(
                    X_model
                )


                # ------------------------------------------------
                # Handle different SHAP output formats
                # ------------------------------------------------

                if isinstance(
                    shap_values,
                    list
                ):

                    shap_values_array = np.array(
                        shap_values[0]
                    )

                else:

                    shap_values_array = np.array(
                        shap_values
                    )


                shap_values_array = (
                    shap_values_array.reshape(-1)
                )


                # ------------------------------------------------
                # SHAP dataframe
                # ------------------------------------------------

                shap_df = pd.DataFrame({

                    "Feature": INPUTS,

                    "SHAP Value":
                        shap_values_array,

                    "Absolute SHAP":
                        np.abs(
                            shap_values_array
                        )

                })


                shap_df = shap_df.sort_values(
                    "Absolute SHAP",
                    ascending=False
                )


                # ------------------------------------------------
                # Display SHAP table
                # ------------------------------------------------

                st.dataframe(

                    shap_df[
                        [
                            "Feature",
                            "SHAP Value"
                        ]
                    ],

                    use_container_width=True,

                    hide_index=True
                )


                # ------------------------------------------------
                # SHAP BAR CHART
                # ------------------------------------------------

                chart_df = shap_df.sort_values(
                    "SHAP Value"
                )


                fig, ax = plt.subplots(
                    figsize=(8, 4.5)
                )


                ax.barh(

                    chart_df["Feature"],

                    chart_df["SHAP Value"]
                )


                ax.axvline(
                    0,
                    linestyle="--"
                )


                ax.set_xlabel(
                    "SHAP Value"
                )


                ax.set_ylabel(
                    "Input Soil Property"
                )


                ax.set_title(
                    f"Feature Contribution to {target}"
                )


                plt.tight_layout()


                st.pyplot(
                    fig
                )


                plt.close(fig)


                # ------------------------------------------------
                # Most influential feature
                # ------------------------------------------------

                most_important = shap_df.iloc[0]

                feature_name = (
                    most_important["Feature"]
                )

                shap_value = (
                    most_important["SHAP Value"]
                )


                if shap_value > 0:

                    interpretation = (
                        f"**{feature_name}** has the strongest "
                        f"influence for this prediction and "
                        f"pushes the model prediction upward "
                        f"(SHAP = {shap_value:.4f})."
                    )

                elif shap_value < 0:

                    interpretation = (
                        f"**{feature_name}** has the strongest "
                        f"influence for this prediction and "
                        f"pushes the model prediction downward "
                        f"(SHAP = {shap_value:.4f})."
                    )

                else:

                    interpretation = (
                        f"**{feature_name}** has the strongest "
                        f"influence, although its SHAP "
                        f"contribution is approximately zero."
                    )


                st.write(
                    interpretation
                )


            except Exception as shap_error:

                st.warning(
                    f"SHAP interpretation could not be "
                    f"generated for {target}: {shap_error}"
                )


        # ====================================================
        # XAI CONCLUSION
        # ====================================================

        st.divider()

        st.subheader(
            "📌 XAI Interpretation"
        )

        st.write(
            "The SHAP analysis identifies the relative "
            "contribution of LL, PL, PI, natural water "
            "content, and specific gravity to each predicted "
            "engineering property."
        )

        st.write(
            "This provides an interpretable view of the "
            "Random Forest model rather than relying only "
            "on prediction accuracy."
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

st.write(
    "**Explainable AI:** SHAP-based local feature contribution"
)
