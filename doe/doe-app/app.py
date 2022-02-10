import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import fun_doe as fd
import re


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.set_page_config(layout="wide")
# collect_numbers = lambda x: [int(i) for i in re.split("[^0-9]", x) if i != ""]

st.title("Monitoring")
st.sidebar.title("Design of Experiments")


# Parameters Setup ------------------------------------------------------------------------------------------------
st.sidebar.title("Parameters")
params_form = st.sidebar.form(key="params")
params_extender = params_form.expander("Set")
with params_extender:
    experiments = params_extender.number_input("Experiments", min_value=1, max_value=50, value=12)
    features = params_extender.number_input("Features", min_value=1, max_value=10, value=4)
    levels = params_extender.multiselect("Enter Levels of factors",
                                         options=[-1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.],
                                         default=[-1., 1.])
    model_order = params_extender.number_input("Model Order", min_value=1, max_value=6, value=1)
    epochs = params_extender.number_input("Random Starts", min_value=1, max_value=50, value=10)
    col1, col2 = params_extender.columns(2)
    interactions_only = col1.checkbox("Interactions only?")
    bias = col2.checkbox("Include bias?")
submit_params_btn = params_form.form_submit_button("Submit")

model, design, design_hist, optimality_hist = fd.coordinate_exchange(experiments=experiments,
                                                                     features=features,
                                                                     epochs=epochs,
                                                                     levels=levels,
                                                                     model_order=model_order,
                                                                     interactions_only=interactions_only,
                                                                     bias=bias)

# Main page ------------------------------------------------------------------------------------------------
col1, col2 = st.columns(2)
design_expander = col1.expander("Design")
with design_expander:
    design_expander.write(design)
    design_expander.write(np.linalg.det(model.T @ model))
model_expander = col1.expander("Model")
with model_expander:
    st.write(model)

history_expander = col2.expander("History")
