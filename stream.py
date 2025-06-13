import streamlit as st
import pandas as pd
from collate_results import make_model_df, average_across_langs, average_across_tests

model, avg_lang, avg_test = st.tabs(["Single Model", "Average Across Languages", "Average Across Test Sets"])

# create dicts for model names paired with paths
nospace_model_dirs = {"BERT": "./results/bert_results_nospace", "CANINE-c": "./results/canine_results_nospace",
                      "CANINE-s": "./results/canine-s_results_nospace"}
space_model_dirs = {"BERT": "./results/bert_results_2", "CANINE-c": "./results/canine_results_2", "CANINE-s": "./results/canine-s_results_2"}

with model:
    st.header("Individual Model Results")
    model_names = ["BERT Base Cased", "CANINE-c", "CANINE-s"]
    model_selection = st.selectbox(label="Pick a model:", options=model_names)
    match model_selection:
        case "BERT Base Cased":
            df = make_model_df("results/bert_results_nospace")
            df2 = make_model_df("results/bert_results_2")
            st.subheader("BERT without spaces")
            st.dataframe(df)
            st.subheader("With spaces")
            st.dataframe(df2)
        case "CANINE-c":
            df = make_model_df("results/canine_results_nospace")
            df2 = make_model_df("results/canine_results_2")
            st.subheader("CANINE-c without spaces")
            st.dataframe(df)
            st.subheader("With spaces")
            st.dataframe(df2)
        case "CANINE-s":
            df = make_model_df("results/canine-s_results_nospace")
            df2 = make_model_df("results/canine-s_results_2")
            st.subheader("CANINE-s without spaces")
            st.dataframe(df)
            st.subheader("With spaces")
            st.dataframe(df2)

with avg_lang:
    st.header("Without spaces")
    df = average_across_langs(nospace_model_dirs)
    st.dataframe(df)
    st.header("With spaces")
    df2 = average_across_langs(space_model_dirs)
    st.dataframe(df2)

with avg_test:
    st.header("Without spaces")
    df = average_across_tests(nospace_model_dirs)
    st.dataframe(df)
    st.header("With spaces")
    df2 = average_across_tests(space_model_dirs)
    st.dataframe(df2)
