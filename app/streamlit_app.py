"""Thin Streamlit UI. All ML logic belongs in src/."""
import streamlit as st

st.set_page_config(page_title="EchoTrace", page_icon="🔊", layout="wide")
st.title("EchoTrace")
st.caption("Search environmental audio using natural-language descriptions.")
st.info("The audio indexing and retrieval pipeline is under construction.")
