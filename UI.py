import streamlit as st
from main import agent


st.sidebar.markdown("## Search and Get here")
st.sidebar.divider()
input = st.sidebar.text_input('Search Here')

st.markdown('## Blogging Deep Agent')
st.divider()
st.text(input)