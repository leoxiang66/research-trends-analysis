import streamlit as st
from widgets import *




# sidebar content
platforms, number_papers, start_year, end_year, hyperparams = render_sidebar()

# body head
with st.form("my_form",clear_on_submit=False):
    st.markdown('''# 👋 Hi, enter your query here :)''')
    query_input = st.text_input(
        'Enter your query:',
        placeholder='''e.g. "Machine learning"''',
        # label_visibility='collapsed',
        value=''
    )

    show_preview = st.checkbox('show paper preview')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Search")


if submitted:
    # body
    render_body(platforms, number_papers, 5, query_input,
                show_preview, start_year, end_year,
                hyperparams,
                hyperparams['standardization'])






