import streamlit as st

st.set_page_config(page_title='NBA MVP Prediction', page_icon = None, layout = 'centered', initial_sidebar_state = 'auto')
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)

st.title('NBA MVP Prediction')

# Labels
label_lowercaser = "Lowercaser"
label_uppercaser = "Uppercaser"

# Sidebar
selectbox = st.sidebar.selectbox(
    "Select a version...",
    ("v1", "v2")
)
radio = st.sidebar.radio(
    "Select a text transformer...",
    (label_lowercaser, label_uppercaser)
)

# Content
st.header(radio)
st.markdown('''
This is a demo. For creating the app,
we used [Streamlit](https://streamlit.io/).
''')
st.subheader(f"{radio} | {selectbox}")
input_text = st.text_input(label='Type some text.', value='This is an example of text.')
if radio==label_lowercaser:
    res = str(input_text).lower()
else:
    res = str(input_text).upper()
st.write(f'Transformed text : {res}')
st.write("Confidence : 80%")
st.progress(80)
age = st.slider('How old are you?', 0, 100, 25)
if age==80:
    st.balloons()
values = st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))
st.file_uploader("You can alway upload a file, but I am pretty confident nothing will happen.")