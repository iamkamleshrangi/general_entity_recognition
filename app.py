import streamlit as st
from gliner import GLiNER
import random
import asyncio
import nest_asyncio
nest_asyncio.apply()

def load_model():
    return GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

model = load_model()
st.title("Named Entity Recognition")
st.markdown("""Enter comma-separated entity labels below (e.g., `Location, Country, City`) on anything else you can think of""")

label_input = st.text_input("Entity labels:", value="Location, Country, City")
input_text = st.text_area("Input TEXT", "The capital of France is Paris.")

if st.button("Extract Entities"):
    labels = [label.strip() for label in label_input.split(",") if label.strip()]
    entities = model.predict_entities(input_text, labels, threshold=0.5)

    def get_random_color():
        h = random.randint(0, 360)
        s = 70 + random.randint(0, 20)
        l = 70 + random.randint(0, 10)
        return f"hsl({h}, {s}%, {l}%)"

    label_colors = {}
    entities = sorted(entities, key=lambda e: e["start"])
    highlighted_text = ""
    last_idx = 0

    for entity in entities:
        start, end = entity["start"], entity["end"]
        label = entity["label"].upper()
        if label not in label_colors:
            label_colors[label] = get_random_color()
        color = label_colors[label]

        highlighted_text += input_text[last_idx:start]
        highlighted_text += (
            f"<span style='background-color: {color}; border-radius: 6px; "
            f"padding: 2px 6px; margin: 0 2px; font-weight: 500;'>"
            f"{input_text[start:end]} "
            f"<span style='background: rgba(255,255,255,0.3); padding: 1px 6px; "
            f"border-radius: 4px; font-size: 0.75em; font-weight: 600;'>{label}</span></span>"
        )
        last_idx = end

    highlighted_text += input_text[last_idx:]
    st.markdown("### Highlighted Entities", unsafe_allow_html=True)
    st.markdown(f"<div style='line-height: 1.9;'>{highlighted_text}</div>", unsafe_allow_html=True)
