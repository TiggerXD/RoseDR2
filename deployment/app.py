import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

#config page
st.set_page_config(
    page_title="RoseDR Disease Detection",
    page_icon="🌹",
    layout="centered"
)

st.title("🌹 RoseDR Disease Detection")
st.write("Upload an image of a rose leaf to detect diseases.")

#load model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "RoseDR_model",
    "RoseDR.pt"
)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


#upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Disease"):

        with st.spinner("Analyzing image..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                image.save(tmp.name)

                results = model(tmp.name)

            result = results[0]

        st.subheader("Prediction Results")

        if len(result.boxes) == 0:

            st.warning("No disease detected.")

        else:

            boxes = result.boxes

            for box in boxes:

                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())

                disease_name = model.names[cls_id]

                st.success(
                    f"**{disease_name}** "
                    f"({confidence*100:.2f}% confidence)"
                )

            annotated_image = result.plot()

            st.image(
                annotated_image,
                caption="Detected Disease(s)",
                use_container_width=True
            )