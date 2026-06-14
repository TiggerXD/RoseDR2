import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="RoseDR",
    page_icon="🌹",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        to bottom,
        #f8fff8,
        #edf7ed
    );
}

.main-title {
    text-align: center;
    color: #2E7D32;
    font-size: 3.5rem;
    font-weight: bold;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #5E7D60;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}

.hero-box {
    background-color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

.result-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 8px solid #43A047;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-top: 15px;
}

[data-testid="stFileUploader"] {
    border: 3px dashed #66BB6A;
    border-radius: 15px;
    padding: 15px;
    background-color: rgba(255,255,255,0.8);
}

.stButton > button {
    background-color: #43A047;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #2E7D32;
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 class='main-title'>RoseDR</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>AI-Powered Rose Disease Detection System</p>",
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-box">
<h2>Welcome to RoseDR</h2>
<p>
Upload a rose leaf image and allow the AI model to identify
potential diseases affecting the plant.
</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "RoseDR_model",
        "RoseDR.pt"
    )
)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Failed to load model.")
    st.code(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a Rose Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Disease Analysis")

        if st.button("Analyze Image"):

            with st.spinner("Analyzing image..."):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as tmp:

                    image.save(tmp.name)

                    results = model(tmp.name)

            result = results[0]

            if len(result.boxes) == 0:

                st.warning("No disease detected.")

            else:

                best_box = max(
                    result.boxes,
                    key=lambda x: float(x.conf)
                )

                disease_id = int(best_box.cls.item())
                confidence = float(best_box.conf.item())

                disease_name = model.names[disease_id]

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3>Disease Detected</h3>
                        <h2>{disease_name}</h2>
                        <h4>Confidence: {confidence:.2%}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            st.subheader("Detection Result")

            annotated_image = result.plot()

            st.image(
                annotated_image,
                use_container_width=True
            )

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:#777777'>
    RoseDR - Rose Disease Recognition System
    <br>
    Powered by YOLOv26
    </div>
    """,
    unsafe_allow_html=True
)