import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.space(50)

st.set_page_config(
    page_title="RoseDR",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        180deg,
        #1f3b2d 0%,
        #2b4a39 50%,
        #365c47 100%
    );
}

.block-container {
    max-width: 1050px;
    padding-top: 1rem;
}

.main-title {
    text-align: center;
    color: #f5fff5;
    font-size: 6rem;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #d9e8d9;
    font-size: 1.4rem;
    margin-bottom: 2rem;
}

.hero-box {
    background: rgba(255,255,255,0.95);
    color: #1f1f1f;
    border-radius: 18px;
    padding: 25px;
    max-width: 850px;
    margin: auto;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

.result-card {
    background: rgba(255,255,255,0.97);
    color: #1f1f1f;
    border-left: 8px solid #43A047;
    border-radius: 15px;
    padding: 20px;
    max-width: 650px;
    margin: auto;
    margin-top: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.95);
    border: 2px solid #6abf69;
    border-radius: 15px;
    padding: 15px;
}

.stButton > button {
    width: 100%;
    height: 3.2rem;
    border-radius: 12px;
    border: none;
    background-color: #43A047;
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #2E7D32;
    color: white;
}

h1, h2, h3, h4, h5, h6 {
    color: white;
}

[data-testid="stImage"] {
    border-radius: 12px;
}

.left-vine {
    position: fixed;
    left: 0;
    top: 0;
    width: 70px;
    height: 100vh;
    background:
        repeating-linear-gradient(
            180deg,
            rgba(120,180,120,0.25) 0px,
            rgba(120,180,120,0.25) 12px,
            transparent 12px,
            transparent 30px
        );
    z-index: -1;
}

.right-vine {
    position: fixed;
    right: 0;
    top: 0;
    width: 70px;
    height: 100vh;
    background:
        repeating-linear-gradient(
            180deg,
            rgba(120,180,120,0.25) 0px,
            rgba(120,180,120,0.25) 12px,
            transparent 12px,
            transparent 30px
        );
    z-index: -1;
}

footer {
    visibility: hidden;
}

</style>

<div class="left-vine"></div>
<div class="right-vine"></div>

""", unsafe_allow_html=True)

st.markdown(
    "<h1 class='main-title'>RoseDR</h1>",
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero-box">
<h2 style="color:#1f1f1f;">Welcome to RoseDR</h2>
<p>
Upload an image of a rose leaf and allow the model to analyze
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

CLASSIFIER_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "ImageSelectionModel",
        "rose_classifier.pth"
    )
)

import torch
import torch.nn as nn
from torchvision import transforms, models

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_classifier():

    classifier = models.resnet34(weights=None)

    classifier.fc = nn.Linear(
        classifier.fc.in_features,
        2
    )

    classifier.load_state_dict(
        torch.load(
            CLASSIFIER_PATH,
            map_location="cpu"
        )
    )

    classifier.eval()

    return classifier

try:
    model = load_model()
    classifier = load_classifier()

except Exception as e:
    st.error("Failed to load model")
    st.code(str(e))
    st.stop()

def is_rose_leaf(image):

    image = image.convert("RGB")

    tensor = classifier_transform(image)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = classifier(tensor)

        probs = torch.softmax(outputs, dim=1)

        pred = torch.argmax(probs, dim=1).item()

        confidence = probs[0][pred].item()

    return pred == 1, confidence

uploaded_file = st.file_uploader(
    "Upload a Rose Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Disease Analysis")

        if st.button("Analyze Image"):

            is_rose, rose_conf = is_rose_leaf(image)

            if not is_rose:

                st.error(
                    f"""
                    This image was classified as a non-rose image.

                    Confidence: {rose_conf:.2%}

                    Please upload a clear image of a rose leaf.
                    """
                )

                st.stop()

            st.success(
                f"Rose leaf verified ({rose_conf:.2%} confidence)"
            )

            with st.spinner("Analyzing image..."):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as tmp:

                    image.save(tmp.name)
                    image.save(tmp.name)

                    results = model(
                        tmp.name,
                        conf=0.75
                    )

            result = results[0]

            if len(result.boxes) == 0:

                st.warning("No disease detected.")

            else:

                healthy_classes = [
                    "healthy",
                    "healthy_leaf",
                    "healthy leaves",
                    "healthy leaf"
                ]

                disease_boxes = []

                for box in result.boxes:

                    cls_id = int(box.cls.item())

                    class_name = model.names[
                        cls_id
                    ].lower()

                    if class_name not in healthy_classes:

                        disease_boxes.append(box)

                if len(disease_boxes) > 0:

                    best_box = max(
                        disease_boxes,
                        key=lambda x: float(x.conf)
                    )

                else:

                    best_box = max(
                        result.boxes,
                        key=lambda x: float(x.conf)
                    )

                disease_id = int(
                    best_box.cls.item()
                )

                confidence = float(
                    best_box.conf.item()
                )

                disease_name = model.names[
                    disease_id
                ]

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3 style="color:#1f1f1f;">Disease Detected</h3>
                        <h2 style="color:#2E7D32;">{disease_name}</h2>
                        <h4 style="color:#444444;">Confidence: {confidence:.2%}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            st.subheader(
                "Detection Result"
            )

            st.image(
                result.plot(),
                result.plot(),
                use_container_width=True
            )