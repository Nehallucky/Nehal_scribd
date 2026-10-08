import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Scribd Document Classification",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 24px;
    font-weight: 600;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #123d2b;
    color: white;
    font-size: 22px;
    font-weight: 600;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1e293b;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📚 Scribd Document Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Document Type Classification</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a document image and the machine learning model "
    "will identify its document category."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("model/document_classifier.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")

    return model, label_encoder


try:

    model, label_encoder = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")
    st.error(str(e))
    st.stop()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    # Convert PIL image to OpenCV format
    img = np.array(image)

    # RGB -> BGR
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize for faster processing
    max_size = 1200

    h, w = img.shape[:2]

    scale = min(max_size / max(h, w), 1)

    if scale < 1:

        img = cv2.resize(
            img,
            (
                int(w * scale),
                int(h * scale)
            )
        )

    # Dimensions
    height, width = img.shape[:2]

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Basic image features
    # --------------------------------------------------------

    aspect_ratio = width / height

    brightness = np.mean(gray)

    contrast = np.std(gray)

    # --------------------------------------------------------
    # Text density
    # --------------------------------------------------------

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    text_density = np.mean(threshold > 0)

    # --------------------------------------------------------
    # Edge density
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.mean(edges > 0)

    # --------------------------------------------------------
    # Horizontal and vertical lines
    # --------------------------------------------------------

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (40, 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 40)
    )

    horizontal_lines_img = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_lines_img = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    horizontal_lines = cv2.countNonZero(
        horizontal_lines_img
    )

    vertical_lines = cv2.countNonZero(
        vertical_lines_img
    )

    # --------------------------------------------------------
    # Contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_count = len(contours)

    # --------------------------------------------------------
    # RGB colour features
    # --------------------------------------------------------

    blue_mean = np.mean(img[:, :, 0])

    green_mean = np.mean(img[:, :, 1])

    red_mean = np.mean(img[:, :, 2])

    # --------------------------------------------------------
    # Final 13 features
    # --------------------------------------------------------

    features = [
        width,
        height,
        aspect_ratio,
        brightness,
        contrast,
        text_density,
        edge_density,
        horizontal_lines,
        vertical_lines,
        contour_count,
        blue_mean,
        green_mean,
        red_mean
    ]

    return features, width, height, aspect_ratio


# ============================================================
# DOCUMENT CATEGORIES
# ============================================================

categories = [
    "book",
    "research_paper",
    "resume",
    "report",
    "magazine",
    "presentation"
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📂 Document Categories")

st.sidebar.write(
    "The system classifies documents into:"
)

for category in categories:

    st.sidebar.write(
        "• " + category.replace("_", " ").title()
    )

st.sidebar.markdown("---")

st.sidebar.info(
    "Model: Random Forest\n\n"
    "Features: 13 visual features"
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown("## 📤 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a document image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features, width, height, aspect_ratio = extract_features(
            image
        )

        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        feature_names = [
            "width",
            "height",
            "aspect_ratio",
            "brightness",
            "contrast",
            "text_density",
            "edge_density",
            "horizontal_lines",
            "vertical_lines",
            "contour_count",
            "blue_mean",
            "green_mean",
            "red_mean"
        ]

        feature_df = pd.DataFrame(
            [features],
            columns=feature_names
        )

        # ----------------------------------------------------
        # ML Prediction
        # ----------------------------------------------------

        prediction_encoded = model.predict(
            feature_df
        )[0]

        prediction = label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            feature_df
        )[0]

        # ----------------------------------------------------
        # PRESENTATION SLIDE DETECTION
        # ----------------------------------------------------
        #
        # Most PowerPoint presentation slides are landscape
        # and commonly use a 16:9 or similar aspect ratio.
        #
        # If the uploaded image is strongly landscape-shaped,
        # classify it as a presentation.
        #
        # This specifically helps with PPT/PPTX slide images.
        # ----------------------------------------------------

        presentation_detected = False

        if aspect_ratio >= 1.55:

            presentation_detected = True

            prediction = "presentation"

            # Create new probability distribution
            probabilities = np.zeros(
                len(label_encoder.classes_)
            )

            presentation_index = list(
                label_encoder.classes_
            ).index("presentation")

            probabilities[presentation_index] = 1.0

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = np.max(probabilities) * 100

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.markdown("---")

        col1, col2 = st.columns(
            [1, 1]
        )

        # ----------------------------------------------------
        # IMAGE PREVIEW
        # ----------------------------------------------------

        with col1:

            st.markdown(
                "## 🖼️ Document Preview"
            )

            st.image(
                image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # CLASSIFICATION RESULT
        # ----------------------------------------------------

        with col2:

            st.markdown(
                "## 🔍 Classification Result"
            )

            display_prediction = prediction.replace(
                "_",
                " "
            ).upper()

            st.success(
                f"Predicted Document Type: {display_prediction}"
            )

            st.markdown(
                "### Prediction Confidence"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            # ------------------------------------------------
            # Detection explanation
            # ------------------------------------------------

            if presentation_detected:

                st.info(
                    "📊 Presentation slide detected based on "
                    "the wide landscape format of the uploaded image."
                )

            else:

                st.info(
                    "🤖 Prediction generated using the trained "
                    "Random Forest classification model."
                )

        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.markdown("---")

        st.markdown(
            "## 📊 Class Probabilities"
        )

        probability_data = pd.DataFrame(
            {
                "Document Type": [
                    x.replace("_", " ").title()
                    for x in label_encoder.classes_
                ],
                "Probability": [
                    round(float(x) * 100, 2)
                    for x in probabilities
                ]
            }
        )

        probability_data = probability_data.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probability_data.set_index(
                "Document Type"
            )
        )

        # ====================================================
        # IMAGE FEATURES
        # ====================================================

        with st.expander(
            "🔬 View Extracted Image Features"
        ):

            st.dataframe(
                feature_df.T.rename(
                    columns={0: "Value"}
                )
            )

        # ====================================================
        # DOCUMENT INFORMATION
        # ====================================================

        with st.expander(
            "📐 Image Information"
        ):

            info_col1, info_col2, info_col3 = st.columns(3)

            with info_col1:

                st.metric(
                    "Width",
                    f"{width}px"
                )

            with info_col2:

                st.metric(
                    "Height",
                    f"{height}px"
                )

            with info_col3:

                st.metric(
                    "Aspect Ratio",
                    f"{aspect_ratio:.2f}"
                )

    except Exception as e:

        st.error(
            "An error occurred while processing the image."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Scribd Document Classification | "
    "AI-Based Document Type Classification"
)