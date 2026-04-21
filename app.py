import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from datetime import datetime

# ====================== CONFIG ======================
st.set_page_config(page_title="Rice Disease Detector", page_icon="🌾", layout="wide")


@st.cache_resource
def load_model():
    model_path = "rice_disease_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file `rice_disease_model.h5` not found!")
        st.stop()
    return tf.keras.models.load_model(model_path)


model = load_model()
class_names = ["bacterial_leaf_blight", "brownspot", "leaf_smut"]

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []


# ====================== GRAD-CAM + BOUNDING BOXES ======================
def get_gradcam_heatmap(img_array, model, class_idx):
    last_conv_layer_name = "Conv_1"  # MobileNetV2 last conv layer
    grad_model = tf.keras.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return np.uint8(255 * heatmap)


def draw_disease_boxes(original_pil: Image.Image, heatmap, disease_name):
    original = np.array(original_pil.convert("RGB"))
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Threshold to find diseased areas
    _, thresh = cv2.threshold(heatmap_resized, 110, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = original.copy()
    spot_count = 0

    for cnt in contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        area = box_w * box_h
        if area > 300:  # remove tiny noise
            cv2.rectangle(
                annotated, (x, y), (x + box_w, y + box_h), (255, 0, 0), 4
            )  # thick red box
            # Optional label on box
            cv2.putText(
                annotated,
                disease_name[:10],
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            spot_count += 1

    return Image.fromarray(annotated), spot_count


# ====================== SIDEBAR HISTORY ======================
st.sidebar.title("📜 Test History")
if st.session_state.history:
    for entry in reversed(st.session_state.history[-10:]):
        st.sidebar.image(entry["image"], use_container_width=True)
        st.sidebar.caption(
            f"**{entry['disease'].replace('_', ' ').title()}** • {entry['confidence']:.1f}%"
        )
        st.sidebar.caption(entry["time"])
        st.sidebar.divider()
else:
    st.sidebar.info("No tests yet.")

# ====================== MAIN APP ======================
st.title("🌾 Rice Leaf Disease Detector")
st.markdown("**Live camera • Red bounding boxes on disease spots • History**")

col1, col2 = st.columns([3, 1])
with col1:
    input_method = st.radio(
        "Choose input method:", ["📤 Upload Image", "📷 Live Camera"], horizontal=True
    )

image = None
source_bytes = None

if input_method == "📤 Upload Image":
    uploaded = st.file_uploader("Upload rice leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        source_bytes = uploaded.getvalue()
else:
    picture = st.camera_input("📸 Point camera at rice leaf and take photo")
    if picture:
        image = Image.open(picture).convert("RGB")
        source_bytes = picture.getvalue()

# ====================== PREDICTION ======================
if image is not None:
    st.image(image, caption="📸 Input Image", use_container_width=True)

    if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        with st.spinner("Analyzing + drawing disease spots..."):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(prediction[0])
            predicted_class = class_names[predicted_idx]
            confidence = float(prediction[0][predicted_idx]) * 100

            # Save to history
            st.session_state.history.append(
                {
                    "image": source_bytes,
                    "disease": predicted_class,
                    "confidence": confidence,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            # Generate red bounding boxes (exactly like your sample)
            heatmap = get_gradcam_heatmap(img_array, model, predicted_idx)
            annotated_img, spot_count = draw_disease_boxes(
                image, heatmap, predicted_class.replace("_", " ").title()
            )

            # Results
            st.success("✅ Prediction Complete!")

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(
                    image.resize((380, 380)),
                    caption="Original Leaf",
                    use_container_width=True,
                )
            with col_b:
                st.image(
                    annotated_img.resize((380, 380)),
                    caption="🔴 Disease Spots Detected",
                    use_container_width=True,
                )

            st.subheader("📊 Results")
            st.metric("Predicted Disease", predicted_class.replace("_", " ").title())
            st.metric("Confidence", f"{confidence:.2f}%")
            st.metric("Number of Spots Detected", spot_count)

            st.write("**All Probabilities**")
            for name, prob in zip(class_names, prediction[0]):
                pct = float(prob) * 100
                st.progress(
                    float(prob),
                    text=f"{name.replace('_', ' ').title()}: **{pct:.1f}%**",
                )

else:
    st.info("👆 Choose upload or camera above to start")

st.caption("Built with Streamlit + TensorFlow • Red bounding boxes like your sample")
