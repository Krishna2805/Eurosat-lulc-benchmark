import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os
import io

# Streamlit app configuration
st.set_page_config(page_title="EuroSAT RGB Classification Report", layout="wide", page_icon="📊")

# Custom CSS for modern, sleek, and appealing UI
st.markdown("""
    <style>
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #e6e9ff 0%, #f9fafb 100%);
        padding: 25px;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    .report-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.6s ease-in;
    }
    .report-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    h1 {
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    h2 {
        color: #1e3a8a;
        font-size: 1.8rem;
        font-weight: 600;
    }
    h3 {
        color: #1e3a8a;
        font-size: 1.4rem;
        font-weight: 500;
        margin-bottom: 15px;
    }
    p, li {
        color: #374151;
        font-size: 1.1rem;
        line-height: 1.7;
    }
    /* Sidebar styles */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        max-width: 280px;
    }
    .sidebar h2 {
        color: #ffffff;
        font-size: 1.8rem;
        text-align: center;
        margin-bottom: 15px;
    }
    .sidebar .stRadio > div {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 12px;
    }
    .sidebar .stRadio label {
        color: #f3f4f6;
        font-size: 1.1rem;
        padding: 12px;
        border-radius: 8px;
        transition: background 0.3s ease, transform 0.2s ease;
    }
    .sidebar .stRadio label:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateX(5px);
    }
    .sidebar .stRadio input:checked + div {
        background: #2563eb;
        color: white;
        font-weight: 600;
    }
    /* Button styles */
    .stButton>button {
        background: #2563eb;
        color: white;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: 500;
        border: none;
        transition: background 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
    }
    .stButton>button:hover {
        background: #1e40af;
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    /* Selectbox styles */
    .stSelectbox>div>div {
        background: #ffffff;
        border-radius: 10px;
        border: 2px solid #d1d5db;
        padding: 12px;
        font-size: 1.1rem;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stSelectbox>div>div:hover {
        border-color: #2563eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .stSelectbox label {
        color: #1e3a8a;
        font-weight: 500;
        font-size: 1.1rem;
    }
    .stSelectbox div[data-baseweb="select"] > div > div {
        background: #dbeafe;
        color: #1e3a8a;
        font-weight: 600;
    }
    /* Slider styles */
    .stSlider>div {
        color: #1e3a8a;
        font-size: 1rem;
    }
    /* Metric and dataframe styles */
    .stMetric {
        background: #f9fafb;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 15px;
    }
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06);
    }
    /* Plot container */
    .plot-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
        background: #f9fafb;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06);
    }
    /* Spinner and alerts */
    .stSpinner>div {
        border-color: #2563eb !important;
    }
    .stSuccess, .stError {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        font-size: 1.1rem;
    }
    .stSuccess {
        background: #d1fae5;
        color: #065f46;
    }
    .stError {
        background: #fee2e2;
        color: #b91c1c;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Responsive design */
    @media (max-width: 768px) {
        .report-container {
            padding: 20px;
        }
        .main {
            padding: 15px;
        }
        .sidebar .sidebar-content {
            padding: 20px;
            max-width: 100%;
        }
        h1 { font-size: 2rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.2rem; }
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Sidebar navigation
with st.sidebar:
    st.markdown("<h2>🔍 Navigation</h2>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Go to", ["Overview", "Real-Time Predictions"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='color: #bfdbfe; text-align: center;'>Built with Streamlit & TensorFlow<br>© 2025 EuroSAT Analysis</p>", unsafe_allow_html=True)

# Load EuroSAT dataset
@st.cache_resource
def load_dataset():
    try:
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'eurosat/rgb',
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            as_supervised=True,
            with_info=True
        )
        return ds_train, ds_val, ds_test, ds_info
    except Exception as e:
        st.error(f"Failed to load EuroSAT dataset: {str(e)}")
        return None, None, None, None

ds_train, ds_val, ds_test, ds_info = load_dataset()
if ds_info is None:
    st.stop()

class_names = ds_info.features['label'].names
num_classes = len(class_names)

# Preprocessing functions
def preprocess_cnn_scratch(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [128, 128])
    return image, label

def preprocess_vgg16(image, label):
    image = tf.image.resize(image, [64, 64])
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, depth=num_classes)

def preprocess_resnet50(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, tf.one_hot(label, depth=num_classes)

# Swish activation for CNN_Scratch
def swish(x):
    return x * tf.nn.sigmoid(x)

# Build models with caching
@st.cache_resource
def build_cnn_scratch():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Activation(swish),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Activation(swish),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Activation(swish),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Activation(swish),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Activation(swish),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Activation(swish),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Activation(swish),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

@st.cache_resource
def build_resnet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

@st.cache_resource
def build_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

# Main content
if page == "Overview":
    st.markdown("<div class='report-container'><h1>🌍 EuroSAT RGB Classification Report</h1></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='report-container'>
        <p>Welcome to the EuroSAT RGB Classification Report. This interactive dashboard evaluates pre-trained CNN models on the EuroSAT RGB dataset, providing detailed metrics, visualizations, and real-time predictions.</p>
        <h3>Dataset Overview</h3>
        <ul>
            <li><b>Dataset</b>: EuroSAT RGB</li>
            <li><b>Classes</b>: {", ".join(class_names)}</li>
            <li><b>Split</b>: 80% Train, 10% Validation, 10% Test</li>
        </ul>
        <h3>Models Available</h3>
        <ul>
            <li><b>CNN_Scratch</b>: Custom-built CNN with Swish activation</li>
            <li><b>VGG16</b>: Transfer learning with pre-trained VGG16</li>
            <li><b>ResNet50</b>: Transfer learning with pre-trained ResNet50</li>
        </ul>
        <p>Use the sidebar to navigate to Real-Time Predictions.</p>
        </div>
    """, unsafe_allow_html=True)

elif page == "Real-Time Predictions":
    st.markdown("<div class='report-container'><h1>🔮 Real-Time Predictions</h1></div>", unsafe_allow_html=True)
    
    # Model selection
    model_choice = st.selectbox("Select Model", ["CNN_Scratch", "VGG16", "ResNet50"], key="model_pred")
    st.markdown(f"<p style='color: #1e3a8a; font-weight: 500;'>Selected Model: {model_choice}</p>", unsafe_allow_html=True)
    
    # Configure model
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE
    
    if model_choice == "CNN_Scratch":
        preprocess_fn = preprocess_cnn_scratch
        model = build_cnn_scratch()
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model_path = 'CNN_best_model.h5'
        img_size = (128, 128)
    elif model_choice == "VGG16":
        preprocess_fn = preprocess_vgg16
        model_path = 'vgg16_eurosat_tfds_finetuned.h5'
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={'swish': swish})
            st.success(f"Loaded pre-trained VGG16 model from {model_path}")
        except Exception as e:
            st.error(f"Failed to load VGG16 model: {str(e)}. Using fresh VGG16 model.")
            model = build_vgg16()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        img_size = (64, 64)
    else:  # ResNet50
        preprocess_fn = preprocess_resnet50
        model = build_resnet50()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model_path = 'ResNet_best_model.h5'
        img_size = (224, 224)
    
    # Load model weights
    if model_choice in ["CNN_Scratch", "ResNet50"]:
        if os.path.exists(model_path):
            try:
                model.load_weights(model_path)
                st.success(f"Loaded pre-trained weights from {model_path}")
            except Exception as e:
                st.error(f"Failed to load weights for {model_choice}: {str(e)}. Using untrained model.")
        else:
            st.error(f"Model weights file {model_path} not found. Using untrained model.")
    
    # Prepare dataset
    try:
        test_ds = ds_test.map(preprocess_fn, AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
        unbatched_test = test_ds.unbatch().take(1000)
        test_images = list(unbatched_test)
    except Exception as e:
        st.error(f"Failed to prepare test dataset: {str(e)}")
        st.stop()
    
    # Interactive controls
    num_predictions = st.slider("Number of Predictions", min_value=1, max_value=5, value=1, key="num_pred")
    
    if st.button("Generate Predictions", key="pred_button"):
        for i in range(num_predictions):
            with st.expander(f"Prediction {i+1}", expanded=True):
                try:
                    image, true_label = random.choice(test_images)
                    input_image = tf.expand_dims(image, axis=0)
                    pred_probs = model.predict(input_image, verbose=0)
                    pred_class = np.argmax(pred_probs[0])
                    
                    # Convert true_label to integer for indexing
                    true_label = tf.argmax(true_label, axis=-1).numpy() if model_choice in ["ResNet50", "VGG16"] else true_label.numpy()
                    
                    # Display image
                    st.markdown("<h3>Image Prediction</h3>", unsafe_allow_html=True)
                    plt.figure(figsize=(6, 6))
                    image_display = image.numpy()
                    if model_choice == "ResNet50":
                        image_display = image_display + [103.939, 116.779, 123.68]
                        image_display = image_display[:, :, ::-1]
                        image_display = np.clip(image_display, 0, 255).astype(np.uint8)
                    elif model_choice == "CNN_Scratch":
                        image_display = np.clip(image_display, 0, 1)
                    plt.imshow(image_display)
                    plt.title(f"True: {class_names[true_label]}\nPredicted: {class_names[pred_class]}")
                    plt.axis('off')
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format='png', dpi=300)
                    img_buf.seek(0)
                    st.image(img_buf, use_container_width=True)
                    
                    # Confidence scores
                    st.markdown("<h3>Confidence Scores</h3>", unsafe_allow_html=True)
                    confidence_df = pd.DataFrame({
                        "Class": class_names,
                        "Confidence": pred_probs[0]
                    }).round(4)
                    st.dataframe(confidence_df, use_container_width=True)
                    
                    # Confidence heatmap
                    st.markdown("<h3>Confidence Heatmap</h3>", unsafe_allow_html=True)
                    confidences = pred_probs[0].reshape(1, -1)
                    plt.figure(figsize=(14, 3))
                    sns.heatmap(
                        confidences,
                        annot=True,
                        fmt='.5f',
                        cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=['Confidence'],
                        annot_kws={'size': 10}
                    )
                    plt.xlabel('Class')
                    plt.title('Prediction Confidence Heatmap')
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.yticks(rotation=0, fontsize=10)
                    plt.tight_layout(pad=2.0)
                    plt.subplots_adjust(bottom=0.3, left=0.1)
                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format='png', dpi=300)
                    img_buf.seek(0)
                    st.image(img_buf, use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction {i+1} failed: {str(e)}")

if __name__ == "__main__":
    st.write("Navigate using the sidebar to explore the report.")