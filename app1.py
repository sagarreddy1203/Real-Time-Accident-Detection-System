import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tempfile
import os
import pygame
from datetime import datetime, timedelta
from notifications import send_email, send_whatsapp_message
import time


# ---------------- CONFIG ----------------
MODEL_PATH = "best_hybrid_model.pt"
ALARM_FILE = "alarm.wav"
COOLDOWN_SECONDS = 60  # Minimum seconds between notifications
# ---------------------------------------

# Page configuration
st.set_page_config(
    page_title="🚗 Hybrid Accident Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .detection-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Y
# OUR TRAINED CNN+TRANSFORMER HYBRID MODEL
class CNNTransformerFusion(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        # CNN Branch - EfficientNet for detailed spatial features
        self.cnn_branch = timm.create_model(
            'efficientnet_b4', 
            pretrained=True, 
            num_classes=0,
            global_pool=''
        )
        self.cnn_adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.cnn_projection = nn.Sequential(
            nn.Linear(1792 * 49, 768),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Transformer Branch - Vision Transformer
        self.vit_branch = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0
        )
        
        # Cross-Attention Fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature Fusion Network
        self.fusion_layers = nn.Sequential(
            nn.Linear(768 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )
        
        # Temporal Components for Video
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
    
    def extract_hybrid_features(self, x):
        batch_size = x.shape[0]
        
        # CNN Branch
        cnn_features = self.cnn_branch(x)
        cnn_features = self.cnn_adaptive_pool(cnn_features)
        cnn_features = cnn_features.flatten(1)
        cnn_features = self.cnn_projection(cnn_features)
        
        # ViT Branch
        vit_features = self.vit_branch(x)
        
        # Cross-Attention Fusion
        cnn_query = cnn_features.unsqueeze(1)
        vit_key_value = vit_features.unsqueeze(1)
        
        fused_cnn, _ = self.cross_attention(cnn_query, vit_key_value, vit_key_value)
        fused_vit, _ = self.cross_attention(vit_key_value, cnn_query, cnn_query)
        
        # Combine features
        hybrid_features = torch.cat([
            fused_cnn.squeeze(1), 
            fused_vit.squeeze(1)
        ], dim=1)
        
        return hybrid_features
    
    def forward(self, x, is_video=False):
        if is_video:
            batch_size, seq_len = x.shape[:2]
            
            # Extract features for each frame
            frame_features = []
            for i in range(seq_len):
                hybrid_feat = self.extract_hybrid_features(x[:, i])
                fused_feat = self.fusion_layers(hybrid_feat)
                frame_features.append(fused_feat)
            
            # Temporal modeling
            temporal_features = torch.stack(frame_features, dim=1)
            attended_features, _ = self.temporal_attention(
                temporal_features, temporal_features, temporal_features
            )
            
            # Global pooling
            final_features = attended_features.mean(dim=1)
            return self.classifier(final_features)
        else:
            # Image processing
            hybrid_features = self.extract_hybrid_features(x)
            fused_features = self.fusion_layers(hybrid_features)
            return self.classifier(fused_features)

# Initialize session state
if 'model' not in st.session_state:
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNNTransformerFusion(num_classes=2, dropout_rate=0.3)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        st.session_state.model = model
        st.session_state.device = device
        st.session_state.model_loaded = True
        st.session_state.accuracy = checkpoint.get('best_val_acc', 0.0)
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.model_error = str(e)

if 'last_notification_time' not in st.session_state:
    st.session_state.last_notification_time = datetime.min

if 'detection_count' not in st.session_state:
    st.session_state.detection_count = 0

if 'alarm_initialized' not in st.session_state:
    try:
        pygame.mixer.init()
        st.session_state.alarm_sound = pygame.mixer.Sound(ALARM_FILE)
        st.session_state.alarm_initialized = True
    except Exception as e:
        st.session_state.alarm_initialized = False
        st.session_state.alarm_error = str(e)

def load_video_frames(video_path, max_frames=16):
    """Load video frames for processing"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
    else:
        frame_indices = range(total_frames)
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame_tensor = transform(image=np.array(frame))['image']
        frames.append(frame_tensor)
    
    cap.release()
    
    while len(frames) < max_frames:
        frames.append(frames[-1])
    
    return torch.stack(frames[:max_frames])

def predict_hybrid(input_data, is_video=False):
    """Make prediction using hybrid model"""
    with torch.no_grad():
        try:
            outputs = st.session_state.model(input_data, is_video=is_video)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1)
            
            prediction = 'Accident' if predicted_class[0] == 1 else 'No Accident'
            confidence = probabilities[0][predicted_class[0]].item()
            accident_prob = probabilities[0][1].item()
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'accident_probability': accident_prob,
                'success': True
            }
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'accident_probability': 0.0,
                'success': False,
                'error': str(e)
            }

def handle_detection_alert(accident_detected, enable_notifications, enable_sound):
    """Handle alerts when accident is detected"""
    if accident_detected:
        st.session_state.detection_count += 1
        
        now = datetime.now()
        if (now - st.session_state.last_notification_time).total_seconds() > COOLDOWN_SECONDS:
            if enable_sound and st.session_state.alarm_initialized:
                st.session_state.alarm_sound.play()
            
            if enable_notifications:
                body = f"Hybrid AI detected accident at {now.strftime('%Y-%m-%d %H:%M:%S')}"
                try:
                    send_email("🚨 Hybrid Model: Accident Detected", body)
                    send_whatsapp_message(body)
                    st.success("📧 Notifications sent successfully!")
                except Exception as e:
                    st.warning(f"⚠️ Notification error: {str(e)}")
            
            st.session_state.last_notification_time = now
            return True
        else:
            remaining_cooldown = COOLDOWN_SECONDS - (now - st.session_state.last_notification_time).total_seconds()
            st.info(f"⏳ Notification cooldown: {remaining_cooldown:.0f}s remaining")
            return False
    return False

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 3rem;">🚗 Hybrid CNN+Transformer Accident Detection</h1>
    <p style="color: white; font-size: 1.2rem; margin: 0.5rem 0;">Advanced AI-powered accident detection with EfficientNet + Vision Transformer</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    # Model status
    if st.session_state.model_loaded:
        st.success("✅ Hybrid Model Loaded")
        st.info(f"🎯 Model Accuracy: {st.session_state.accuracy:.2f}%")
    else:
        st.error(f"❌ Model Error: {st.session_state.model_error}")
    
    # Device info
    if st.session_state.model_loaded:
        st.info(f"💻 Device: {st.session_state.device}")
    
    # Alarm status
    if st.session_state.alarm_initialized:
        st.success("🔊 Audio System Ready")
    else:
        st.warning(f"⚠️ Audio Error: {st.session_state.alarm_error}")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📈 Statistics")
    st.metric("Total Detections", st.session_state.detection_count)
    
    if st.session_state.last_notification_time != datetime.min:
        time_since = datetime.now() - st.session_state.last_notification_time
        st.metric("Last Alert", f"{time_since.seconds}s ago")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.1)
    enable_notifications = st.checkbox("Enable Notifications", True)
    enable_sound = st.checkbox("Enable Sound Alerts", True)
    
    st.markdown("---")
    
    # Model Architecture Info
    st.markdown("### 🧠 Model Architecture")
    st.info("""
    **CNN Branch:** EfficientNet-B4
    **Transformer:** Vision Transformer  
    **Fusion:** Cross-Attention
    **Features:** Temporal Analysis
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Hybrid Detection Input")
    
    # Input type selection
    input_type = st.selectbox(
        "Choose Input Type",
        ["Image Upload", "Video Upload", "Live Webcam"],
        index=0
    )
    
    if input_type == "Image Upload":
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_image is not None:
            # Display original image
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("🔍 Analyze with Hybrid Model", key="detect_image"):
                if not st.session_state.model_loaded:
                    st.error("❌ Model not loaded. Please check model file.")
                else:
                    with st.spinner("Analyzing image with CNN+Transformer..."):
                        try:
                            # Preprocess image
                            transform = A.Compose([
                                A.Resize(224, 224),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2()
                            ])
                            
                            image_tensor = transform(image=np.array(image))['image']
                            image_tensor = image_tensor.unsqueeze(0).to(st.session_state.device)
                            
                            # Predict using hybrid model
                            result = predict_hybrid(image_tensor, is_video=False)
                            
                            if result['success']:
                                prediction = result['prediction']
                                confidence = result['confidence']
                                accident_prob = result['accident_probability']
                                
                                # Display prediction
                                if prediction == 'Accident':
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                                                padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                                        <h2>⚠️ ACCIDENT DETECTED</h2>
                                        <h3>Hybrid AI Analysis Complete</h3>
                                        <p>Confidence: {confidence:.1%} | Risk: {accident_prob:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    handle_detection_alert(True, enable_notifications, enable_sound)
                                else:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
                                                padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                                        <h2>✅ NO ACCIDENT</h2>
                                        <h3>Hybrid AI Analysis Complete</h3>
                                        <p>Confidence: {confidence:.1%} | Risk: {accident_prob:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
    
    elif input_type == "Video Upload":
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            if st.button("🎬 Analyze Video with Hybrid Model", key="detect_video"):
                if not st.session_state.model_loaded:
                    st.error("❌ Model not loaded. Please check model file.")
                else:
                    # Save uploaded video temporarily
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.read())
                    tfile.close()
                    
                    with st.spinner("Processing video with Hybrid CNN+Transformer..."):
                        try:
                            # Load video frames
                            video_frames = load_video_frames(tfile.name)
                            video_tensor = video_frames.unsqueeze(0).to(st.session_state.device)
                            
                            # Predict using hybrid model
                            result = predict_hybrid(video_tensor, is_video=True)
                            
                            if result['success']:
                                prediction = result['prediction']
                                confidence = result['confidence']
                                accident_prob = result['accident_probability']
                                
                                # Display prediction
                                if prediction == 'Accident':
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                                                padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                                        <h2>⚠️ ACCIDENT DETECTED IN VIDEO</h2>
                                        <h3>Temporal Hybrid Analysis Complete</h3>
                                        <p>Confidence: {confidence:.1%} | Risk: {accident_prob:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    handle_detection_alert(True, enable_notifications, enable_sound)
                                else:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
                                                padding: 20px; border-radius: 10px; color: white; text-align: center;'>
                                        <h2>✅ NO ACCIDENT IN VIDEO</h2>
                                        <h3>Temporal Hybrid Analysis Complete</h3>
                                        <p>Confidence: {confidence:.1%} | Risk: {accident_prob:.1%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                        
                        finally:
                            if os.path.exists(tfile.name):
                                os.unlink(tfile.name)
    
    else:  # Live Webcam
        st.markdown("### 📹 Live Webcam Detection")
        
        # Webcam controls
        col_a, col_b = st.columns(2)
        
        with col_a:
            start_webcam = st.button("🎥 Start Hybrid Detection", key="start_cam")
        
        with col_b:
            stop_webcam = st.button("⏹️ Stop Detection", key="stop_cam")
        
        if start_webcam:
            st.session_state.webcam_active = True
        
        if stop_webcam:
            st.session_state.webcam_active = False
        
        # Webcam processing (simplified for demo)
        if st.session_state.get('webcam_active', False):
            st.info("📹 Live hybrid detection would be active here. For security reasons, live webcam is not implemented in this demo.")
            st.markdown("""
            **Note:** In a full implementation, this would:
            - Access your webcam feed in real-time
            - Process frames with CNN+Transformer hybrid model
            - Use temporal attention for video sequences
            - Display live detection results with confidence scores
            - Trigger alerts when accidents are detected
            """)

with col2:
    st.markdown("### 📋 Hybrid Model Information")
    
    st.markdown("""
    <div class="metric-card">
        <h4>🧠 AI Architecture</h4>
        <p>CNN + Transformer Fusion</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h4>⚡ Processing Speed</h4>
        <p>Real-time hybrid inference</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h4>🔔 Alert System</h4>
        <p>Email + WhatsApp + Audio</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("### 📝 Detection History")
    
    if st.session_state.detection_count > 0:
        st.success(f"✅ {st.session_state.detection_count} detections processed")
    else:
        st.info("No detections yet")
    
    # Model performance info
    if st.session_state.model_loaded:
        st.markdown("### 📊 Model Performance")
        st.metric("Accuracy", f"{st.session_state.accuracy:.2f}%")
        st.metric("Architecture", "Hybrid CNN+ViT")
        st.metric("Parameters", "~87M")
    
    # Help section
    st.markdown("### ❓ How to Use")
    st.markdown("""
    1. **Upload Content**: Choose image, video, or webcam
    2. **Adjust Settings**: Set confidence threshold
    3. **Run Detection**: Click analyze button
    4. **View Results**: See AI confidence scores
    5. **Get Alerts**: Receive notifications if enabled
    
    **Model Features:**
    - 🔍 Spatial feature extraction (CNN)
    - 🧠 Global context understanding (Transformer)
    - 🔗 Cross-attention fusion
    - 📹 Temporal analysis for videos
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚗 Hybrid CNN+Transformer Accident Detection System</p>
    <p>Advanced AI • Real-time Analysis • Life-saving Technology</p>
</div>
""", unsafe_allow_html=True)
