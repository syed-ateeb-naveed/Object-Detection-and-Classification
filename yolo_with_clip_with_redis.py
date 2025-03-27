import streamlit as st
import torch
import clip
import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ffmpeg
import tempfile
import cv2
import asyncio
import sys
import redis
import json
from datetime import datetime

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Function to log detections
def log_detection(prediction, timestamp):
    detection_entry = {"object": prediction, "timestamp": timestamp}
    redis_client.rpush("detections", json.dumps(detection_entry))

# if sys.platform == "win32":
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# --- YOLO Setup ---
from ultralytics import YOLO

# FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# Load YOLO model (ensure the YOLO weights path is correct)
@st.cache_resource
def load_yolo_model():
    yolo_model = YOLO("best.pt")
    return yolo_model

yolo_model = load_yolo_model()

# Function to run YOLO on an image and return multiple cropped regions
def crop_with_yolo(pil_image):
    # Convert PIL image to numpy array in BGR format for YOLO
    image_np = np.array(pil_image.convert("RGB"))[:, :, ::-1]  # Convert RGB to BGR
    # Run detection
    results = yolo_model(image_np)
    cropped_images = []
    
    if not results or len(results) == 0:
        st.warning("No objects detected by YOLO.")
        return [pil_image]  # Return original image if no detections

    first_result = results[0]
    # Check if bounding boxes were detected
    if first_result.boxes is None or len(first_result.boxes.xyxy) == 0:
        st.warning("No bounding boxes found.")
        return [pil_image]

    # Loop over each bounding box and crop the image
    for box in first_result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cropped_img = pil_image.crop((x1, y1, x2, y2))
        cropped_images.append(cropped_img)
    
    return cropped_images

# --- CLIP and Model Training Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_clip_model():
    with st.spinner("🚀 Loading CLIP Model... Please wait."):
        model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

clip_model, preprocess = load_clip_model()

def load_dataset(dataset_path):
    features, labels, class_names = [], [], []
    
    for label, class_name in enumerate(os.listdir(dataset_path)): 
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            for img_file in os.listdir(class_path):
                image_path = os.path.join(class_path, img_file)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image).cpu().numpy()
                features.append(image_features)
                labels.append(label)
    
    return np.vstack(features), np.array(labels), class_names

def save_new_object(images, object_name):
    directory = f"dataset/{object_name}"
    os.makedirs(directory, exist_ok=True)
    
    for idx, image in enumerate(images):
        image.save(f"{directory}/{object_name}_{idx}.jpg")

@st.cache_resource
def train_model():
    dataset_path = "dataset"
    class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    if len(class_folders) < 2:
        st.warning("⚠️ At least two classes are required to train the model.")
        return None, None

    with st.spinner("⚙️ Training the model... Please wait."):
        features, labels, class_names = load_dataset(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, clf.predict(X_test))
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy * 100:.2f}%")
        
    return clf, class_names

def classify(image, clf, class_names):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor).cpu().numpy()

    prediction = clf.predict(image_features)
    probabilities = clf.predict_proba(image_features)[0]

    predicted_class = class_names[prediction[0]]
    confidence = probabilities[prediction[0]] * 100  # Convert to percentage

    return predicted_class, confidence, probabilities

def get_detections():
    detections = redis_client.lrange("detections", 0, -1)  # Get all detections
    return [json.loads(d) for d in detections]  # Convert JSON strings to dictionaries

def extract_frames_cv2(video_path, output_folder, frame_rate=12):
    os.makedirs(output_folder, exist_ok=True)
    print(frame_rate)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Failed to open video file!")
        return [], []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get original FPS of the video
    print(f"Original FPS: {fps}")
    frame_interval = max(1, fps // frame_rate)  # Extract every Nth frame
    print(f"Frame interval: {frame_interval}")
    
    frames = []
    frame_indices = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_id % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_id:03d}.png")
            cv2.imwrite(frame_filename, frame)  # Save frame
            
            frames.append(Image.open(frame_filename))  # Convert to PIL Image
            frame_indices.append(frame_id)
        
        frame_id += 1

    cap.release()
    
    if not frames:
        st.error("❌ OpenCV did not generate any frames!")
        return [], []
    
    return frames, frame_indices

def extract_frames_ffmpeg(video_path, output_folder, frame_rate=5):
    os.makedirs(output_folder, exist_ok=True)
    output_pattern = os.path.join(output_folder, "frame_%03d.png")
  
    try:
        process = (
            ffmpeg
            .input(video_path)
            .output(output_pattern, vf=f'fps={frame_rate}')
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Debugging Step: Check if images were created
        frame_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
        
        if not frame_files:
            st.error("❌ FFmpeg did not generate any frames!")
            return [], []
        
        frames = [Image.open(os.path.join(output_folder, f)) for f in frame_files]
        frame_indices = [int(f.split('_')[1].split('.')[0]) for f in frame_files]
        
        return frames, frame_indices

    except ffmpeg.Error as e:
        st.error("❌ Error extracting frames using FFmpeg!")
        st.write(e.stderr.decode())  # Show detailed error message
        return [], []  # Ensure the function does not return None
# --- Streamlit App ---
def app():
    st.title("🔍 Object Classification and Detection")

    # User authentication
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 Login to Access the App")
        password = st.text_input("Enter password", type="password")
        if st.button("Login"):
            if password == "rao":
                st.session_state["authenticated"] = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")

    if st.session_state["authenticated"]:
        st.sidebar.success("✅ Authenticated")
        
        option = st.sidebar.selectbox("Choose an option:", ["Add Object", "Detect Object"])
        clf, class_names = None, None

        if option == "Add Object":
            st.header("📤 Add a New Object")
            object_name = st.text_input("Enter Object Name")
            person_name = st.text_input("Enter Your Name")
            
            if object_name and person_name:
                st.write("Upload Four Images of the Object:")
                uploaded_images = []
                front_image = st.file_uploader("Upload Front Side Image", type=["jpg", "jpeg", "png"])
                back_image = st.file_uploader("Upload Back Side Image", type=["jpg", "jpeg", "png"])
                corner_image = st.file_uploader("Upload Corner View Image", type=["jpg", "jpeg", "png"])
                side_image = st.file_uploader("Upload Side View Image", type=["jpg", "jpeg", "png"])

                if front_image and back_image and corner_image and side_image:
                    uploaded_images = [Image.open(front_image), Image.open(back_image),
                                       Image.open(corner_image), Image.open(side_image)]
                    folder_name = f"{person_name}_{object_name}"
                    save_new_object(uploaded_images, folder_name)
                    st.success(f"📂 Object '{object_name}' added successfully! You can now train the model.")
                    
                    if st.button("Train Model Now"):
                        clf, class_names = train_model()
                        if clf:
                            st.success("🎉 Model training completed!")

        elif option == "Detect Object":
            st.header("🕵️‍♂️ Detect an Object")
            choice = st.radio("Choose detection method", ["By Image", "By Video"])

            if choice == "By Image":
                test_image = st.file_uploader("Upload an Image for Classification", type=["jpg", "jpeg", "png"])
                if test_image:
                    image = Image.open(test_image)
                    st.image(image, caption="Uploaded Image", width=300)

                    # ---- New: YOLO cropping step for multiple detections ----
                    with st.spinner("🔄 Cropping image using YOLO..."):
                        cropped_images = crop_with_yolo(image)
                    # Display each cropped image
                    for idx, cropped_image in enumerate(cropped_images):
                        st.image(cropped_image, caption=f"YOLO Cropped Region {idx+1}", width=300)
                    # -------------------------------------------------------------

                    with st.spinner("🔍 Loading trained model..."):
                        clf, class_names = train_model()

                    if clf:
                        st.write("### Classification Results")
                        for idx, cropped_image in enumerate(cropped_images):
                            prediction, confidence, probabilities = classify(cropped_image, clf, class_names)
                            st.write(f"**Region {idx+1} Prediction:** {prediction}")
                            st.write(f"**Confidence:** {confidence:.2f}%")
                            st.write("**Class Probabilities:**")
                            for i, cls_name in enumerate(class_names):
                                st.write(f"   - {cls_name}: {probabilities[i] * 100:.2f}%")
                            st.markdown("---")
                        # Optional warning if none of the detections are confident enough

            elif choice == "By Video":
                video_file = st.file_uploader("Upload a video for object detection", type=["mp4", "avi", "mov"])
                if video_file:
                    temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
                    with open(temp_video_path, "wb") as f:
                        f.write(video_file.read())

                    st.video(temp_video_path)
                    extracted_folder = os.path.join("extracted_frames", os.path.splitext(video_file.name)[0])
                    os.makedirs(extracted_folder, exist_ok=True)

                    with st.spinner("📸 Extracting frames..."):
                        frames, frame_indices = extract_frames_cv2(temp_video_path, extracted_folder)

                    if frames:
                        st.success(f"✅ Frames extracted and saved in: `{extracted_folder}`")

                        if st.button("🚀 Start Classification"):
                            detected_objects = {}
                            successful_detections = 0
                            unsuccessful_detections = 0

                            with st.spinner("🔍 Loading trained model..."):
                                clf, class_names = train_model()

                            if clf:
                                for frame, frame_index in zip(frames, frame_indices):
                                    # ---- Run YOLO cropping on each frame ----
                                    cropped_regions = crop_with_yolo(frame)
                                    # Classify each detected region
                                    for idx, cropped_region in enumerate(cropped_regions):
                                        prediction, confidence, _ = classify(cropped_region, clf, class_names)
                                        st.image(cropped_region, caption=f"Frame {frame_index} - Region {idx+1}: {prediction} ({confidence:.2f}%)", width=300)
                                        if confidence > 50:
                                            detected_objects[prediction] = detected_objects.get(prediction, 0) + 1
                                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            log_detection(prediction, timestamp)
                                            successful_detections += 1
                                        else:
                                            unsuccessful_detections += 1

                                st.write("### Detected Objects Summary:")
                                if detected_objects:
                                    for obj, count in detected_objects.items():
                                        st.write(f"- **{obj}** detected in {count} regions across frames.")
                                else:
                                    st.warning("⚠️ No confident objects detected in the video.")

                                st.write("### Detection Summary:")
                                st.write(f"**Successful detections:** {successful_detections}")
                                st.write(f"**Unsuccessful detections:** {unsuccessful_detections}")

                                

if __name__ == "__main__":
    app()
