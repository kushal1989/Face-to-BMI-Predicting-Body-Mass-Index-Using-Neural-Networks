import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import sqlite3
import os
import time
import matplotlib.pyplot as plt


# ------------------------------- #
#          DATABASE SETUP         #
# ------------------------------- #

def init_db():
    """Initialize the database and create required tables."""
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')

    # Create bmi_records table
    cursor.execute('''CREATE TABLE IF NOT EXISTS bmi_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        bmi REAL NOT NULL,
                        category TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )''')

    conn.commit()
    conn.close()


# ------------------------------- #
#          MODEL SETUP            #
# ------------------------------- #

@st.cache_resource
def load_bmi_model():
    """Load the pre-trained BMI model."""
    model_path = r"C:\Users\USER\PycharmProjects\MINI PROJECT\Optimized_EfficientNetV2_Model.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return load_model(model_path)


bmi_model = load_bmi_model()

# Normalization parameters
bmi_mean = 25.0  # Replace with training mean
bmi_std = 5.0  # Replace with training standard deviation

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ------------------------------- #
#        HELPER FUNCTIONS         #
# ------------------------------- #

def classify_bmi(bmi):
    """Classify BMI into categories and provide health insights."""
    if bmi < 18.5:
        category = "Underweight"
        advice = "Consider consuming a nutrient-rich diet with more calories and engaging in strength-building exercises."
    elif 18.5 <= bmi < 25:
        category = "Normal"
        advice = "Maintain your current lifestyle with a balanced diet and regular physical activity."
    elif 25 <= bmi < 30:
        category = "Overweight"
        advice = "Incorporate more physical activity into your routine and consider reducing high-calorie food intake."
    else:  # BMI >= 30
        category = "Obese"
        advice = "Consult a healthcare provider for a tailored weight-loss plan, including diet and exercise."
    return category, advice


# ------------------------------- #
#        HELPER FUNCTIONS         #
# ------------------------------- #

def add_bmi_record(username, bmi, category):
    """Add BMI record to the database."""
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO bmi_records (username, bmi, category) VALUES (?, ?, ?)", (username, bmi, category))
    conn.commit()
    conn.close()


def get_bmi_history(username):
    """Retrieve BMI records for a specific user."""
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT bmi, category, timestamp FROM bmi_records WHERE username = ? ORDER BY timestamp DESC",
                   (username,))
    records = cursor.fetchall()
    conn.close()
    return records


# ------------------------------- #
#       BMI HISTORY LOGIC         #
# ------------------------------- #

def show_bmi_history(username):
    """Display BMI history as a table and graph."""
    st.subheader("BMI History")
    records = get_bmi_history(username)

    if not records:
        st.info("No BMI records found.")
        return

    # Convert records to a DataFrame for better visualization
    import pandas as pd
    df = pd.DataFrame(records, columns=["BMI", "Category", "Timestamp"])

    # Display records in a table
    st.table(df)

    # Plot BMI history
    plt.figure(figsize=(10, 6))
    plt.plot(df["Timestamp"], df["BMI"], marker="o", linestyle="-", color="b")
    plt.title("BMI History Over Time")
    plt.xlabel("Date")
    plt.ylabel("BMI")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)


def authenticate_user(username, password):
    """Authenticate user credentials."""
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user


def register_user(username, password):
    """Register a new user."""
    try:
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


# ------------------------------- #
#          AUTHENTICATION         #
# ------------------------------- #

def login():
    """Handle user login and authentication."""
    st.markdown("<h1 style='text-align: center;'>Face to BMI</h1>", unsafe_allow_html=True)
    st.title("Login")
    st.sidebar.image(r"C:\Users\USER\Downloads\facetobmi.png", use_column_width=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    new_user_button = st.button("New User? Register Here")

    if login_button:
        user = authenticate_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.experimental_rerun()  # Redirect to the main application
        else:
            st.error("Invalid username or password.")

    if new_user_button:
        st.session_state.register = True
        st.experimental_rerun()  # Redirect to the signup page


def signup():
    """Handle user registration."""
    st.markdown("<h1 style='text-align: center;'>Face to BMI</h1>", unsafe_allow_html=True)
    st.title("Signup")
    st.sidebar.image(r"C:\Users\USER\Downloads\facetobmi.png", use_column_width=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    signup_button = st.button("Register")

    if signup_button:
        success = register_user(username, password)
        if success:
            st.success("Registration successful! Please login.")
            st.session_state.register = False
            st.experimental_rerun()  # Redirect to the login page
        else:
            st.error("Username already exists. Please try a different username.")


# ------------------------------- #
#     REAL-TIME CAMERA LOGIC      #
# ------------------------------- #

def run_camera():
    """Stream live video, detect faces, and enable BMI prediction."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera. Please make sure the webcam is connected.")
        return

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    video_placeholder = st.empty()  # Placeholder for video feed
    snapshot_button = st.sidebar.button("Capture Snapshot")  # Snapshot button
    snapshot_frame = None  # Store captured frame

    FRAME_RATE = 10
    last_time = time.time()

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading from the camera.")
            break

        # Process frame at a fixed frame rate
        current_time = time.time()
        if current_time - last_time >= 1 / FRAME_RATE:
            last_time = current_time

            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract face for BMI prediction
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (128, 128))
                face_img = np.expand_dims(face_img, axis=0) / 255.0

                # Predict BMI
                predicted_bmi = bmi_model.predict(face_img)[0][0]
                predicted_bmi = (predicted_bmi * bmi_std) + bmi_mean
                predicted_bmi = np.clip(predicted_bmi, 10, 60)

                # Classify BMI and display
                bmi_category, _ = classify_bmi(predicted_bmi)
                cv2.putText(frame, f"BMI: {predicted_bmi:.2f} ({bmi_category})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert frame to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Capture snapshot
            if snapshot_button:
                snapshot_frame = frame.copy()
                st.success("Snapshot captured!")
                break

    cap.release()
    st.session_state.camera_active = False
    st.info("Camera stopped.")

    # Process snapshot if captured
    if snapshot_frame is not None:
        process_snapshot(snapshot_frame)




# ------------------------------- #
#          IMAGE UPLOAD LOGIC     #
# ------------------------------- #

def upload_image():
    """Upload an image for BMI prediction."""
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        try:
            # Read and display uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, caption="Uploaded Image", channels="RGB", width=300)

            # Convert to grayscale for face detection
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            if len(faces) == 0:
                st.warning("No face detected in the uploaded image.")
                return

            for (x, y, w, h) in faces:
                # Extract face for BMI prediction
                face_img = image[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (128, 128))
                face_img = np.expand_dims(face_img, axis=0) / 255.0

                # Predict BMI
                predicted_bmi = bmi_model.predict(face_img)[0][0]
                predicted_bmi = (predicted_bmi * bmi_std) + bmi_mean
                predicted_bmi = np.clip(predicted_bmi, 10, 60)

                # Classify BMI
                bmi_category, health_advice = classify_bmi(predicted_bmi)

                # Display results
                st.subheader("Predicted BMI:")
                st.write(f"{predicted_bmi:.2f} ({bmi_category})")
                st.subheader("Health Insights:")
                st.write(health_advice)

                # Store BMI record in the database
                if "username" in st.session_state:
                    add_bmi_record(st.session_state.username, predicted_bmi, bmi_category)

        except Exception as e:
            st.error(f"Error processing the image: {e}")
    else:
        st.info("Upload an image to get started.")



# ------------------------------- #
#       BMI HISTORY LOGIC         #
# ------------------------------- #

def show_bmi_history(username):
    """Display BMI history as a table and graph."""
    st.subheader("BMI History")
    records = get_bmi_history(username)

    if not records:
        st.info("No BMI records found.")
        return

    # Display records in a table
    st.table(records)

    # Prepare data for plotting
    dates = [record[2] for record in records]
    bmi_values = [record[0] for record in records]

    # Plot BMI history
    plt.figure(figsize=(10, 6))
    plt.plot(dates, bmi_values, marker="o", linestyle="-", color="b")
    plt.title("BMI History Over Time")
    plt.xlabel("Date")
    plt.ylabel("BMI")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)


# ------------------------------- #
#          STREAMLIT UI           #
# ------------------------------- #

def main_app():
    """Main application after login."""
    st.title("BMI Prediction App")
    st.sidebar.title(f"Welcome, {st.session_state.username}")

    # Add Logout button to the sidebar
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()  # Restart the app

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    option = st.sidebar.selectbox("Select an option", ["Live Camera Preview", "Upload Image", "View BMI History"])

    if option == "Live Camera Preview":
        start_button = st.sidebar.button("Start Camera")
        stop_button = st.sidebar.button("Stop Camera")

        if start_button:
            if not st.session_state.camera_active:
                st.session_state.camera_active = True
                run_camera()

        if stop_button:
            st.session_state.camera_active = False
            st.info("Stopping Camera...")

    elif option == "Upload Image":
        upload_image()

    elif option == "View BMI History":
        show_bmi_history(st.session_state.username)



def main():
    """Main Streamlit application."""
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "register" not in st.session_state:
        st.session_state.register = False

    if st.session_state.register:
        signup()
    elif not st.session_state.logged_in:
        login()
    else:
        main_app()



if __name__ == "__main__":
    main()
