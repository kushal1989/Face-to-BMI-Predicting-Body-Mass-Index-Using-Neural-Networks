# 🧠 Face-to-BMI: AI-Powered Body Mass Index Prediction

**Face-to-BMI** is a deep learning application that estimates an individual's Body Mass Index (BMI) using facial images. Leveraging a pre-trained EfficientNetV2 model, this tool provides real-time BMI predictions via live camera feed or uploaded images. It also offers personalized health insights and maintains a user-specific BMI history.

---

## Login
![BMI_Prediction_Streamlit_App](https://github.com/user-attachments/assets/d346afa9-f420-40bf-add5-fcc1b6fe7eb4)

## Predicting via Image
![BMI_Prediction_Streamlit_App (1)](https://github.com/user-attachments/assets/b3445f2b-2eac-4317-a4ab-254e7bcde54c)

## Predicting via Live camera Preview
https://github.com/user-attachments/assets/d0cece41-14db-492f-bb82-64822c084b3e

---

## 🚀 Features

- **Real-Time BMI Prediction**: Capture facial images using your webcam for instant BMI estimation.
- **Image Upload**: Upload a photo to receive a BMI prediction.
- **User Authentication**: Secure login and registration to track personal BMI history.
- **BMI History Log**: View and analyze your BMI trends over time with graphical insights.
- **Health Insights**: Receive personalized advice based on your BMI category.

---

## 🧠 Model Training

The BMI prediction model was trained on the [Illinois DOC Labeled Faces Dataset](https://www.kaggle.com/datasets/davidjfisher/illinois-doc-labeled-faces-dataset), which contains over 68,000 labeled mugshots. The dataset includes both frontal and side-profile images, each annotated with metadata such as date of birth and name, making it ideal for training facial recognition models.

To train the model:

1. Run `EfficientNetV2.py` to train the model.
2. After training, you will receive a file named `Optimized_EfficientNetV2_Model.keras`. 
3. Replace the `model_path` in `app.py` with the path to this saved model file:

    ```python
    model_path = r"C:\path\to\your\Optimized_EfficientNetV2_Model.keras"

## 📊 BMI Classification Categories
- **Underweight**: BMI < 18.5  
  - *Advice*: Consider consuming a nutrient-rich diet with more calories and engaging in strength-building exercises.
- **Normal**: 18.5 ≤ BMI < 25  
  - *Advice*: Maintain your current lifestyle with a balanced diet and regular physical activity.
- **Overweight**: 25 ≤ BMI < 30  
  - *Advice*: Incorporate more physical activity into your routine and consider reducing high-calorie food intake.
- **Obese**: BMI ≥ 30  
  - *Advice*: Consult a healthcare provider for a tailored weight-loss plan, including diet and exercise.
