import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import base64
import google.generativeai as genai
import os

def set_background_and_text(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        font-weight: 900;
    }}
    .stSidebar, .css-1d391kg {{
        background-color: #0d2b42 !important;
        color: #ffffff !important;
        font-weight: 900;
    }}
    .stSidebar .css-1v3fvcr, .stSidebar .css-q8sbsg {{
        color: #ffffff !important;
        font-weight: 900;
    }}
    * {{
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 900 !important;
    }}
    .stFileUploader {{
        background-color: #0d2b42 !important;
        border: 2px dashed #00b4d8 !important;
        border-radius: 10px;
        color: #ffffff !important;
        font-weight: 900 !important;
    }}
    .stFileUploader label, .stFileUploader div, .stFileUploader span {{
        color: #004080 !important;
        font-weight: 900 !important;
    }}
    header {{
        background-color: #0d2b42 !important;
        color: #ffffff !important;
        font-weight: 900;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
def preprocess_image(img, size=(224, 224)):
    img = img.convert('L')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def classify_ui():
    model = BaselineCNN()
    model.load_state_dict(torch.load('Model_Status/Breast_Cancer.pth', map_location=torch.device('cpu')))
    model.eval()
    set_background_and_text("Images_App/proxy-image.jpeg")

    st.title("breast Cancer Classifier")
    st.write("Upload an image to classify it as Normal or breast Cancer.")

    main_class_names = ["Normal", "breast Cancer"]

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.item()

        predicted_class = 1 if prediction >= 0.5 else 0
        confidence = prediction if predicted_class == 1 else 1 - prediction
        
        st.write(f"**Predicted Class:** {main_class_names[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.4f}")


def chat_ui():
    genai.configure(api_key=st.secrets["aModel_key"])
    
    st.title("Chat with the Breast Cancer Classifier")
    st.write("Ask me anything about breast cancer classification!")
    flash = genai.GenerativeModel('gemini-1.5-flash')
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "أنا مساعد طبي، أقدم ردًا دقيقًا بلغة المستخدم👋"}
        ]

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept new user input
    if prompt := st.chat_input("Ask me anything about breast cancer"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Keep only last 5 messages for context
        context_messages = st.session_state.messages[-5:]

        # Format them as input (adjust depending on your model's expected format)
        formatted_context = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in context_messages]
        )

        # Pass context to the model
        response = flash.generate_content(formatted_context)
        reply = response.text

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    
    
st.sidebar.image("Images_App/faculty_of_Appied_Medical_Sciences.jpeg")
st.sidebar.markdown("# Faculty of Health Applied Medical Science")



st.sidebar.markdown("## **Supervisor:** Dr. Diana Abbas Al Sherif")
st.sidebar.markdown("**Team Members:**")
st.sidebar.markdown("1. Mohamed Osama")
st.sidebar.markdown("2. Shahed Hazem")
st.sidebar.markdown("3. Sama Mustafa")
st.sidebar.markdown("4. Duaa Ali ")
st.sidebar.markdown("5. Aya Abdel Basset ")
st.sidebar.markdown("6. Khaled Nour El Din")
st.sidebar.markdown("7. Mohamed Ibrahim")

tab = st.sidebar.radio("📚 App Views", [
    "🔬 Model classification",
    "🧠🤖 Chat with chatbot",
])

if tab == "🔬 Model classification":
    classify_ui()

elif tab == "🧠🤖 Chat with chatbot":
    chat_ui()