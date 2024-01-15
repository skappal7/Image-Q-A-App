import streamlit as st
import requests
import base64
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/impira/layoutlm-document-qa"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"An error occurred during the API request: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def image_to_text_with_layoutlm(image_path, question):
    try:
        with open(image_path, "rb") as f:
            img = f.read()

        payload = {
            "inputs": {
                "image": base64.b64encode(img).decode("utf-8"),
                "question": question
            }
        }

        output = query(payload)
        return output
    except FileNotFoundError:
        return f"Image file not found: {image_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    st.title("Image to Text Converter with LayoutLM Document QA")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    question = st.text_input("Enter a question about the image:")

    if uploaded_file is not None and question:
        # Display the selected image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform image to text conversion with LayoutLM Document QA
        if st.button("Convert Image to Text"):
            result = image_to_text_with_layoutlm(uploaded_file.name, question)
            st.subheader("Text Extracted:")
            st.write(result)

if __name__ == "__main__":
    main()
