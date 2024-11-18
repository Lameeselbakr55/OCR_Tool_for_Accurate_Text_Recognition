import numpy as np
import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image as PILImage
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
# Function to load and display the image
from io import BytesIO
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load models once
id_card_model = YOLO('C:/Users/Lamees/Desktop/OCR/detect_id_card.pt')
credit_card_model = YOLO('C:/Users/Lamees/Desktop/OCR/best_cr3.pt')
object_detection_model = YOLO('C:/Users/Lamees/Desktop/OCR/nid_objects.pt')

# Function to calculate the squared distance from the origin
def distance_from_origin(point):
    x, y = point[0][0], point[0][1]  
    return x**2 + y**2

# Function to preprocess the image
# Preprocessing function for better OCR

# def preprocess_image(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Apply thresholding to convert the image to binary
#     _, binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     # Return preprocessed image
#     return binary

# Function to preprocess the image for OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mean_intensity = np.mean(blur)
    threshold_value = max(0, min(255, mean_intensity + 20))
    _, binary = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

# Function to detect the ID card and pass it to the existing code
def detect_id_card(image):
    cropped_image = None  # Initialize cropped_image to None
    class_name = None  # Initialize class_name to None

    # Perform inference to detect the ID card
    id_card_results = id_card_model(image)

    # Crop the ID card from the image
    for result in id_card_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]
            class_id = int(box.cls[0].item())  # Class ID
            class_name = result.names[class_id]  # Changed to int(cls) for correct indexing


    return cropped_image,class_name


 # Function to plot image with bounding boxes
def plot_image_with_boxes(image, boxes, title="Image with Boxes"):
    # Convert the image to RGB format for proper display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw the boxes on the image
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle in red

    # Display the image in Streamlit
    st.image(image_rgb, caption=title, use_column_width=True)


# Function to expand bounding box height only
def expand_bbox_height(bbox, x_scale = 1 , y_scale=1.2, image_shape=None):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_width = int(width * x_scale)
    new_height = int(height * y_scale)
    new_x1 = max(center_x - new_width // 2, 0)
    new_x2 = min(center_x + new_width // 2, image_shape[1])
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, image_shape[0])
    return [new_x1, new_y1, new_x2, new_y2]

# Function to draw and crop region of interest (ROI)
def draw_and_crop_roi(image, x1, y1, x2, y2):
    # Crop the ROI from the original image
    roi = image[y1:y2, x1:x2]
    return roi

 # Function to process the cropped image
def process_nid_card_image(cropped_image):
    # Create a copy of the image for plotting expanded boxes
    expanded_image = cropped_image.copy()

    # Variables to store extracted values
    extracted_data = []
    extracted_crops = []

    # Load the trained YOLO model for objects (fields) detection
    results = object_detection_model(cropped_image)
    # Collect original bounding boxes
    original_boxes = []

    # Loop through the results
    for result in results:
        boxes = [box.xyxy[0].tolist() for box in result.boxes]
        original_boxes.extend(boxes)

        # Iterate through each detected box and class
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()  # Original bounding box
            class_id = int(box.cls[0].item())  # Class ID
            class_name = result.names[class_id]  # Class name

            # Expand the bounding box height
            if class_name == 'FN':
               expanded_bbox = expand_bbox_height(bbox, x_scale=1.5,y_scale=1.2, image_shape=cropped_image.shape)
            else:
              expanded_bbox = expand_bbox_height(bbox, x_scale=1, y_scale=1.2, image_shape=cropped_image.shape)

            # Append the class name and expanded bounding box to the list
            extracted_data.append([class_name, expanded_bbox])

            # Draw the expanded bounding boxes on the copy image
            x1, y1, x2, y2 = map(int, expanded_bbox)  # Ensure coordinates are integers
            #cv2.rectangle(expanded_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cropped_roi = draw_and_crop_roi(cropped_image, x1, y1, x2, y2)

            # Append the class name and cropped image (ROI) to the list
            extracted_crops.append((class_name, cropped_roi))



    # Plot original image with original bounding boxes
    # plot_image_with_boxes(cropped_image.copy(), original_boxes)

    # Plot expanded image with expanded bounding boxes
    # plot_image_with_boxes(expanded_image, [box for _, box in extracted_data])

    return extracted_crops

def thresholding_cropped_images(extracted_crops,card_type):
    """
    Process a list of extracted crops and apply binary thresholding
    for specific class names.

    Parameters:
    - extracted_crops: List of tuples, where each tuple contains a
                       class name and a cropped image.

    Returns:
    - List of tuples containing class names and their corresponding
      binary thresholded images.
    """

    binary_images = []
    back_threshold_classes = {'Ex_d', 'Ex_m','Ex_y', 'Gender', 'Religion','Marital_status','Job','Job2','Id'}
    front_threshold_classes = {'FN', 'LN', 'Add1','Add2', 'Id'}
    if card_type == 'front-up':
      threshold_clasees = front_threshold_classes
    else:
      threshold_clasees = back_threshold_classes

    kernel = np.ones((5, 5), np.uint8)
    for class_name, crop in extracted_crops:
        if class_name in threshold_clasees:
            gray_cropped = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            median = cv2.medianBlur(gray_cropped, 3)
            th_cropped = cv2.adaptiveThreshold(
                median,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                111,
                25
            )
            #gray_image = cv2.bilateralFilter(th_cropped,1,100,100)


            binary_images.append((class_name, th_cropped))


    return binary_images

def extract_id_information(binary_thresholds,card_type):
    # fields = ["firstName", "lastName", "address", "nid"]
    back_fields =  ['Ex_d', 'Ex_m','Ex_y', 'Gender', 'Religion','Marital_status','Job','Job2','Id']
    front_fields = ["FN", "LN", "Add1", "Add2","Id"]
    alpha_fields = ["LN", "Add1", "Add2","firstName", "lastName", "address","Gender", "Religion","Marital_status","Job","Job2"]

    if card_type == 'front-up':
      fields = front_fields
    else:
      fields = back_fields

    data = {field: "" for field in fields}
    for class_name, th_cropped in binary_thresholds:
        if class_name in alpha_fields :
            extracted_text = pytesseract.image_to_string(
                th_cropped,
                config='-c tessedit_create_utf8=1 --oem 3 --psm 12',
                lang='ara'
            )
        elif class_name == 'FN':
            extracted_text = pytesseract.image_to_string(
                th_cropped,
                config='-c tessedit_create_utf8=1 --oem 3 --psm 7',
                lang='ara'
            )
        else:
            extracted_text = pytesseract.image_to_string(
                th_cropped,
                config='-c tessedit_char_whitelist=1234567890 --psm 8',
                lang='hin'
            )
        cleaned_text = " ".join(extracted_text.splitlines()).strip()
        if cleaned_text:
            if data[class_name]:
                data[class_name] += " " + cleaned_text
            else:
                data[class_name] = cleaned_text
    df = pd.DataFrame([data])
    return df


# Function to extract text from image using Tesseract
def extract_text_from_image(image, bbox, lang, config):
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    processed_image = preprocess_image(cropped_image)
    text = pytesseract.image_to_string(processed_image, lang=lang, config=config)
    return text.strip()

# Language and config dictionary for different labels
label_lang_config = {
    'Card number': {'lang': 'cardnumv2+train+eng', 'config': '--psm 6'},
    'Cardholder': {'lang': 'eng', 'config': '--psm 7'},
    'Expiry date': {'lang': 'exp+numbers+eng', 'config': '--psm 7 -c tessedit_char_whitelist=0123456789/'},
}

# def credit_card_process(image):
#             # Use YOLOv8 to detect bounding boxes
#         results = credit_card_model (image)
#         data = []

#         # Loop through each detection and apply Tesseract
#         for result in results:  # Loop over the results (each result is for one image)
#             for box in result.boxes:  # Loop over detected boxes
#                 x_min, y_min, x_max, y_max = box.xyxy[0]  # Get bounding box coordinates
#                 label = result.names[int(box.cls)]  # Get the class label
#                 # Extract the cropped image and apply OCR to extract text
#                 text = extract_text_from_image(image, (x_min, y_min, x_max, y_max))
#                 data.append([label, text])
#                 # Print detected label and text
#                 #st.write(f"Detected: {label} -> Extracted Text: {text}")
#                 # Optionally, display the bounding box and detected text
#                 # Optionally, draw the bounding box on the image (for display purposes)
#                 image_with_box = cv2.rectangle(image.copy(), (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
#                 image_with_text = cv2.putText(image_with_box, text, (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#                 st.image(image_with_text, caption="Image with Bounding Boxes and Extracted Text", use_column_width=True)
  
#         # Convert the extracted data into a DataFrame for display
#         if data:
#             df = pd.DataFrame(data, columns=['Label', 'Extracted Text'])
#             st.write(df)


# Function to handle image processing and data extraction
def credit_card_process(image):
    #image_np = np.array(image.convert('RGB'))
    results = credit_card_model(image)
    data = {'card_number': None, 'expiry_date': None, 'cardholder': None}

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            label = result.names[int(box.cls)]

            if label in label_lang_config:
                lang = label_lang_config[label]['lang']
                config = label_lang_config[label]['config']
            else:
                lang = 'eng'
                config = '--psm 7'

            text = extract_text_from_image(image, (x_min, y_min, x_max, y_max), lang, config)

            if label == 'Card number':
                data['card_number'] = text
            elif label == 'Expiry date':
                data['expiry_date'] = text
            elif label == 'Cardholder':
                data['cardholder'] = text
        credit_df = pd.DataFrame([data])


    return credit_df


def load_img(image_data):
    try:
        # Use BytesIO to read the uploaded image file
        image = np.array(PILImage.open((image_data)).convert('RGB'))
        #st.image(image, caption="Original Image", use_column_width=True)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


# Main function
def main():
        # Display logos
    col1, col2, col3 = st.columns([3,3, 2])
    with col1:
        st.image("C:/Users/Lamees/Desktop/OCR/q.png", width=170)
    with col2:
        st.image("C:/Users/Lamees/Desktop/OCR/D.png", width=100)
    with col3:
        st.image("C:/Users/Lamees/Desktop/OCR/NBE.png", width=80)
    

    st.title("Credit Cards or National ID OCR")
    st.write("Upload the card image to detect.")
  
    # Create a checkbox to choose if operations will de done on specfic column or all dataset
    selected_item = st.radio('Select card you want to analyze first :', ['Credit Card', 'National ID'],key="Rs_1")
    # Check if the checkbox is selected
    
    #file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"],label_visibility="collapsed", accept_multiple_files=False)

    # Allow users to upload images or use their mobile camera
    file = st.file_uploader("Upload an image file or use your camera", type=["png", "jpg", "jpeg"],
                             label_visibility="collapsed", accept_multiple_files=False)
    if file is not None:
        
        image = load_img(file)  # Load and display the uploaded image
        if selected_item == 'Credit Card' : 
            st.image(image, caption="Original Image", use_column_width=True)

            credit_df = credit_card_process(image)   
            st.write(credit_df)    
            # Display the extracted text in structured fields
            # if data['card_number']:
            #     st.markdown('<div class="header">Card Number:</div>', unsafe_allow_html=True)
            #     st.text_area('', value=data['card_number'], height=50, key='card_number',
            #                 help="Extracted card number", placeholder="Card number...",
            #                 max_chars=50)

            # if data['expiry_date']:
            #     st.markdown('<div class="header">Expiry Date:</div>', unsafe_allow_html=True)
            #     st.text_area('', value=data['expiry_date'], height=50, key='expiry_date',
            #                 help="Extracted expiry date", placeholder="Expiry date...",
            #                 max_chars=50)

            # if data['cardholder']:
            #     st.markdown('<div class="header">Card Holder:</div>', unsafe_allow_html=True)
            #     st.text_area('', value=data['cardholder'], height=50, key='cardholder',
            #                 help="Extracted card holder name", placeholder="Cardholder name...",
            #                 max_chars=100)

        else : 
            detected_card , card_type = detect_id_card(image)
            st.image(detected_card, caption="cropped Image", use_column_width=True)

            extraced_crops = process_nid_card_image(detected_card)
            binary_thresholds = thresholding_cropped_images(extraced_crops,card_type)
            # st.write(binary_thresholds)
            #for i, (class_name, binary_image) in enumerate(binary_thresholds):
             #   st.image(binary_image)
 
            id_df = extract_id_information(binary_thresholds,card_type)  
            st.write(id_df) 

if __name__ == "__main__":
    main()
