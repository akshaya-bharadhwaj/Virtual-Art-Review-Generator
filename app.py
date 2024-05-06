# app.py
import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import random
import itertools
from colorsys import rgb_to_hls, hls_to_rgb  # Import for color relationships
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import traceback





def analyze_composition(uploaded_file_comp):
    # Read the image
    if uploaded_file_comp is not None:
        # Open the image file
        image = cv2.imdecode(np.asarray(bytearray(uploaded_file_comp.read()),dtype = np.uint8),1)
        
        # Get image dimensions
        height, width, _ = image.shape

        # Define the rule of thirds lines
        third_h = height // 3
        third_w = width // 3

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection using the Canny edge detector
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours were found
        if contours:
            # Calculate the area of each contour
            areas = [cv2.contourArea(contour) for contour in contours]

            # Find the index of the contour with the maximum area
            max_area_index = np.argmax(areas)

            # Get the contour with the maximum area
            max_area_contour = contours[max_area_index]

            # Calculate the total area of all contours
            total_area = sum(areas)

            # Calculate the percentage of the image covered by the largest contour
            coverage_percentage = (areas[max_area_index] / total_area) * 100

            # Get the centroid of the largest contour
            M = cv2.moments(max_area_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = width // 2, height // 2

            # Check if the centroid is within the central rectangle of the rule of thirds grid
            if third_w < cx < 2 * third_w and third_h < cy < 2 * third_h:
                feedback = "Your composition is well-balanced and follows the rule of thirds."
            else:
                feedback = "Consider adjusting the composition to align with the rule of thirds for a more balanced look."

            # Symmetry detection
            symmetry_feedback= detect_symmetry(image)
            feedback += "\n\nSymmetry:\n" + symmetry_feedback
            
            # Balance detection
            balance_feedback = detect_balance(image)
            feedback += "\n\nBalance:\n" + balance_feedback

            return feedback

        else:
            return "No contours found. Please choose an image with clear visual elements."
    else:
        return "No image file uploaded. Please choose an image for composition analysis."


    

def detect_symmetry(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection using the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None:
        horizontal_lines = []
        vertical_lines = []
        for line in lines:
            rho, theta = line[0]
            if np.degrees(theta) < 10 or np.degrees(theta) > 170:
                horizontal_lines.append((rho, theta))
            elif 80 < np.degrees(theta) < 100:
                vertical_lines.append((rho, theta))

        # Determine the strength of symmetry based on the number of detected lines
        if len(horizontal_lines) > 1 and len(vertical_lines) > 1:
            return "Strong bilateral symmetry detected. The composition exhibits a high degree of balance and visual harmony, with prominent symmetrical elements enhancing its aesthetic appeal."
        elif len(horizontal_lines) == 1 or len(vertical_lines) == 1:
            return "Moderate symmetry detected. While some symmetrical elements are present, further adjustments may be needed to achieve optimal balance and visual coherence."
        else:
            return "Weak or no detectable symmetry. The composition lacks clear symmetrical elements, potentially leading to a less cohesive visual experience."
    else:
        return "No symmetrical elements detected. The composition does not exhibit any evident symmetry, suggesting a more dynamic or asymmetrical arrangement of elements."

def detect_balance(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection using the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Calculate the total number of edge pixels
    total_edge_pixels = np.count_nonzero(edges)

    # Calculate the center of mass of edge pixels
    M = cv2.moments(edges)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = image.shape[1] // 2, image.shape[0] // 2

    # Calculate the distance of the center of mass from the center of the image
    distance_from_center = np.sqrt((cx - image.shape[1] // 2) ** 2 + (cy - image.shape[0] // 2) ** 2)

    # Calculate the balance score (closer to 0 means more balanced)
    balance_score = distance_from_center / np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)

    # Determine the strength of balance based on the balance score
    if balance_score < 0.05:
        return "Balance: The composition is perfectly balanced. The distribution of visual elements creates a harmonious and cohesive visual experience."
    elif balance_score < 0.1:
        return "Balance: The composition is well-balanced. Visual elements are distributed evenly, contributing to a pleasing and balanced aesthetic."
    elif balance_score < 0.2:
        return "Balance: The composition is reasonably balanced. While some minor adjustments may be beneficial, overall, the visual weight is distributed effectively."
    else:
        return "Balance: The composition may benefit from adjustments for better balance. Certain areas may appear more visually dominant, potentially detracting from the overall harmony and balance."



def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def evaluate_color_relationships(dominant_colors_hex):
    if dominant_colors_hex:
        dominant_colors_rgb = [hex_to_rgb(hex_color) for hex_color in dominant_colors_hex]


        # Contrast analysis
        contrast_ratios = []
        for i in range(len(dominant_colors_rgb) - 1):
            for j in range(i + 1, len(dominant_colors_rgb)):
                color1_rgb = dominant_colors_rgb[i]
                color2_rgb = dominant_colors_rgb[j]
                brightness_ratio = max(color1_rgb) / max(color2_rgb)
                contrast_ratios.append(brightness_ratio)

        average_contrast_ratio = sum(contrast_ratios) / len(contrast_ratios)
        feedback = f"\n\nAverage contrast ratio: {average_contrast_ratio:.2f}. The contrast ratio measures the difference in brightness between the foreground and background colors in your artwork. It helps determine the visual clarity and readability of your composition. \n"

        if average_contrast_ratio < 1.6:
            feedback += "Consider increasing contrast for stronger visual impact."
        elif average_contrast_ratio > 4.5:
            feedback += "Consider softening contrast for a more balanced look."
        else:
            feedback += "Contrast is well-balanced."

        # Harmony analysis (using hue, lightness, and saturation)
        hls_values = [rgb_to_hls(*color) for color in dominant_colors_rgb]
        hue_differences = [abs(h1[0] - h2[0]) for h1, h2 in itertools.combinations(hls_values, 2)]

        average_hue_difference = sum(hue_differences) / len(hue_differences)

        if average_hue_difference < 30:
            feedback += "\nColors are harmonious and visually appealing."
        elif average_hue_difference > 150:
            feedback += "\nColors are intentionally dissonant, creating visual tension."
        else:
            feedback += "\nConsider adjusting colors to enhance harmony or create intentional dissonance."

        # Complements identification
        complements = []
        for i in range(len(dominant_colors_rgb)):
            hue = hls_values[i][0]
            complement_hue = (hue + 180) % 360
            complement_rgb = hls_to_rgb(complement_hue, hls_values[i][1], hls_values[i][2])
            complements.append("#" + "".join(str(int(component)).zfill(2) for component in complement_rgb))

        # feedback += f"\nComplementary color pairs:\n{', '.join(complements)}"

        # Color scheme recognition
        color_scheme_info = identify_color_scheme(hls_values)
        if color_scheme_info:
            feedback += f"\n\nColor scheme: {color_scheme_info['description']}\n"

        return feedback
    else:
        return "No dominant colors identified in the image."

def identify_color_scheme(dominant_colors_hls):
    scheme = None
    scheme_info = {}

    # Complementary scheme
    if len(dominant_colors_hls) == 2 and abs(dominant_colors_hls[0][0] - dominant_colors_hls[1][0]) >= 150:
        scheme = "Complementary"

    # Analogous scheme
    elif all(abs(h1[0] - h2[0]) <= 30 for h1, h2 in itertools.combinations(dominant_colors_hls, 2)):
        scheme = "Analogous"

    # Triadic scheme
    elif len(dominant_colors_hls) == 3 and all(abs(h1 - h2) >= 120 for h1, h2 in itertools.combinations(dominant_colors_hls, 2)):
        scheme = "Triadic"

    # Tetradic scheme
    elif len(dominant_colors_hls) == 4 and (
        all(abs(h1 - h2) >= 60 for h1, h2 in itertools.combinations(dominant_colors_hls, 2)) or
        all(abs(h1 - h2) >= 120 for h1, h2 in itertools.combinations(dominant_colors_hls, 2))
    ):
        scheme = "Tetradic"

    # Monochromatic scheme
    elif all(abs(h1 - h2) <= 5 for h1, h2 in itertools.combinations(dominant_colors_hls, 2)):
        scheme = "Monochromatic"

    if scheme:
        scheme_info = {
            "scheme": scheme,
            "description": f"The artwork uses a {scheme} color scheme, which {get_scheme_description(scheme)}."
        }

    return scheme_info

def get_scheme_description(scheme):
    descriptions = {
        "Complementary": "creates high contrast and visual excitement",
        "Analogous": "offers a harmonious and serene feel",
        "Triadic": "provides balance and visual interest",
        "Tetradic": "creates a dynamic and complex look",
        "Monochromatic": "conveys unity and simplicity"
    }
    return descriptions.get(scheme, "has a unique color combination")


def evaluate_color_harmony_advanced(uploaded_file_color):
    # Check if a file was uploaded
    if uploaded_file_color is not None:
        # Open the image file
        image = cv2.imdecode(np.asarray(bytearray(uploaded_file_color.read()), dtype=np.uint8), 1)

        # Reshape the image to a 2D array of pixels
        pixels = image.reshape((-1, 3))

        # Use k-means clustering to find dominant colors
        num_colors = 5  
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)

        # Get the RGB values of the dominant colors
        dominant_colors = kmeans.cluster_centers_.astype(int)

        # Convert RGB to hexadecimal for better representation
        dominant_colors_hex = ['#%02x%02x%02x' % (color[2], color[1], color[0]) for color in dominant_colors]

        # Create a color palette
        color_palette = [tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)) for color in dominant_colors_hex]

        # Visualize the color palette using matplotlib
        fig, ax = plt.subplots(figsize=(num_colors, 1))
        ax.imshow([color_palette], aspect='auto')
        ax.axis('off')

        # Convert the plot to a PNG image
        image_streamlit = BytesIO()
        plt.savefig(image_streamlit, format='png')
        plt.close()
        image_streamlit.seek(0)

        # Display the color palette in Streamlit
        # st.image(image_streamlit, caption='Dominant Colors')

        # Provide feedback based on the average contrast ratio
        feedback = f"Dominant colors:\n"
        feedback += f'![color](data:image/png;base64,{base64.b64encode(image_streamlit.read()).decode()})\n'
        
        feedback += evaluate_color_relationships(dominant_colors_hex)
        feedback += "\n"

        return feedback
    else:
        return "No image file uploaded. Please choose an image for color harmony evaluation."

    
# Function to evaluate colour distribution
def evaluate_color_distribution(uploaded_file_color):
    # Read the image
    if uploaded_file_color is not None:
        # Open the image file
        image = cv2.imdecode(np.asarray(bytearray(uploaded_file_color.read()), dtype=np.uint8), 1)

        # Get image dimensions
        height, width, _ = image.shape

        # Define the rule of thirds lines
        third_h = height // 3
        third_w = width // 3

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract the hue component
        hue_channel = hsv_image[:, :, 0]

        # Threshold values for hues corresponding to different thirds
        lower_third_hue = 0
        upper_third_hue = 60  

        # Create masks for each third of the image
        mask_lower_third = cv2.inRange(hue_channel, lower_third_hue, upper_third_hue)
        mask_middle_third = cv2.inRange(hue_channel, upper_third_hue, 2 * upper_third_hue)
        mask_upper_third = cv2.inRange(hue_channel, 2 * upper_third_hue, 180)

        # Calculate the percentage of pixels in each third
        lower_third_percentage = (np.count_nonzero(mask_lower_third) / mask_lower_third.size) * 100
        middle_third_percentage = (np.count_nonzero(mask_middle_third) / mask_middle_third.size) * 100
        upper_third_percentage = (np.count_nonzero(mask_upper_third) / mask_upper_third.size) * 100

        # Provide feedback based on the color distribution within the rule of thirds
        feedback = "Color distribution within the rule of thirds:\n"

        # Feedback for the lower third
        if lower_third_percentage < 20:
            feedback += "The lower third lacks color presence. Consider adding more colors to this area.\n"
        elif lower_third_percentage > 40:
            feedback += "The lower third has a dominant color presence. Ensure it aligns with your artistic intent.\n"
        else:
            feedback += "The color distribution in the lower third is balanced.\n"

        # Feedback for the middle third
        if middle_third_percentage < 20:
            feedback += "The middle third lacks color presence. Consider adding more colors to this area.\n"
        elif middle_third_percentage > 40:
            feedback += "The middle third has a dominant color presence. Ensure it aligns with your artistic intent.\n"
        else:
            feedback += "The color distribution in the middle third is balanced.\n"

        # Feedback for the upper third
        if upper_third_percentage < 20:
            feedback += "The upper third lacks color presence. Consider adding more colors to this area.\n"
        elif upper_third_percentage > 40:
            feedback += "The upper third has a dominant color presence. Ensure it aligns with your artistic intent.\n"
        else:
            feedback += "The color distribution in the upper third is balanced.\n"

        return feedback
    else:
        return "No image file uploaded for color distribution analysis."

# Function for colour temperature analysis
def evaluate_color_temperature(uploaded_file_color):
    # Read the image
    if uploaded_file_color is not None:
        # Open the image file
        image = cv2.imdecode(np.asarray(bytearray(uploaded_file_color.read()), dtype=np.uint8), 1)

        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # Split LAB channels
        l_channel, _, _ = cv2.split(lab_image)

        # Calculate average brightness (L channel)
        average_brightness = np.mean(l_channel)

        # Provide feedback based on color temperature
        temperature_feedback = "Color temperature analysis:\n"

        if average_brightness < 128:
            temperature_feedback += "The image tends to be cooler. \n"
        else:
            temperature_feedback += "The image tends to be warmer. \n"

        return temperature_feedback
    else:
        return "No image file uploaded for color temperature analysis."



# Function to extract dominant colors from an image using K-means clustering
def extract_color_palette(uploaded_file_color, num_colors=5):
    if uploaded_file_color is not None:
        # Open the uploaded image file
        image = cv2.imdecode(np.asarray(bytearray(uploaded_file_color.read()), dtype=np.uint8), 1)

        # Reshape the image to a 2D array of pixels
        pixels = image.reshape((-1, 3))

        # Use K-means clustering to find dominant colors
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)

        # Get the RGB values of the dominant colors
        dominant_colors = kmeans.cluster_centers_.astype(int)

        # Convert RGB to hexadecimal for better representation
        dominant_colors_hex = ['#%02x%02x%02x' % (color[2], color[1], color[0]) for color in dominant_colors]

        return dominant_colors_hex
    else:
        return None

# Function to display color palette
def display_color_palette(palette):
    for color in palette:
        st.markdown(f'<div style="width: 30px; height: 30px; background-color: {color};"></div>', unsafe_allow_html=True)


# Streamlit app main function
def main():
    st.sidebar.title("Analysis Options")
    selected_feedback = st.sidebar.radio("Select an Option", ["Composition Analysis", "Color Harmony Evaluation", "Color Distribution Analysis", "Color Temperature Analysis", "Color Palette Extractor"])

    st.title("Virtual Art Review")
    st.subheader("How to Use:")
    st.write("1. Choose an image file for each analysis using the file uploaders below.")
    st.write("2. Once an image is uploaded, the analysis results will be displayed.")
    st.write("3. Review the feedback provided for each analysis to gain insights into your artwork's composition and color aesthetics.")

    if selected_feedback == "Composition Analysis":
        st.header("Composition Analysis")
        st.write("Composition analysis examines the arrangement of elements in your artwork. It evaluates factors such as balance, symmetry, and focal points to provide feedback on the overall layout.")
        uploaded_file = st.file_uploader("Choose an image for analysis", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            result = analyze_composition(uploaded_file)
            st.write(result)
            st.write("*Please note: The feedback provided is objective and based on analysis algorithms. Artists should consider their creative intent when interpreting and acting on the feedback.*")

    elif selected_feedback == "Color Harmony Evaluation":
        st.header("Color Harmony Evaluation")
        st.write("Color harmony evaluation assesses the color relationships in your artwork.It analyzes factors such as contrast, harmony, and color schemes to provide insights into the visual appeal and coherence of your color palette.")
        uploaded_file = st.file_uploader("Choose an image for analysis", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            result = evaluate_color_harmony_advanced(uploaded_file)
            st.write(result)
            st.write("*Please note: The feedback provided is objective and based on analysis algorithms. Artists should consider their creative intent when interpreting and acting on the feedback.*")

    elif selected_feedback == "Color Distribution Analysis":
        st.header("Color Distribution Analysis (Rule of Thirds)")
        st.write("Color distribution analysis examines how colors are distributed across different sections of your artwork, particularly following the rule of thirds. It provides feedback on the balance and impact of color placement within the composition.")
        uploaded_file = st.file_uploader("Choose an image for analysis", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            result = evaluate_color_distribution(uploaded_file)
            st.write(result)
            st.write("*Please note: The feedback provided is objective and based on analysis algorithms. Artists should consider their creative intent when interpreting and acting on the feedback.*")

    elif selected_feedback == "Color Temperature Analysis":
        st.header("Color Temperature Analysis")
        st.write("Color temperature analysis evaluates the overall warmth or coolness of your artwork's color palette. It helps determine the emotional tone and mood conveyed by the colors used in the composition.")
        uploaded_file = st.file_uploader("Choose an image for analysis", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            result = evaluate_color_temperature(uploaded_file)
            st.write(result)
            st.write("*Please note: The feedback provided is objective and based on analysis algorithms. Artists should consider their creative intent when interpreting and acting on the feedback.*")

    elif selected_feedback == "Color Palette Extractor":
        st.header("Color Palette Extractor")
        st.write("Extract dominant colors from your uploaded image.")

        # File uploader for image
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        # Slider for selecting number of colors in palette
        num_colors = st.slider("Number of Colors", min_value=3, max_value=10, value=5)

        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Extract color palette
            colors = extract_color_palette(uploaded_file, num_colors)

            if colors:
                # Display color palette
                st.write("Dominant Colors:")
                display_color_palette(colors)
            else:
                st.write("Error extracting colors from the image.")



if __name__ == "__main__":
    main()