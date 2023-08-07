import streamlit as st
import cv2
import numpy as np
from grad_cam import gradcam
from streamlit_drawable_canvas import st_canvas
from PIL import Image


st.title("Digit Recognition with GradCAM")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
st.sidebar.title("Draw a Digit")

if uploaded_file is not None:
    try:
        image_data = uploaded_file.read()
        img_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), 1)
        sup_img, heatmap, recognized_digit = gradcam(img_array)
    except:
        print("Please check your file and try again!")
        st.write("## Please check your file and Try Again !")


with st.sidebar:

    # def draw_on_canvas(event, x, y, flags, param):
    #     global drawn_img
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         cv2.circle(drawn_img, (x, y), 10, (255, 255, 255), -1)
    #         canvas.image(drawn_img, channels="BGR")

    # cv2.namedWindow('Canvas')
    # cv2.setMouseCallback('Canvas', draw_on_canvas)
    
    canvas = st_canvas(
        fill_color="#000000",
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=448,
        height=448,
        drawing_mode="freedraw",
        key="canvas",
    )
        

if st.button("Recognize"):
    if canvas.image_data is not None and uploaded_file is None:
        img_array = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        sup_img, heatmap, recognized_digit = gradcam(img_array)

    # col1, col2 = st.beta_columns(2)
    if img_array.sum() != 0:
        # read_sup_img = Image.open("data/gradcam_out/img_class_{}.png".format(recognized_digit))
        # read_sup_img = np.array(read_sup_img)
        st.image([img_array, sup_img], caption=['Drawn Image', 'GradCAM'], clamp=True, width=338)

        st.subheader("Recognized Digit:")
        digit = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">{}</p>'.format(recognized_digit)
        st.markdown(digit, unsafe_allow_html=True)

    else:
        print("The Canvas/Image is empty, draw a digit to continue !")
        st.write("## The Canvas is empty, draw a digit to continue !")