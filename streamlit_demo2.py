import pandas as pd
import streamlit as st
import moviepy.editor as moviepy
from PIL import Image
import cv2
from engine import *
from model import SASNet
from models import build_model


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def object_detection_video(ort_session, device, transform):
    st.title("Crowd Detection for Videos")
    st.subheader("""
    
    This is blah blah blah
    
    """)
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video is not None:

        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(vid, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")

        cap = cv2.VideoCapture(vid)
        w, h = 1024, 768
        print(w, h)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 5.0, (w, h))
        data_flow = []
        while True:
            _, image = cap.read()
            if _:

                if len(data_flow) % 10 == 0:

                    image = cv2.resize(image, (1024, 768))
                    print(image.shape)

                    # pre-processing
                    img = transform(image)
                    img = img.unsqueeze(0)

                    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
                    ort_outs = ort_session.run(None, ort_inputs)
                    outputs = np.array(ort_outs[0])

                    predict_cnt = np.sum(outputs) / 1000

                    outputs = np.uint8(outputs.squeeze() * 255 / outputs.max())
                    outputs = cv2.cvtColor(outputs, cv2.COLOR_GRAY2BGR)
                    outputs = cv2.addWeighted(image, 0.5, outputs, 0.5, 0)

                    out.write(outputs)

                data_flow.append(predict_cnt)
            else:
                break

        new = pd.DataFrame()
        new["Count"] = data_flow
        st.line_chart(new)
        cap.release()
        cv2.destroyAllWindows()


def object_detection_image(ort_session, device, transform):
    st.title('Crowd Detection for Images')
    st.subheader("""
    This is blah blah blah
    """)
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        img_raw = Image.open(file).convert('RGB').resize((1024, 768))

        st.image(img_raw, caption="Uploaded Image")
        my_bar = st.progress(0)

        samples = transform(img_raw).unsqueeze(0)

        print(samples.shape)

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(samples)}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = np.array(ort_outs[0])

        predict_cnt = np.sum(outputs) / 1000

        img_to_draw = np.array(outputs.squeeze())

        img_to_draw = np.uint8(img_to_draw * 255 / img_to_draw.max())
        img_raw = np.array(img_raw)
        img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_GRAY2BGR)

        print(img_to_draw.shape)
        print(img_raw.shape)

        img_to_draw = cv2.addWeighted(img_raw, 0.5, img_to_draw, 0.5, 0)

        st.image(img_to_draw, caption='Processed Image.')
        st.write(predict_cnt)
        my_bar.progress(100)


def main():
    new_title = '<p style="font-size: 42px;">Crowd Detection!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    args = {
        "model_path": "./models/SHHA.pth",
        "block_size": 32,
        "log_para": 1000,
        "batch_size": 4
    }

    device = torch.device('cuda')

    import onnxruntime
    ort_session = onnxruntime.InferenceSession("models/SHHA.onnx",
                                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    read_me = st.markdown("""
    
    Blah Blah Blah
    
    """)
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox("MODE", ("About", "Object Detection(Image)", "Object Detection(Video)"))

    if choice == "Object Detection(Image)":
        read_me_0.empty()
        read_me.empty()
        object_detection_image(ort_session, device, transform)
    elif choice == "Object Detection(Video)":
        read_me_0.empty()
        read_me.empty()
        object_detection_video(ort_session, device, transform)
        try:
            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4', 'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video")
        except OSError:
            ''

    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
