import streamlit as st
import moviepy.editor as moviepy
from PIL import Image
import cv2
from engine import *
from models import build_model


def object_detection_video(model, device, transform):

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
        _, image = cap.read()
        h, w = image.shape[:2]
        w = w // 128 * 64
        h = h // 128 * 64
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))

        while True:
            _, image = cap.read()
            if _:

                # round the size
                width, height, _ = image.shape
                print(width, height)
                new_width = width // 128 * 64
                new_height = height // 128 * 64
                image = cv2.resize(image, (new_height, new_width))
                # pre-processing
                img = transform(image)

                print(img.shape)

                samples = torch.Tensor(img).unsqueeze(0)
                samples = samples.to(device)
                # run inference
                outputs = model(samples)
                outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

                outputs_points = outputs['pred_points'][0]

                threshold = 0.5
                # filter the predictions
                points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
                predict_cnt = int((outputs_scores > threshold).sum())

                print(predict_cnt)
                # draw the predictions
                size = 2
                img_to_draw = np.array(image)
                for p in points:
                    img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

                out.write(img_to_draw)
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


def object_detection_image(model, device, transform):
    st.title('Crowd Detection for Images')
    st.subheader("""
    This is blah blah blah
    """)
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:

        img_raw = Image.open(file).convert('RGB')

        st.image(img_raw, caption="Uploaded Image")
        my_bar = st.progress(0)

        # round the size
        width, height = img_raw.size
        print(width, height)
        new_width = width // 128 * 64
        new_height = height // 128 * 64
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # pre-processing
        img = transform(img_raw)

        print(img.shape)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        print(predict_cnt)
        # draw the predictions
        size = 2
        img_to_draw = np.array(img_raw)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        st.image(img_to_draw, caption='Processed Image.')
        my_bar.progress(100)


def main():
    new_title = '<p style="font-size: 42px;">Crowd Detection!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    args = {
        "backbone": "vgg16_bn",
        "row": 2,
        "line": 2,
        "output_dir": "./logs/",
        "weight_path": "./weights/SHTechA.pth",
        "gpu_id": 0
    }

    device = torch.device('cuda')

    # get the P2PNet
    model = build_model(args)

    # move to GPU
    model.to(device)

    # load trained model
    if args["weight_path"] is not None:
        checkpoint = torch.load(args["weight_path"], map_location=device)
        model.load_state_dict(checkpoint['model'])

    # convert to eval mode
    model.eval()

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
        object_detection_image(model, device, transform)
    elif choice == "Object Detection(Video)":
        read_me_0.empty()
        read_me.empty()
        object_detection_video(model, device, transform)
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
