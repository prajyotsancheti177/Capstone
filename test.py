import cv2
import mediapipe as mp
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import streamlit as st

angle_variances_list = []
actual_angle_list = []


class PoseDetector:
    def __init__(
        self,
        mode=False,
        upBody=False,
        smooth=True,
        modelComplex=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.modelComplex = modelComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.upBody,
            self.smooth,
            self.modelComplex,
            self.detectionCon,
            self.trackCon,
        )

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        self.mpDraw.draw_landmarks(
            img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
        )
        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list

    def calculate_angle(self, lm_list, joint1, joint2, joint3):
        p1 = lm_list[joint1]
        p2 = lm_list[joint2]
        p3 = lm_list[joint3]

        angle = math.degrees(
            math.atan2(p3[2] - p2[2], p3[1] - p2[1])
            - math.atan2(p1[2] - p2[2], p1[1] - p2[1])
        )
        if angle < 0:
            angle += 360

        return angle


def main():
    st.title("Cricket Bowler Pose Analysis")

    # # Local video file path
    # video_path = "compressed2.mp4"

    # cap = cv2.VideoCapture(video_path)

    video_data = st.file_uploader("Upload file", ["mp4", "mov", "avi"])

    temp_file_to_save = "./temp_file_1.mp4"
    # temp_file_result = "./temp_file_2.mp4"

    cap = None

    # func to save BytesIO on a drive
    def write_bytesio_to_file(filename, bytesio):
        """
        Write the contents of the given BytesIO to a file.
        Creates the file or overwrites the file if it does
        not exist yet.
        """
        with open(filename, "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(bytesio.getbuffer())

    if video_data:
        # save uploaded video to disc
        write_bytesio_to_file(temp_file_to_save, video_data)

        # read it with cv2.VideoCapture(),
        # so now we can process it with OpenCV functions
        cap = cv2.VideoCapture(temp_file_to_save)

    if cap is None:
        st.warning("Please upload a video.")
        return

    detector = PoseDetector()
    prev_lm_list = []
    p_time = time.time()
    fps = 0
    paused = False
    frame_index = 0

    # Create a placeholder for the video
    video_placeholder = st.empty()

    while True:
        if not paused:
            ret, img = cap.read()
            if not ret:
                break

            frame_index += 1

            img = detector.find_pose(img)
            curr_lm_list = detector.find_position(img, draw=False)

            if len(prev_lm_list) != 0 and len(curr_lm_list) != 0:
                curr_time = time.time()
                time_difference = curr_time - p_time

                if time_difference > 0:
                    fps = 1 / time_difference
                else:
                    fps = 0

                if len(curr_lm_list) == len(prev_lm_list):
                    actual_angles = [
                        detector.calculate_angle(curr_lm_list, joint1, joint2, joint3)
                        for joint1, joint2, joint3 in [
                            (28, 26, 24),
                            (26, 24, 12),
                            (12, 14, 16),
                        ]
                    ]
                    actual_angle_list.append(actual_angles)

                    if actual_angle_list[-1][0] > 180:
                        cv2.circle(
                            img,
                            (curr_lm_list[26][1], curr_lm_list[26][2]),
                            15,
                            (0, 0, 255),
                            cv2.FILLED,
                        )

                    if actual_angle_list[-1][1] < 120 or actual_angle_list[-1][1] > 240:
                        cv2.circle(
                            img,
                            (curr_lm_list[24][1], curr_lm_list[24][2]),
                            15,
                            (0, 0, 255),
                            cv2.FILLED,
                        )

                    if actual_angle_list[-1][2] > 195:
                        cv2.circle(
                            img,
                            (curr_lm_list[14][1], curr_lm_list[14][2]),
                            15,
                            (0, 0, 255),
                            cv2.FILLED,
                        )

                cv2.putText(
                    img,
                    f"Right Knee Angle: {actual_angle_list[-1][0]:.2f} degrees",
                    (70, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    img,
                    f"Hip Angle: {actual_angle_list[-1][1]:.2f} degrees",
                    (70, 150),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    img,
                    f"Right Elbow Angle: {actual_angle_list[-1][2]:.2f} degrees",
                    (70, 200),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 0, 255),
                    2,
                )

            prev_lm_list = curr_lm_list
            img = cv2.resize(img, (800, 600))

        video_placeholder.image(img, channels="BGR")

        # st.markdown(f"**Frame Index:** {frame_index}")
        # if len(actual_angle_list) > 0:
        #     st.markdown(f"**Right Knee Angle:** {actual_angle_list[-1][0]:.2f} degrees")
        #     st.markdown(f"**Hip Angle:** {actual_angle_list[-1][1]:.2f} degrees")
        #     st.markdown(
        #         f"**Right Elbow Angle:** {actual_angle_list[-1][2]:.2f} degrees"
        #     )

        # key = st.button("Pause/Play")
        # if key:
        #     paused = not paused
        #     if paused:
        #         frame_index -= 1

        # if st.button("Exit"):
        #     break

    cap.release()
    cv2.destroyAllWindows()

    plt.figure(figsize=(8, 6))
    angle1_list = [frame[0] for frame in actual_angle_list]
    # Plotting the original data
    # plt.plot(angle1_list, label="Angle Variances", marker="o")
    # Adding a horizontal line at y=180
    plt.axhline(y=170, color="black", linestyle="--", label="Threshold")
    # Creating a boolean list indicating where the condition is met
    above_threshold = np.array(angle1_list) > 170
    # Plotting the original data below the threshold
    plt.plot(angle1_list, label="Knee Angle", marker="o", color="blue")
    # Plotting the portion of the line above the threshold in red
    plt.plot(
        np.where(above_threshold, angle1_list, np.nan),
        label="Above Threshold",
        marker="o",
        color="red",
    )
    # Filling the area above the line with red color
    plt.fill_between(
        range(len(angle1_list)),
        angle1_list,
        170,
        where=above_threshold,
        color="red",
        alpha=0.3,
    )
    # Adding labels and title
    plt.xlabel("Frame Index")
    plt.ylabel("Knee Angle ")
    plt.title("Knee Angle Over Frames")
    # Displaying legend
    plt.legend()
    # Displaying the plot
    plt.grid(True)
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    angle2_list = [frame[1] for frame in actual_angle_list]
    # Plotting the original data
    # plt.plot(angle2_list, label="Angle Variances", marker="o")
    # Adding a horizontal line at y=120
    plt.axhline(y=120, color="black", linestyle="--", label="Threshold")
    # Creating a boolean list indicating where the condition is met
    below_threshold = np.array(angle2_list) < 120
    plt.axhline(y=240, color="black", linestyle="--", label="Threshold")
    # Creating a boolean list indicating where the condition is met
    above_threshold = np.array(angle2_list) > 240
    # Plotting the original data below the threshold
    plt.plot(angle2_list, label="Hip Angle", marker="o", color="blue")
    # Plotting the portion of the line above the threshold in red
    plt.plot(
        np.where(above_threshold, angle2_list, np.nan),
        label="Above Threshold",
        marker="o",
        color="red",
    )
    # Plotting the portion of the line below the threshold in red
    plt.plot(
        np.where(below_threshold, angle2_list, np.nan),
        label="Below Threshold",
        marker="o",
        color="red",
    )
    # Filling the area above and below the line with red color
    plt.fill_between(
        range(len(angle2_list)),
        angle2_list,
        120,
        where=below_threshold,
        color="red",
        alpha=0.3,
    )
    plt.fill_between(
        range(len(angle2_list)),
        angle2_list,
        240,
        where=above_threshold,
        color="red",
        alpha=0.3,
    )
    # Adding labels and title
    plt.xlabel("Frame Index")
    plt.ylabel("Hip Angle ")
    plt.title("Hip Angle Over Frames")
    # Displaying legend
    plt.legend()
    # Displaying the plot
    plt.grid(True)
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    angle3_list = [frame[2] for frame in actual_angle_list]
    # Plotting the original data
    # plt.plot(angle3_list, label="Angle Variances", marker="o")
    # Adding a horizontal line at y=195
    plt.axhline(y=195, color="black", linestyle="--", label="Threshold")
    # Creating a boolean list indicating where the condition is met
    above_threshold = np.array(angle3_list) > 195
    # Plotting the original data below the threshold
    plt.plot(angle3_list, label="Elbow Angles", marker="o", color="blue")
    # Plotting the portion of the line above the threshold in red
    plt.plot(
        np.where(above_threshold, angle3_list, np.nan),
        label="Above Threshold",
        marker="o",
        color="red",
    )
    # Filling the area below the line with red color
    plt.fill_between(
        range(len(angle3_list)),
        angle3_list,
        195,
        where=above_threshold,
        color="red",
        alpha=0.3,
    )
    # Adding labels and title
    plt.xlabel("Frame Index")
    plt.ylabel("Elbow Angle ")
    plt.title("Elbow Angle Over Frames")
    # Displaying legend
    plt.legend()
    # Displaying the plot
    plt.grid(True)
    st.pyplot(plt)


if __name__ == "__main__":
    main()
