import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from videoplayer import ExerciseVideoPlayer
from keras import Sequential
from keras.applications.efficientnet import preprocess_input
from keras.layers import Input, Lambda, Dense
from keras.applications import EfficientNetB7
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# for frame recorder
IMAGE_SIZE = (720, 720)
FRAME_INTERVAL = 10
# for calculating joint angles
UPPER_JOINTS = {
    "LEFT_SHOULDER": (
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP
    ),
    "RIGHT_SHOULDER": (
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP
    ),
    "LEFT_ELBOW": (
        mp_pose.PoseLandmark.LEFT_SHOULDER, 
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST
    ),
    "RIGHT_ELBOW": (
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST
    ),
}
LOWER_JOINTS = {
    "LEFT_HIP": (
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE
    ),
    "RIGHT_HIP": (
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE
    ),
    "LEFT_KNEE": (
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE
    ),
    "RIGHT_KNEE": (
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    )
}
JOINT_GROUPS = [UPPER_JOINTS, LOWER_JOINTS]
# for joint change plot
UPDATE_INTERVAL = 0.1
TOP_THRESHOLD = 140
BOTTOM_THRESHOLD = 135
WINDOW_LENGTH = 50
# for cnn
CNN_IMAGE_SIZE = (600, 600)

def calc_angle(p1, p2, p3):
    """Calculate the angle between three points p1, p2, and p3"""
    # create vectors
    AB = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    BC = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    # calc dot product and magnitudes
    dot_product = np.dot(AB, BC)
    mag_AB = np.linalg.norm(AB)
    mag_BC = np.linalg.norm(BC)
    # calculate angle
    angle_rad = np.arccos(dot_product / (mag_AB * mag_BC))
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calc_joint_angles(landmarks):
    """Calculate the angles of the key joints (e.g. {"LEFT_SHOULDER": 5, ...})"""
    joint_angles = {}
    for joint_group in JOINT_GROUPS:
        for joint, (index1, index2, index3) in joint_group.items():
            p1 = landmarks[index1]
            p2 = landmarks[index2]
            p3 = landmarks[index3]
            angle = calc_angle(p1, p2, p3)
            joint_angles[joint] = angle
    return joint_angles

#     #     # check the actual joint values for thresholds
#     #     if (joint_angles["LEFT_KNEE"] < BOTTOM_THRESHOLD and joint_angles["LEFT_KNEE"] < BOTTOM_THRESHOLD or       # squat bottom
#     #         joint_angles["RIGHT_KNEE"] < BOTTOM_THRESHOLD and joint_angles["RIGHT_KNEE"] < BOTTOM_THRESHOLD) or \
#     #         (joint_angles["LEFT_KNEE"] > TOP_THRESHOLD and joint_angles["LEFT_KNEE"] > TOP_THRESHOLD or            # squat top
#     #          joint_angles["RIGHT_KNEE"] > TOP_THRESHOLD and joint_angles["RIGHT_KNEE"] > TOP_THRESHOLD) or \
#     #         (joint_angles["LEFT_ELBOW"] < BOTTOM_THRESHOLD or joint_angles["RIGHT_ELBOW"] < BOTTOM_THRESHOLD) or \
#     #         (joint_angles["LEFT_ELBOW"] > TOP_THRESHOLD and joint_angles["RIGHT_ELBOW"] > TOP_THRESHOLD):
#     #         print(joint_angles)
#     #         return True
#     elbow_joint.append((joint_angles["LEFT_ELBOW"], joint_angles["RIGHT_ELBOW"]))
#     knee_joint.append((joint_angles["LEFT_KNEE"], joint_angles["RIGHT_KNEE"]))

#     return False

def track_reps(bot_indices, top_indices):
    reps = []
    for i, bot in enumerate(bot_indices):
        # find nearest top before (start of rep)
        prev_tops = top_indices[top_indices < bot]
        if len(prev_tops) == 0:
            continue
        start_top = prev_tops[-1]
        # if same top as the previous bot, skip (part of same rep)
        if len(reps) > 0 and start_top == reps[-1][0]:
            continue
        # find nearest top after (end of rep)
        next_tops = top_indices[top_indices > bot]
        if len(next_tops) == 0:
            continue
        end_top = next_tops[0]
        reps.append((start_top, bot, end_top))
    return reps


def detect_terminal(elbow_angles_smoothed, knee_angles_smoothed):
    """Detect potential terminal positions"""
    # detect peaks and troughs
    elbow_bot_indices, _ = find_peaks(-elbow_angles_smoothed, height=-BOTTOM_THRESHOLD)
    elbow_top_indices, _ = find_peaks(elbow_angles_smoothed, height=TOP_THRESHOLD)
    knee_bot_indices, _ = find_peaks(-knee_angles_smoothed, height=-BOTTOM_THRESHOLD)
    knee_top_indices, _ = find_peaks(knee_angles_smoothed,  height=TOP_THRESHOLD)
    elbow_reps = track_reps(elbow_bot_indices, elbow_top_indices)
    knee_reps = track_reps(knee_bot_indices, knee_top_indices)
    if len(elbow_reps) > len(knee_reps):
        dominant_joint = "elbow"
        reps = elbow_reps
    else:
        dominant_joint = "knee"
        reps = knee_reps
    return dominant_joint, reps


def classify_exercise(frames, reps):
    """Classify the exercise based on top and bottom positions"""
    model = Sequential([
    Input((224, 224, 3)),
    Lambda(lambda x: preprocess_input(x)),
    EfficientNetB7(include_top=False, weights='imagenet', pooling='max'),
    Dense(6, activation='softmax')
    ])
    model.load_weights("model.weights.h5")
    
    classes = ["pushup", "squat"]
    predictions = []

    for frame_index in reps[0]:
        frame = frames[frame_index]
        frame = cv2.resize(frame, (224,224))
        frame = np.expand_dims(frame, axis=0)
        prediction_array = model.predict(frame, verbose=0)
        prediction = classes[np.argmax(prediction_array) // 2]
        predictions.append(prediction)
    predicted_exercise = max(set(predictions), key=predictions.count)
    return predicted_exercise
        

# TODO
def generate_feedback(rep):
    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default=0, help="Path of video file (omit for webcam)")
    parser.add_argument("--draw", action="store_true", help="Enable live drawing of joint angles (lower FPS)")
    args = parser.parse_args()

    if args.file == "demo1":
        file = "squat.mp4"
    elif args.file == "demo2":
        file = "pushup.mp4"
    else:
        file = args.file
    cap = cv2.VideoCapture(file)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    print("Draw mode is", "enabled" if args.draw else "disabled")

    # set up plot
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.window.wm_geometry("+1000+100")
    elbows_line, = ax.plot([], [], label="Elbows", color="blue")
    knees_line, = ax.plot([], [], label="Knees", color="red")
    ax.axhline(y=TOP_THRESHOLD, color="black", linestyle="--", label="Top Threshold")
    ax.axhline(y=BOTTOM_THRESHOLD, color="black", linestyle="--", label="Bottom Threshold")
    ax.set_title("Smoothed Joint Angles")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Average Angle (degrees)")
    ax.legend(loc = "lower left")
    
    frames = []
    knee_angles = []
    elbow_angles = []
    knee_angles_smoothed = []
    elbow_angles_smoothed = []
    last_graph_update = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as pose:
        frame_num = 0
        # While camera is open
        while cap.isOpened():
          success, image = cap.read()
          # also when video ends if video input
          if not success:
            print("Ignoring empty camera frame.")
            # draw final angle plot
            break

          # Process each frame
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = pose.process(image)

           # process landmarker results
          if results.pose_landmarks:
              landmarks = results.pose_landmarks.landmark
              # record the joint angles for every interval
              if frame_num % FRAME_INTERVAL == 0:
                  joint_angles = calc_joint_angles(landmarks)
                  elbow_angles.append((joint_angles["LEFT_ELBOW"] + joint_angles["RIGHT_ELBOW"]) / 2)
                  knee_angles.append((joint_angles["LEFT_KNEE"] + joint_angles["RIGHT_KNEE"]) / 2)

              # Draw the pose annotation on the image.
              image.flags.writeable = True
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
              mp_drawing.draw_landmarks(
                  image,
                  results.pose_landmarks,
                  mp_pose.POSE_CONNECTIONS,
                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
              # show the image
              image = cv2.resize(image, IMAGE_SIZE)
              frames.append(image)
              cv2.imshow("MediaPipe Pose", image)
              
              # draw the smoothed joint angle plot
              if args.draw:
                  curr_time = time.time()
                  if curr_time - last_graph_update >= UPDATE_INTERVAL and len(knee_angles) > WINDOW_LENGTH:
                      # smooth data
                      elbow_angles_smoothed = savgol_filter(elbow_angles, window_length=WINDOW_LENGTH, polyorder=3)
                      knee_angles_smoothed = savgol_filter(knee_angles, window_length=WINDOW_LENGTH, polyorder=3)
                      # update smoothed plot data
                      elbows_line.set_ydata(elbow_angles_smoothed)
                      elbows_line.set_xdata(range(len(elbow_angles_smoothed)))
                      knees_line.set_ydata(knee_angles_smoothed)
                      knees_line.set_xdata(range(len(knee_angles_smoothed)))
                      # top_markers.set_data(top_indices, [knee_angles_smoothed[i] for i in top_indices])
                      # bottom_markers.set_data(bottom_indices, [knee_angles_smoothed[i] for i in bottom_indices])
                      # readjust and draw
                      ax.relim()
                      ax.autoscale_view()
                      fig.canvas.draw()
                      fig.canvas.flush_events()
                      
                      # update last update time
                      last_graph_update = curr_time
          key = cv2.waitKey(5) & 0xFF
          # esc to cancel
          if key == 27:
            break
          # space to pause
          elif key == 32:
            while True:
                if cv2.waitKey(5) & 0xFF == 32:
                    break
    cap.release()
    elbow_angles_smoothed = savgol_filter(elbow_angles, window_length=WINDOW_LENGTH, polyorder=3)
    knee_angles_smoothed = savgol_filter(knee_angles, window_length=WINDOW_LENGTH, polyorder=3)
    # update smoothed plot data
    elbows_line.set_ydata(elbow_angles_smoothed)
    elbows_line.set_xdata(range(len(elbow_angles_smoothed)))
    knees_line.set_ydata(knee_angles_smoothed)
    knees_line.set_xdata(range(len(knee_angles_smoothed)))
    # detect and plot terminal positions
    dominant_joint, reps = detect_terminal(elbow_angles_smoothed, knee_angles_smoothed)
    for (start_top, bot, end_top) in reps:
        if dominant_joint == "elbow":
            plt.scatter(start_top, elbow_angles_smoothed[start_top], color="green", label="Start of Rep")
            plt.scatter(bot, elbow_angles_smoothed[bot], color="yellow", label="Bottom of Rep")
            plt.scatter(end_top, elbow_angles_smoothed[end_top], color="purple", label="End of Rep")
        else:
            plt.scatter(start_top, knee_angles_smoothed[start_top], color="green", label="Start of Rep")
            plt.scatter(bot, knee_angles_smoothed[bot], color="yellow", label="Bottom of Rep")
            plt.scatter(end_top, knee_angles_smoothed[end_top], color="purple", label="End of Rep")
    # readjust and draw
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    # pass terminal frames into cnn to classify exercise
    exercise = classify_exercise(frames, reps)
    print("Classified Exercise:", exercise)
    # TODO: generate feedback for each rep
    for rep in reps:
        rep = generate_feedback(rep)
    # open a plot for each rep that can be played back and has feedback
    while True:
        if cv2.waitKey(5) & 0xFF == 32:
            # plt.close("all")
            player = ExerciseVideoPlayer(frames, reps)
            player.show()
            while True:
                plt.pause(0.001)


if __name__ == "__main__":
    main()