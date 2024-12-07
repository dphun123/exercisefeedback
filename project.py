import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# for frame recorder
IMAGE_SIZE = (720, 720)
FRAME_INTERVAL = 10
# for calculating joint angles
UPPER_JOINTS = {
    'LEFT_SHOULDER': (
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP
    ),
    'RIGHT_SHOULDER': (
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP
    ),
    'LEFT_ELBOW': (
        mp_pose.PoseLandmark.LEFT_SHOULDER, 
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST
    ),
    'RIGHT_ELBOW': (
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST
    ),
}
LOWER_JOINTS = {
    'LEFT_HIP': (
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE
    ),
    'RIGHT_HIP': (
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE
    ),
    'LEFT_KNEE': (
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE
    ),
    'RIGHT_KNEE': (
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    )
}
JOINT_GROUPS = [UPPER_JOINTS, LOWER_JOINTS]
# for joint change plot
UPDATE_INTERVAL = 0.1
TOP_THRESHOLD = 140
BOTTOM_THRESHOLD = 110
WINDOW_LENGTH = 50

hip_angles = []
elbow_angles = []
top_indices = []
bottom_indices = []

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
#     #     if (joint_angles['LEFT_KNEE'] < BOTTOM_THRESHOLD and joint_angles['LEFT_HIP'] < BOTTOM_THRESHOLD or       # squat bottom
#     #         joint_angles['RIGHT_KNEE'] < BOTTOM_THRESHOLD and joint_angles['RIGHT_HIP'] < BOTTOM_THRESHOLD) or \
#     #         (joint_angles['LEFT_KNEE'] > TOP_THRESHOLD and joint_angles['LEFT_HIP'] > TOP_THRESHOLD or            # squat top
#     #          joint_angles['RIGHT_KNEE'] > TOP_THRESHOLD and joint_angles['RIGHT_HIP'] > TOP_THRESHOLD) or \
#     #         (joint_angles['LEFT_ELBOW'] < BOTTOM_THRESHOLD or joint_angles['RIGHT_ELBOW'] < BOTTOM_THRESHOLD) or \
#     #         (joint_angles['LEFT_ELBOW'] > TOP_THRESHOLD and joint_angles['RIGHT_ELBOW'] > TOP_THRESHOLD):
#     #         print(joint_angles)
#     #         return True
#     elbow_joint.append((joint_angles['LEFT_ELBOW'], joint_angles['RIGHT_ELBOW']))
#     knee_joint.append((joint_angles['LEFT_KNEE'], joint_angles['RIGHT_KNEE']))

#     return False

# def detect_terminal(landmarks):
#     """Detect potential terminal positions"""
#     global hip_angles, elbow_angles, top_indices, bottom_indices
#     # detect peaks and troughs
#     top_indices, _ = find_peaks(hip_angles_smoothed, distance=20)
#     bottom_indices, _ = find_peaks(-hip_angles_smoothed, distance=20)


def main():
    if len(sys.argv) == 1:
        cap = cv2.VideoCapture(0)
    elif len(sys.argv) == 2:
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        print("usage: python project.py [file]")

    # set up plot
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.window.wm_geometry("+1000+100")
    elbows_line, = ax.plot([], [], label="Elbows", color='blue')
    hips_line, = ax.plot([], [], label="Hips", color='red')
    ax.axhline(y=TOP_THRESHOLD, color='black', linestyle='--', label='Top Threshold')
    ax.axhline(y=BOTTOM_THRESHOLD, color='black', linestyle='--', label='Bottom Threshold')
    ax.set_title('Smoothed Joint Angles')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Average Angle (degrees)')
    ax.legend().set_loc('lower left')
    
    last_graph_update = 0
    
    with mp_pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as pose:
        # While camera is open
        frame_num = 0
        while cap.isOpened():
          success, image = cap.read()
          # also when video ends if video input
          if not success:
            print("Ignoring empty camera frame.")
            # TODO: detect terminal positions, pass them into cnn to determine exercise
            # generate feedback for each rep
            # open a plot for each rep that can be played back and has feedback
            time.sleep(10)
            return

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
                  elbow_angles.append((joint_angles['LEFT_ELBOW'] + joint_angles['RIGHT_ELBOW']) / 2)
                  hip_angles.append((joint_angles['LEFT_HIP'] + joint_angles['RIGHT_HIP']) / 2)

              # Draw the pose annotation on the image.
              image.flags.writeable = True
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
              mp_drawing.draw_landmarks(
                  image,
                  results.pose_landmarks,
                  mp_pose.POSE_CONNECTIONS,
                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
              # Flip the image horizontally for a selfie-view display.
              image = cv2.resize(image, IMAGE_SIZE)
              cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
              
              curr_time = time.time()
              # draw the smoothed joint angle plot
              if curr_time - last_graph_update >= UPDATE_INTERVAL and len(hip_angles) > WINDOW_LENGTH:
                  # smooth data
                  elbow_angles_smoothed = savgol_filter(elbow_angles, window_length=WINDOW_LENGTH, polyorder=3)
                  knee_angles_smoothed = savgol_filter(hip_angles, window_length=WINDOW_LENGTH, polyorder=3)
                  # update smoothed plot data
                  elbows_line.set_ydata(elbow_angles_smoothed)
                  elbows_line.set_xdata(range(len(elbow_angles_smoothed)))
                  hips_line.set_ydata(knee_angles_smoothed)
                  hips_line.set_xdata(range(len(knee_angles_smoothed)))
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

if __name__ == "__main__":
    main()