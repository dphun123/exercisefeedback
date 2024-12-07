import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

IMAGE_SIZE = (720, 720)
FRAME_INTERVAL = 10
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
SUM_ANGLE_THRESH = 5
TOP_THRESHOLD = 150
BOTTOM_THRESHOLD = 120
# for joint change plot
MAX_POINTS = 200
UPDATE_INTERVAL = 0.5

prev_angles = None
all_angle_changes = []


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

def detect_terminal(landmarks):
    """Detect potential terminal positions"""
    global prev_angles
    joint_angles = calc_joint_angles(landmarks)
    # store prev on first
    if prev_angles is None:
        prev_angles = joint_angles
        return False
    sum_angle_change = 0
    # sum angle change of all joints
    for joint_group in JOINT_GROUPS:
        for joint in joint_group.keys():
            angle_change = abs(prev_angles[joint] - joint_angles[joint])
            sum_angle_change += angle_change
    all_angle_changes.append(sum_angle_change)  
    if len(all_angle_changes) > MAX_POINTS:
        all_angle_changes.pop(0)
    # update prev for next
    prev_angles = joint_angles
    # threshold for potential terminal
    if sum_angle_change <= SUM_ANGLE_THRESH:
        # check the actual joint values for thresholds
        if (joint_angles['LEFT_KNEE'] < BOTTOM_THRESHOLD and joint_angles['LEFT_HIP'] < BOTTOM_THRESHOLD or       # squat bottom
            joint_angles['RIGHT_KNEE'] < BOTTOM_THRESHOLD and joint_angles['RIGHT_HIP'] < BOTTOM_THRESHOLD) or \
            (joint_angles['LEFT_KNEE'] > TOP_THRESHOLD and joint_angles['LEFT_HIP'] > TOP_THRESHOLD or            # squat top
             joint_angles['RIGHT_KNEE'] > TOP_THRESHOLD and joint_angles['RIGHT_HIP'] > TOP_THRESHOLD) or \
            (joint_angles['LEFT_ELBOW'] < BOTTOM_THRESHOLD or joint_angles['RIGHT_ELBOW'] < BOTTOM_THRESHOLD) or \
            (joint_angles['LEFT_ELBOW'] > TOP_THRESHOLD and joint_angles['RIGHT_ELBOW'] > TOP_THRESHOLD):
            print(joint_angles)
            return True

    return False


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
    line, = ax.plot([], [])
    threshold_line, = ax.plot([], [], color='r', linestyle='--')
    ax.set_title('Change in Joint Angles')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Total Change in Joint Angles')
    
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
            #TODO: 
            return

          # Process each frame
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = pose.process(image)

          # process landmarker results
          if results.pose_landmarks:
              landmarks = results.pose_landmarks.landmark
              if frame_num % FRAME_INTERVAL == 0:
                  detect_terminal(landmarks)
              # TODO: set up cnn here that checks for top/bottom positions
              #
              # if detect_terminal(landmarks):
              #     print("TERMINAL POSITION DETECTED!")
                  
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
              # draw the sum angle plot
              if curr_time - last_graph_update >= UPDATE_INTERVAL:
                  # update data
                  line.set_ydata(all_angle_changes)
                  line.set_xdata(range(len(all_angle_changes)))
                  threshold_line.set_ydata([SUM_ANGLE_THRESH] * len(all_angle_changes))
                  threshold_line.set_xdata(range(len(all_angle_changes)))
                  # readjust and draw
                  ax.relim()
                  ax.autoscale_view()
                  # redraw the plot
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