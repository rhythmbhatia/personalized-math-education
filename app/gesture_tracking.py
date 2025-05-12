import cv2
import mediapipe as mp

def track_hand():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)

    while True:
        success, image = cap.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            print("Hand detected")

        cv2.imshow("Hand Tracker", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# To test
if __name__ == "__main__":
    track_hand()
