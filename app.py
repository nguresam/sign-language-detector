import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

model = tf.keras.models.load_model('smnist.h5')

@app.route('/')
def index():
    github_link = "https://github.com/your-username/your-repo-name"
    return render_template('index.html', github_link=github_link)
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    mphands = mp.solutions.hands
    hands = mphands.Hands()

    while True:
        _, frame = cap.read()

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks

        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = frame.shape[1]
                y_min = frame.shape[0]
                for lm in handLMs.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.shape[0] > 0 and roi.shape[1] > 0:  # Check if the region of interest is valid
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_resized = cv2.resize(roi_gray, (28, 28))
                    roi_normalized = roi_resized / 255.0
                    roi_final = np.expand_dims(roi_normalized, axis=-1)
                    roi_final = np.expand_dims(roi_final, axis=0)

                    prediction = model.predict(roi_final)
                    pred_probs = prediction[0]
                    max_prob_idx = np.argmax(pred_probs)
                    predicted_letter = chr(ord('A') + max_prob_idx)  # Convert index to character
                    confidence = pred_probs[max_prob_idx]

                    text = f"Predicted: {predicted_letter} (Confidence: {confidence:.2f})"
                    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        #cv2.imshow("Frame", frame)
        #cap.release()
        #cv2.destroyAllWindows()
                # ... (Rest of your code to process hand gestures)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Add a GitHub link in the sidebar
github_link = 'https://github.com/nguresam/sign-language-detector'
app.jinja_env.globals.update(github_link=github_link)

if __name__ == '__main__':

    app.run(debug=True)
    cv2.imshow("Frame", frame)
    cv2.destroyAllWindows()
    cap.release()