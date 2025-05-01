from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model, tokenizer, and label encoder
model = load_model("data/bodo_bot_model.h5")
with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("data/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("data/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare responses dictionary
responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}


def preprocess_input(text):
    sequence = tokenizer.texts_to_sequences([text])  # No lower(), no strip() if not in training
    padded = pad_sequences(sequence, maxlen=model.input_shape[1])
    return padded


# Define a function to predict the intent and return the response
def predict_intent(text):
    padded_input = preprocess_input(text)
    prediction = model.predict(padded_input)
    predicted_class = np.argmax(prediction, axis=1)
    intent = le.inverse_transform(predicted_class)
    response = np.random.choice(responses.get(intent[0], ["निमाहा हो, आं बेखौ बुजियाखै।"]))
    print("User message:", text)
    print("Tokenized:", tokenizer.texts_to_sequences([text]))
    print("Padded:", preprocess_input(text))

    return response


@app.route("/chat", methods=["POST"])
def chat():
    # Get the message from the user
    message = request.json.get("message")
    if not message:
        return jsonify({"error": "No message provided!"}), 400

    # Get the response from the model
    response = predict_intent(message)

    return jsonify({"response": response})


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT Render provides
    app.run(debug=True, host="0.0.0.0", port=port)
    
