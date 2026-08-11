from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_FOLDER_PATH = Path(__file__).resolve().parent / "my_final_emotion_model"
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print("Loading AI model from local disk...")
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_FOLDER_PATH))
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_FOLDER_PATH))
        _model.eval()

    return _tokenizer, _model


def predict(text, threshold=0.3):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    results = [
        {"emotion": EMOTION_LABELS[i], "score": float(prob)}
        for i, prob in enumerate(probs)
        if prob >= threshold
    ]
    results.sort(key=lambda item: item["score"], reverse=True)

    if not results:
        top_index = int(probs.argmax())
        return {
            "text": text,
            "threshold": threshold,
            "top_emotion": EMOTION_LABELS[top_index],
            "top_score": float(probs[top_index]),
            "results": [],
            "all_scores": [
                {"emotion": EMOTION_LABELS[i], "score": float(prob)}
                for i, prob in enumerate(probs)
            ],
        }

    return {
        "text": text,
        "threshold": threshold,
        "top_emotion": results[0]["emotion"],
        "top_score": results[0]["score"],
        "results": results,
        "all_scores": [
            {"emotion": EMOTION_LABELS[i], "score": float(prob)}
            for i, prob in enumerate(probs)
        ],
    }


def print_prediction(prediction):
    print(f"\nInput: '{prediction['text']}'")

    if not prediction["results"]:
        print(f"- No emotions crossed the threshold of {prediction['threshold']:.2f}")
        print(
            f"- Strongest emotion: {prediction['top_emotion'].capitalize()} "
            f"({prediction['top_score'] * 100:.1f}%)"
        )
        return

    for item in prediction["results"]:
        print(f"- {item['emotion'].capitalize()}: {item['score'] * 100:.1f}%")


if __name__ == "__main__":
    load_model()
    print("\nModel loaded successfully! Type 'quit' to exit.")

    while True:
        user_input = input("\nEnter a sentence: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            print("Please enter some text.")
            continue

        print_prediction(predict(user_input))
