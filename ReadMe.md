# Emotion Detection Web App

A Flask-based web app that predicts emotions from text using a fine-tuned transformer model.

## Included in this repository
- App server: `app.py`
- Inference logic: `finalModel.py`
- UI files: `templates/index.html`, `static/style.css`
- Python dependencies: `requirements.txt`

## Not included in this repository
Large files are intentionally ignored to keep the repository lightweight and GitHub-compatible:
- Virtual environment (`.venv/`)
- Raw/processed datasets (`*.csv`)
- Zipped archives (`*.zip`)
- Model weights folder (`my_final_emotion_model/`)
- Documents and personal media files

## Run locally
1. Create a virtual environment and activate it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained model folder at:
   - `my_final_emotion_model/`
4. Start the app:
   ```bash
   python app.py
   ```

Then open the local Flask URL shown in your terminal.
