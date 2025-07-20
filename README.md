# 🌱 Plant Species Classifier

A web application that uses deep learning to identify plant species from uploaded images.

![Plant Species Classifier](https://img.shields.io/badge/AI-Plant%20Classification-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)

## 🚀 Features

- **Image Upload**: Easily upload plant images for identification
- **Real-time Classification**: Get instant predictions of plant species
- **Confidence Scores**: View confidence levels for predictions
- **Top 5 Predictions**: See alternative possible species matches
- **Visual Analysis**: Interactive charts showing prediction confidence
- **Prediction History**: Track and review previous identifications

## 📋 Requirements

- Python 3.8+
- Streamlit
- TensorFlow
- Pillow
- NumPy
- Plotly
- Pandas

## 🛠️ Installation

1. Clone this repository
   ```
   git clone <repository-url>
   cd TreeSpeciesFrontend
   ```

2. Create a virtual environment
   ```
   python -m venv venvTrees
   ```

3. Activate the virtual environment
   ```
   # On Windows
   venvTrees\Scripts\activate
   
   # On macOS/Linux
   source venvTrees/bin/activate
   ```

4. Install dependencies
   ```
   pip install -r requirements.txt
   ```

5. Run the application
   ```
   streamlit run app.py
   ```

## 🌐 Deployment

This application can be deployed on:

- [Streamlit Community Cloud](https://share.streamlit.io/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Render](https://render.com/)
- [Railway](https://railway.app/)

## 🧠 Model Architecture

The application uses a VGG16-based transfer learning model trained on a custom plant species dataset.

## 📸 How It Works

1. Upload a clear image of a plant
2. The model preprocesses the image (resizing, normalization)
3. The deep learning model analyzes the image
4. Results display the predicted species and confidence level
5. Additional visualizations show alternative matches

## 📊 Project Structure

```
TreeSpeciesFrontend/
├── app.py                    # Main Streamlit application
├── plant_classifier_model.h5 # Trained TensorFlow model
├── class_names.pkl           # Class names mapping
├── idx_to_class.pkl          # Index to class mapping
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🔮 Future Improvements

- Mobile optimization
- Multi-language support
- Plant care recommendations
- Integration with plant databases
- Offline mode support

## 📄 License

[MIT License](LICENSE)