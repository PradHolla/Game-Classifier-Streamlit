# Streamlit Game Classifier [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/pradholla/game-classifier-streamlit)
Streamlit is a Python framework which is used to build Data Science and Machine Learning web apps purely in Python.

This is my first project in Streamlit. This is a basic image classifier application. The games on which the classifier is built on are Assassin's Creed Unity and Hitman. 
The model was trained using my own screenshots of the said games. This is a fun project I decided to do since I was new and familiarizing myself with Streamlit.
 
To try out this this app, clone the repo by running:
```bash
git clone https://github.com/PradHolla/Game-Classifier-Streamlit.git
```
`cd` into the folder and install the necessary libraries by running:
```bash
pip install -r requirements.txt
```
To run the app:
```bash
streamlit run streamlit_app.py
```

After the app is loaded to your browser, simply upload a screenshot of either AC Unity or Hitman and the prediction and confidence will be displayed.
