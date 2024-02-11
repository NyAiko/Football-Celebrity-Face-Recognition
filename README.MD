# Football Face recognition

**Model:** Non-Deep Learning football star face recognition [application](https://football-celebrity-face-recognition.streamlit.app/)

## Libary Used

1. **Python** 
2. **Pandas**  
3. **Matplotlib**  
4. **Scikit-Learn**  
5. **Scikit-image**
6. **Streamlit**

## Visual Overview of Data

The data are faces collected from publicly available sources and free of rights from Google Images.
It has 5 classes: Ronaldo, Neymar, Modric, Messi, Mbappe, Ibrahimovic, Haaland, Benzema and Unknown. 
After removing the duplicated images, we have 1382 images in total.

## Model training:
0. **Image Processing**: Resize image into (80,80), put them into grayscale and then perform histogram equalization on each images
1. **Features extraction**: I used Histogram of Oriented Gradients(HOG) to convert images as feature vectors
2. **Model training**: Trained a Support Vector Machine(SVM) algorithms as a classifier
3. **Tune Hyperparameters**: Tuned the C parameters of the SVM and found C=2.0 the optimal
4. **Deployment**: Web app and its deployments made simple with 'Streamlit' available [here](https://football-celebrity-face-recognition.streamlit.app/)
## Model Performance
The trained model has 86% of accuracy on the test set.

## How to Open the Notebook

To explore how the model works, follow these steps:

1. **Install Jupyter Notebook:** If not already installed, run `pip install notebook` in your terminal or command prompt.

2. **Download the Notebook:** Obtain the predictive model notebook from the designated repository or source.

3. **Navigate to the Notebook's Directory:** Open your terminal or command prompt, use `cd` to navigate to the directory where the notebook is located.

4. **Launch Jupyter Notebook:** Type `jupyter notebook` in the terminal and press Enter. This will open a new tab in your web browser.

5. **Access the Notebook:** In the Jupyter Notebook interface, navigate to the directory where the notebook is located and click on the notebook file (with a `.ipynb` extension).

6. **Run the Notebook Cells:** Once the notebook is open, run each cell sequentially to observe the model's functionality and visualize the results.


