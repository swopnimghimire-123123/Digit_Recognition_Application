#  Real-Time Digit Recognition Dashboard

An interactive AI dashboard that recognizes handwritten digits in real time using a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.

Users can draw a digit directly in the browser and the model instantly predicts the number while showing confidence scores and probability distribution.

---

#  Features

* Real-time handwritten digit recognition
* Interactive drawing canvas
* Live AI predictions
* Confidence score visualization
* Probability distribution chart
* Top-3 predicted digits
* Download processed 28×28 input image
* Clean Streamlit dashboard interface

---

#  Dataset

The model is trained using the **MNIST handwritten digits dataset**.

Dataset details:

| Split      | Samples         |
| ---------- | --------------- |
| Training   | 60,000          |
| Testing    | 10,000          |
| Image Size | 28×28 grayscale |

Digits range from **0–9**.

---

#  Model Architecture

The CNN used in this project is deeper than a basic MNIST model and includes **Batch Normalization and Dropout** to improve generalization.

```
Input (28x28x1)

Conv2D (32 filters, 3x3, ReLU)
BatchNormalization

Conv2D (32 filters, 3x3, ReLU)
MaxPooling2D
Dropout (0.25)

Conv2D (64 filters, 3x3, ReLU)
BatchNormalization

Conv2D (64 filters, 3x3, ReLU)
MaxPooling2D
Dropout (0.25)

Flatten

Dense (128, ReLU)
BatchNormalization
Dropout (0.5)

Dense (10, Softmax)
```

---

#  Training Pipeline

The model is trained using **data augmentation and adaptive learning rate control** to improve performance.

### Data Preprocessing

* Normalization: pixel values scaled to **0–1**
* Reshape: **28×28×1** for CNN input

### Data Augmentation

Applied using `ImageDataGenerator`:

* Rotation range: **10°**
* Zoom range: **10%**
* Width shift: **10%**
* Height shift: **10%**

This helps the model generalize better to different handwriting styles.

---

#  Training Parameters

| Parameter       | Value                           |
| --------------- | ------------------------------- |
| Optimizer       | Adam                            |
| Learning Rate   | 0.001                           |
| Loss Function   | Sparse Categorical Crossentropy |
| Batch Size      | 128                             |
| Epochs          | 20                              |
| Training Method | Augmented data generator        |
| Validation Data | MNIST Test Set                  |

---

#  Training Callbacks

Two callbacks are used to stabilize training.

### Early Stopping

Stops training when validation loss stops improving.

| Parameter            | Value           |
| -------------------- | --------------- |
| Monitor              | Validation Loss |
| Patience             | 5               |
| Restore Best Weights | Yes             |

### Reduce Learning Rate on Plateau

Reduces learning rate if validation loss stagnates.

| Parameter | Value           |
| --------- | --------------- |
| Monitor   | Validation Loss |
| Patience  | 3               |
| Factor    | 0.5             |

---

#  Model Performance

Final training results:

| Metric              | Value   |
| ------------------- | ------- |
| Training Accuracy   | ~99.15% |
| Validation Accuracy | ~99.59% |
| Test Accuracy       | ~99.61% |
| Test Loss           | ~0.0116 |

The model achieves **very high accuracy due to augmentation and regularization techniques**.

---

#  Installation

Clone the repository

```
git clone https://github.com/yourusername/digit-recognition-dashboard.git
cd digit-recognition-dashboard
```

Install dependencies

```
pip install -r requirements.txt
```

---

#  Running the Dashboard

Start the Streamlit app:

```
streamlit run app.py
```

The dashboard will open in your browser.

---

#  How to Use

1. Draw a digit (0–9) on the canvas
2. The AI predicts the number instantly
3. View:

   * predicted digit
   * confidence score
   * probability distribution
   * top-3 predictions
4. Download the processed **28×28 input image**

---

#  Project Structure

```
digit-recognition-dashboard
│
├── app.py
├── training_model.py
│
├── model
│   └── digit_model.h5
│
├── requirements.txt
└── README.md
```

---

#  Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* OpenCV
* NumPy
* Matplotlib

---

#  Author

Swopnim Ghimire
Data Science Student
Kathmandu University
