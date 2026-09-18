# 🧠 Facial Emotion Classification Using Deep Learning

<p align="center">

<img src="https://img.shields.io/badge/Deep%20Learning-Facial%20Emotion%20Recognition-blueviolet?style=for-the-badge">
<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow">
<img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras">
<img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-green?style=for-the-badge&logo=opencv">
<img src="https://img.shields.io/badge/Interface-Gradio-yellow?style=for-the-badge">

</p>

<p align="center">

**A Deep Learning-based Facial Emotion Classification System using CNN, VGG16, and ResNet50V2 Transfer Learning**

</p>

<p align="center">

<a href="#-overview">Overview</a> • <a href="#-features">Features</a> • <a href="#-architecture">Architecture</a> • <a href="#-models">Models</a> • <a href="https://github.com/shahidazam2020-oss/Deep-Learning-Model-for-Facial-Emotion-Classification/tree/master/test">Dataset</a> • <a href="#-results">Evaluation</a> • <a href="#-deployment">Deployment</a> • <a href="#-installation">Installation</a>

</p>

---

## 📌 Project Overview

This project implements a **Deep Learning Facial Emotion Classification system** capable of classifying facial expressions into **seven emotion categories**.

The project explores multiple deep learning approaches, beginning with a custom **Convolutional Neural Network (CNN)** and progressing toward **Transfer Learning** using pretrained **VGG16** and **ResNet50V2** architectures.

The final workflow includes:

* 🖼️ Facial image preprocessing
* 🔄 Image augmentation
* ⚖️ Class-weight balancing
* 🧠 Custom CNN architecture
* 🚀 VGG16 transfer learning
* ⚡ ResNet50V2 transfer learning
* 📊 Confusion matrix analysis
* 📋 Classification reports
* 📈 ROC curve analysis
* 💾 Model checkpointing
* 🧪 Model evaluation
* 🌐 Gradio-based image prediction interface

The trained ResNet50V2 model is also prepared for deployment, allowing users to upload an image and obtain a predicted facial emotion.

---

# 🎯 Objectives

The primary objectives of this project are to:

1. Develop a deep learning model for facial emotion classification.
2. Compare different CNN-based architectures.
3. Investigate the effectiveness of transfer learning.
4. Apply image augmentation to improve model generalization.
5. Address class imbalance through class weighting.
6. Evaluate models using multiple classification metrics.
7. Analyze prediction errors using confusion matrices.
8. Examine class-wise ROC performance.
9. Save trained models for later use.
10. Build an interactive prediction interface using Gradio.

---

# 😊 Emotion Classes

The system classifies facial expressions into **7 emotion categories**:

| Class | Emotion      |
| :---: | :----------- |
|   😠  | **Angry**    |
|   🤢  | **Disgust**  |
|   😨  | **Fear**     |
|   😄  | **Happy**    |
|   😐  | **Neutral**  |
|   😢  | **Sad**      |
|   😲  | **Surprise** |

### Label Mapping

```text
0 → Angry
1 → Disgust
2 → Fear
3 → Happy
4 → Neutral
5 → Sad
6 → Surprise
```

---
  
# 🏗️ Project Architecture

```text
                    ┌─────────────────────────┐
                    │     Facial Images       │
                    │      Train / Test       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Image Preprocessing   │
                    │  Resize + Normalization │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Data Augmentation     │
                    │ Rotation / Zoom / Shift │
                    │    Horizontal Flip      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Class Balancing      │
                    │      Class Weights      │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
                  ▼              ▼              ▼
             ┌────────┐    ┌─────────┐    ┌────────────┐
             │  CNN   │    │ VGG16   │    │ ResNet50V2 │
             └────┬───┘    └────┬────┘    └─────┬──────┘
                  │              │              │
                  └──────────────┼──────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Evaluation        │
                    │ Accuracy / Confusion    │
                    │ Matrix / Classification │
                    │ Report / ROC Analysis   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Trained Model (.keras)│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Gradio Interface     │
                    │    Upload Image →       │
                    │    Emotion Prediction   │
                    └─────────────────────────┘

```

---

# 🔬 Deep Learning Models

The notebook experiments with three major approaches.

## 1️⃣ Custom CNN

The project first develops a custom convolutional neural network containing multiple convolutional blocks.

The architecture includes:

* `Conv2D`
* ReLU activation
* Batch Normalization
* Max Pooling
* Dropout
* L2 regularization
* Fully connected layers
* Softmax output layer

The CNN processes facial images at **48 × 48** resolution with a grayscale input configuration.

```text
Input Image
     │
     ▼
Conv2D 32
     │
     ▼
Conv2D 64
     │
     ▼
Batch Normalization
     │
     ▼
Max Pooling
     │
     ▼
Dropout
     │
     ▼
Conv2D 128
     │
     ▼
Conv2D 256
     │
     ▼
Batch Normalization
     │
     ▼
Max Pooling
     │
     ▼
Conv2D 512
     │
     ▼
Classification
```

---

# 🚀 2️⃣ VGG16 Transfer Learning

The second major approach uses **VGG16 pretrained on ImageNet**.

Images are resized to:

```text
224 × 224 × 3
```

The pretrained VGG16 convolutional base is used as a feature extractor, with selected layers made trainable for fine-tuning.

### Custom Classification Head

```text
VGG16
  │
  ▼
Flatten
  │
  ▼
Dense 1024 + ReLU
  │
  ▼
Dropout 0.5
  │
  ▼
Dense 512 + ReLU
  │
  ▼
Dropout 0.5
  │
  ▼
Dense 7 + Softmax
```

The model uses:

* ImageNet pretrained weights
* Fine-tuning
* Adam optimizer
* Categorical cross-entropy
* Dropout regularization
* Class weighting

---

# ⚡ 3️⃣ ResNet50V2 Transfer Learning

The project also implements **ResNet50V2** using ImageNet pretrained weights.

The model is partially fine-tuned by freezing earlier layers and allowing the final portion of the network to train.

### ResNet50V2 Architecture

```text
Input Image
     │
     ▼
ResNet50V2
     │
     ▼
Dropout 0.25
     │
     ▼
Batch Normalization
     │
     ▼
Flatten
     │
     ▼
Dense 64 + ReLU
     │
     ▼
Batch Normalization
     │
     ▼
Dropout 0.5
     │
     ▼
Dense 7 + Softmax
     │
     ▼
Emotion Prediction
```

The notebook saves the trained model as:

```text
ResNet50_Transfer_Learning.keras
```

and also creates:

```text
Resnet_model_version_2.keras
```

---

# 🗂️ Dataset Structure

The repository contains separate **training** and **testing** directories.

```text
📦 Project
│
├── 📁 train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── 📁 test/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── 📓 Deep Learning Model for Facial Emotion Classification.ipynb
└── 📄 README.md
```

The notebook uses directory-based image loading with Keras `ImageDataGenerator`.

---

# 🧹 Data Preprocessing

The project performs several preprocessing operations before model training.

### Image Resizing

For the custom CNN:

```text
48 × 48
```

For transfer-learning models:

```text
224 × 224
```

### Pixel Normalization

Pixel values are rescaled from:

```text
0–255
```

to:

```text
0–1
```

using:

```python
rescale = 1 / 255.
```

---

# 🔄 Data Augmentation

Training images are augmented using transformations such as:

| Augmentation    | Configuration |
| --------------- | ------------- |
| Rotation        | ±10°          |
| Zoom            | 0.2           |
| Width Shift     | 0.1           |
| Height Shift    | 0.1           |
| Horizontal Flip | Enabled       |
| Fill Mode       | Nearest       |

The purpose is to expose the model to variations in facial appearance and reduce dependence on exact training-image configurations.

---

# ⚖️ Class Imbalance Handling

The notebook calculates **balanced class weights** using:

```python
compute_class_weight(
    class_weight='balanced'
)
```

The resulting weights are supplied during model training:

```python
class_weight=class_weights_dict
```

This gives the training process a mechanism to account for differences in the number of images available for each emotion class.

---

# 🧪 Training Strategy

The project uses several training techniques to improve model training and generalization.

### Training Components

```text
✔ Image Augmentation
✔ Class Weighting
✔ Transfer Learning
✔ Fine-Tuning
✔ Dropout
✔ Batch Normalization
✔ L2 Regularization
✔ Early Stopping
✔ Learning Rate Reduction
✔ Model Checkpointing
✔ Training Log Generation
```

---

# 🛡️ Training Callbacks

The transfer-learning experiments use callbacks including:

### 💾 ModelCheckpoint

Saves the best-performing model according to validation loss.

### ⏹️ EarlyStopping

Stops training when monitored validation performance does not improve according to the configured patience.

### 📉 ReduceLROnPlateau

Reduces the learning rate when validation performance stops improving.

### 📋 CSVLogger

Stores training information in a log file.

---

# 📊 Model Evaluation

The project does not rely exclusively on accuracy.

Multiple evaluation techniques are implemented:

### 📈 Accuracy

Measures the proportion of correctly classified samples.

### 📋 Classification Report

The notebook generates class-level:

* Precision
* Recall
* F1-score
* Support

### 🔲 Confusion Matrix

The confusion matrix provides a class-by-class view of:

```text
Actual Emotion
      ↓
Predicted Emotion
```

This helps identify which emotions are frequently confused by the model.

### 📈 ROC Analysis

The project also calculates ROC curves and AUC values for the seven emotion classes.

---

# 🔍 Evaluation Workflow

```text
                    Trained Model
                         │
                         ▼
                  Test Dataset
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
          Accuracy   Confusion   Classification
                      Matrix       Report
              │          │          │
              └──────────┼──────────┘
                         │
                         ▼
                    ROC Analysis
```

---

# 🧠 Model Development Journey

The notebook follows an incremental deep learning workflow:

```text
                    Facial Emotion Dataset
                              │
                              ▼
                     Custom CNN Model
                              │
                              ▼
                      Model Evaluation
                              │
                              ▼
                     Data Augmentation
                              │
                              ▼
                      VGG16 Transfer
                              │
                              ▼
                       VGG Evaluation
                              │
                              ▼
                    ResNet50V2 Transfer
                              │
                              ▼
                  Fine-Tuning + Evaluation
                              │
                              ▼
                    Saved Keras Model
                              │
                              ▼
                    Gradio Deployment
```

---

# 🌐 Deployment

The project includes an interactive **Gradio** prediction workflow.

The saved ResNet50V2 model is loaded and used for inference.

### Prediction Pipeline

```text
          📷 User Image
                │
                ▼
        Image Preprocessing
                │
                ▼
          Resize 224×224
                │
                ▼
        Normalize Pixel Values
                │
                ▼
          ResNet50V2 Model
                │
                ▼
        Softmax Probabilities
                │
                ▼
       Highest Probability Class
                │
                ▼
       😊 Emotion Prediction
```

The deployment code maps the model's numerical output back to the corresponding emotion label.

---

# 🖥️ Gradio Interface

The notebook prepares a Gradio-based interface where a user can upload an image.

```text
┌─────────────────────────────────────────────┐
│          🧠 FACIAL EMOTION AI              │
├─────────────────────────────────────────────┤
│                                             │
│        📷 Upload Facial Image               │
│                                             │
│              [ Upload ]                     │
│                                             │
│                 ↓                           │
│                                             │
│        🤖 Deep Learning Model               │
│                                             │
│                 ↓                           │
│                                             │
│       🎯 Predicted Emotion                  │
│                                             │
└─────────────────────────────────────────────┘
```

---

# 🛠️ Technology Stack

| Technology      | Purpose                       |
| --------------- | ----------------------------- |
| 🐍 Python       | Programming Language          |
| 🧠 TensorFlow   | Deep Learning Framework       |
| 🔥 Keras        | Neural Network Development    |
| 👁️ OpenCV      | Computer Vision               |
| 🖼️ PIL         | Image Processing              |
| 📊 NumPy        | Numerical Computing           |
| 🐼 Pandas       | Data Processing               |
| 📈 Matplotlib   | Visualization                 |
| 🎨 Seaborn      | Statistical Visualization     |
| 🤗 Gradio       | Interactive Deployment        |
| ☁️ Google Colab | Model Development Environment |

---

# 📁 Repository Structure

```text
📦 Facial-Emotion-Classification
│
├── 📁 train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── 📁 test/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── 📓 Deep Learning Model for Facial Emotion Classification.ipynb
│
├── 🧠 ResNet50_Transfer_Learning.keras
│
├── 🧠 Resnet_model_version_2.keras
│
└── 📄 README.md
```

> **Note:** Keep the actual repository structure synchronized with the files you have uploaded to GitHub. The model files above are included in the README structure only if they are present in your repository.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Move into the project directory:

```bash
cd YOUR-REPOSITORY
```

Install the required libraries:

```bash
pip install tensorflow
pip install numpy pandas matplotlib seaborn
pip install opencv-python pillow
pip install scikit-learn
pip install gradio
```

---

# ▶️ Running the Project

### Option 1 — Google Colab

Upload the notebook to Google Colab and make sure the dataset directories are available at the paths expected by the notebook.

The notebook currently uses paths such as:

```text
/content/train
/content/test
```

Run the notebook cells sequentially.

---

### Option 2 — Local Environment

Place the dataset folders in the project directory:

```text
project/
├── train/
├── test/
└── notebook.ipynb
```

Then update the dataset paths in the notebook if required.

---

# 🔬 Experimental Components

The notebook demonstrates several important deep learning concepts:

```text
                    ┌─────────────────────┐
                    │ Facial Image Data   │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
       Preprocessing     Augmentation      Class Weighting
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                     ┌─────────────────┐
                     │ Deep Learning   │
                     │ Model Training  │
                     └────────┬────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
               CNN          VGG16       ResNet50V2
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                        Evaluation
                              │
                              ▼
                         Deployment
```

---

# 💡 Key Learning Outcomes

This project demonstrates practical experience with:

* Convolutional Neural Networks
* Transfer Learning
* Image Classification
* Fine-Tuning pretrained networks
* TensorFlow/Keras
* Image augmentation
* Class imbalance handling
* Model regularization
* Model checkpointing
* Early stopping
* Learning-rate scheduling
* Classification metrics
* Confusion matrix interpretation
* ROC/AUC analysis
* Model serialization
* Interactive AI deployment

---

# 📌 Important Implementation Details

### Custom CNN

```text
Input → 48×48×1
```

### Transfer Learning Models

```text
Input → 224×224×3
```

### Number of Classes

```text
7
```

### Output Activation

```text
Softmax
```

### Loss Function

```text
Categorical Cross-Entropy
```

### Main Optimizers

```text
Adam
```

---

# 📊 Evaluation Dashboard

| Evaluation Component     | Implemented |
| :----------------------- | :---------: |
| Training Accuracy        |      ✅      |
| Test/Validation Accuracy |      ✅      |
| Precision                |      ✅      |
| Recall                   |      ✅      |
| F1-Score                 |      ✅      |
| Confusion Matrix         |      ✅      |
| Classification Report    |      ✅      |
| ROC Curve                |      ✅      |
| AUC                      |      ✅      |
| Class Weighting          |      ✅      |
| Data Augmentation        |      ✅      |
| Model Checkpointing      |      ✅      |

---

# 🎯 Project Highlights

<p align="center">

|         🧠 Models        | 😊 Classes |    🖼️ Input    | 🚀 Deployment |
| :----------------------: | :--------: | :-------------: | :-----------: |
| CNN • VGG16 • ResNet50V2 |      7     | 48×48 / 224×224 |     Gradio    |

</p>

---

# 🔮 Future Improvements

Potential future extensions include:

* Real-time webcam emotion recognition
* Face detection before emotion classification
* More extensive hyperparameter optimization
* Cross-validation experiments
* Additional pretrained architectures
* Ensemble learning
* Model explainability using Grad-CAM
* Confidence-score visualization
* Improved handling of minority emotion classes
* Mobile or web application deployment
* API-based inference
* Model quantization for lightweight deployment

---

# ⚠️ Limitations

The notebook represents an experimental deep learning workflow and includes multiple model configurations.

Some implementation choices are environment-specific, particularly the Google Colab paths and Google Drive model-storage paths.

The test directory is used as validation/evaluation data in parts of the notebook, so users should distinguish between **validation during model development** and a completely independent **held-out test set** when reporting final experimental results.

---

# 👨‍💻 About Me
## Shahid Azam

**MS Computer Science Student**
**Specialization in Artificial Intelligence**

*Institute of Management Sciences, Peshawar, Pakistan*

I enjoy building complete software systems—from networking and operating system concepts to backend architecture, machine learning, and distributed systems.

- 🎓 **Degree / Field:** MS in Computer Science (Specialization in Artificial Intelligence)
- 🏆 **Achievements:** Designed and executed 30+ machine learning and data visualization projects delivering actionable insights
- 🎯 **Focus Areas:** Data Visualization, AI/ML, Big Data Mining
- 📬 **Contact:** shahidazam2020@gmail.com
- <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="18" height="18" valign="middle" /> WhatsApp: <a href="https://wa.me/+923412772594" target="_blank">Chat on WhatsApp</a>

---
# 🤝 Connect With Me

<p align="center">

<a href="https://github.com/shahidazam2020-oss">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</a>

<a href="https://www.linkedin.com/in/shahid-azam-mughal-787b58235">
<img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
</a>

<a href="mailto:shahidazam2020@gmail.com">
<img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
</a>

</p>

---

# ⭐ Repository

If you find this project useful for learning or experimentation, consider giving the repository a ⭐.

<p align="center">

**⭐ Star this repository • 🍴 Fork it • 🧠 Explore the code • 🚀 Build something new**

</p>

---

# 📜 License

This project is intended for educational, research, and experimental purposes.

If you reuse or extend this project, please provide appropriate attribution to the original work and dataset sources where applicable.

---

<p align="center">

### 🧠 From Facial Images → Deep Learning → Emotion Intelligence

**Built with Python, TensorFlow & Computer Vision**

</p>

---
# ⭐ Support

*If you find this repository useful for learning Machine Learning, consider giving it a ⭐.*

*Your feedback, suggestions, and contributions are welcome.*

---

<p align="center">
  <b>🧠 Learn Machine Learning • 💻 Build Models • 📊 Analyze Data • 🚀 Create Projects</b>
</p>

<p align="center">
  Made with ❤️ for learning, experimentation, and continuous improvement.
</p>
