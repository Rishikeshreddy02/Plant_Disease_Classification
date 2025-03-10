# **Plant Disease Classification**  

## **Overview**  
This project focuses on classifying plant diseases using deep learning techniques. The model is trained on a dataset of **17,000 images** from the **PlantVillage dataset** and leverages CNN architectures, including **VGG16, ResNet50, and Vision Transformers (ViTs) like DINOViT**, to achieve high classification accuracy. The goal is to assist in **early disease detection**, improving agricultural productivity and plant health management.  

## **Features**  
✅ **Deep Learning Models**: Utilized **VGG16, ResNet50, and Vision Transformers (DINOViT)** for image classification.  
✅ **Extensive Dataset**: Trained on **17,000 images** for robust performance.  
✅ **Data Augmentation & Preprocessing**: Implemented various transformation techniques to enhance model generalization.  

## **Dataset**  
The dataset consists of **15 plant disease categories**, sourced from the **PlantVillage dataset**:  

- **Pepper__bell___Bacterial_spot**  
- **Pepper__bell___healthy**  
- **Potato___Early_blight**  
- **Potato___Late_blight**  
- **Potato___healthy**  
- **Tomato_Bacterial_spot**  
- **Tomato_Early_blight**  
- **Tomato_Late_blight**  
- **Tomato_Leaf_Mold**  
- **Tomato_Septoria_leaf_spot**  
- **Tomato_Spider_mites_Two_spotted_spider_mite**  
- **Tomato__Target_Spot**  
- **Tomato__Tomato_YellowLeaf__Curl_Virus**  
- **Tomato__Tomato_mosaic_virus**  
- **Tomato_healthy**  

## **Technologies Used**  
🔹 **Python**  
🔹 **TensorFlow & Keras**  
🔹 **OpenCV**  
🔹 **NumPy & Pandas**  
🔹 **Matplotlib & Seaborn**  

## **Model Training**  

### **1️⃣ Data Preprocessing**  
- Image resizing, normalization, and augmentation.  

### **2️⃣ Model Selection**  
- Implemented **transfer learning** with **VGG16 and ResNet50**.  
- Experimented with **Vision Transformers (DINOViT)** for improved feature extraction.  

### **3️⃣ Training & Evaluation**  
- Fine-tuned models to achieve optimal accuracy.  
- Evaluated performance using **accuracy and loss metrics**.  

## **How to Run on Kaggle**  

### **Step 1: Import the Dataset**  
1. Go to [Kaggle](https://www.kaggle.com/).  
2. Search for **"PlantVillage Dataset"** and add it to your Kaggle notebook as an **input dataset**.  

### **Step 2: Clone the Repository**  
Run the following command in a Kaggle notebook to clone this repository:  
```bash
!git clone https://github.com/yourusername/plant-disease-classification.git
```
Move into the project directory:  
```bash
%cd plant-disease-classification
```

### **Step 3: Install Dependencies**  
Run the following command to install required libraries:  
```bash
!pip install -r requirements.txt
```

### **Step 4: Run the Notebook**  
1. Open the **Plant_Disease_Classification.ipynb** notebook in Kaggle.  
2. Execute the cells to train and evaluate the model.  

## **Results**  
📌 The model achieved hing accuracy** on the validation set.  
📌 **Vision Transformers** showed improved classification capabilities compared to CNN-based models.  

## **Future Work**  
🚀 Expand dataset with **more plant species**.  
🚀 Deploy the model as a **web application**.  
🚀 Improve model efficiency for **real-time classification**.  

## **Acknowledgments**  
📌 Inspired by various **plant disease detection studies** in deep learning.  

---
