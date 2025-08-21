# Alzheimer's MRI Classification Using Deep Learning Models
A deep learning project comparing state-of-the-art convolutional neural networks (CNNs) to classify Alzheimer's disease stages from MRI scans.

## 🔍 Project Overview

This project applies deep learning models to classify Alzheimer's disease stages using brain MRI scans. It addresses the limitations of conventional diagnosis methods, which are often time-consuming and prone to human error, by providing an efficient and reliable diagnostic support system for medical professionals. We focused on four distinct stages of Alzheimer's progression: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. The project evaluates and compares the performance of sev...

## 🎯 Motivation & Goals

The primary goal of this project was to explore how different deep learning models can assist in detecting and classifying Alzheimer's Disease from MRI scans, aiming for a solution that is both accurate and practical for real-world application. The key objectives were to:

- **Design a Classification System**: Implement and evaluate deep learning models on a common dataset to fairly compare their performance.
- **Compare Performance**: Measure the effectiveness of each model in identifying different stages of Alzheimer's using metrics such as accuracy, precision, and recall.
- **Analyze Efficiency**: Evaluate the computational efficiency of each model by looking at factors like training time and model size, which are crucial for real-world applications.
- **Validate Visually**: Conduct visual tests on randomly selected MRI scans to validate the models' reliability from a non-technical perspective.

## 🗂 Table of Contents

- Installation  
- Usage  
- Project Structure  
- Dataset  
- Methodology  
- Results & Evaluation  
- Tech Stack  
- Future Work  
- Contact  

## 🔧 Installation

To get started, clone the repository and install the required Python packages.

```bash
git clone <https://github.com/VinayTrinadh7755/Alzheimers_MRI_Classification_Using_Deep_Learning_Models.git>
cd <Alzheimers_MRI_Classification_Using_Deep_Learning_Models>
pip install -r requirements.txt
```

## 🚀 Usage

Simply open the Jupyter notebooks in the project directory to run the analysis, model training, and visualizations.

```bash
jupyter notebook Densenet.ipynb
```

## 📁 Project Structure

```
.
├── Densenet.ipynb                   # Implementation of DenseNet model
├── Googlenet.ipynb                  # Implementation of GoogLeNet model
├── Resnet.ipynb                     # Implementation of ResNet model
├── VGG.ipynb                        # Implementation of VGG model
└── README.md                        # This file
```

## 📊 Dataset

The core of this project is the **Alzheimer's Dataset** from Kaggle, which contains brain MRI scans classified into four cognitive stages: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. The images capture subtle structural changes that are difficult for the human eye to detect, making it an ideal task for deep learning.

We addressed the imbalance in the dataset using a **weighted sampling technique** during training to ensure the model learned fairly from all categories.

## ⚙️ Methodology

### Deep Learning Models

This project evaluates four popular convolutional neural networks (CNNs) for classification:

- **GoogLeNet**: Utilizes inception blocks to perform multi-scale feature extraction.  
- **DenseNet**: Employs dense connections to maximize information flow between layers.  
- **ResNet**: Features residual connections to address the vanishing gradient problem in deep networks.  
- **VGGNet**: A traditional, deep CNN with a simple and uniform architecture.

### Training & Evaluation

The models were trained on a common dataset using consistent preprocessing techniques. Their performance was evaluated using various metrics, including **accuracy**, **precision**, **recall**, and **F1-score**.

## 🏆 Results & Evaluation

After a comprehensive evaluation, **GoogLeNet** was found to be the best-performing model, achieving a **test accuracy of 99.21%**. This model also provided the best trade-off between accuracy and computational efficiency.

### Key Observations:

- GoogLeNet was able to capture features at multiple scales, allowing it to converge quickly without overfitting.
- ResNet and DenseNet also performed well, particularly in identifying moderate and mild dementia cases.
- VGG-16 lagged behind the other models due to its slower convergence and less sophisticated architecture.
- Classification of early-stage diseases (Very Mild Demented vs. Non-Demented) remains the most challenging due to the subtle visual differences in MRI scans.

## 🛠 Tech Stack

- **Languages & Tools**: Python, Jupyter Notebook  
- **Frameworks**: PyTorch  
- **Data Handling**: NumPy, Pandas, PIL  
- **Visualization**: Matplotlib, Seaborn  
- **Algorithms**: VGGNet, GoogLeNet, ResNet, DenseNet  

## 🌱 Future Work

- Explore advanced deep learning models like Deep Reinforcement Learning (Deep RL) for more complex environments.  
- Introduce stochasticity to evaluate the robustness of the implemented models.  
- Incorporate larger and more diverse datasets to improve model generalization.  

## 📬 Contact

**Vinay Trinadh Naraharisetty**  
📧 vinaytrinadh9910@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/VinayTrinadh)  
🔗 [GitHub](https://github.com/VinayTrinadh)  

Feel free to reach out for questions, feedback, or collaboration!
