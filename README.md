# Rice Grain Tracking and Classification for Automated Quality Inspection 🌾🤖

This project automates the **quality inspection of rice grains** using computer vision and deep learning.  
It detects, classifies, and tracks individual rice grains in video footage — ensuring that **each grain is counted only once** across frames.  

🚀 Built during my **project-based internship at SSG Embedded Solutions (Jan–Feb 2024)**.  
Achieved **91–94% accuracy** in grain classification and delivered a working end-to-end solution.  


## 📌 Problem Statement
Manual rice quality inspection is slow, error-prone, and subjective.  
This project aims to **automate grain inspection** by:
- Detecting rice grains in video
- Classifying grains by type/quality
- Tracking their movement to **avoid double-counting**
- Producing accurate per-class counts for analysis


## 🔧 Key Highlights
- ✅ **Custom dataset** → Collected and annotated 150+ rice grain images  
- ✅ **YOLOv8 for detection & classification** → Custom-trained for rice grains  
- ✅ **Centroid-based tracking algorithm** → Each grain receives a unique ID  
- ✅ **Consistent counting** → Grains are not double-counted across frames  
- ✅ **Automated reporting** → Final per-class grain counts after video processing  

## Output Video Frame
<img src="imag/frame2.png" width="400">
## 🛠️ Tech Stack
- **Python** (OpenCV, NumPy, SciPy)  
- **YOLOv8** (Ultralytics) for detection & classification  
- **Custom Object Tracking Algorithm** (Hungarian algorithm + centroid logic)  
- **Matplotlib / Visualization** for results  
- **Jupyter Notebooks** for experimentation  


## Current Implementation

At this stage, the model has been trained and evaluated on 2 major classes from the dataset, namely:

Broken Rice Grains

Full Rice Grains

## Dataset Scope

The full dataset actually contains some examples of other 3 out of 5 distinct classes of rice grain quality, including variations in size, shape, and texture.
This subset was chosen for the initial proof of concept to demonstrate the feasibility of applying deep learning to rice quality inspection.


## 📊 Results
- Achieved **91–94% accuracy** in classification  
- Successfully tracked grains across frames with unique IDs  
- Generated **automated grain counts by class**  


## 🎥 Demo
- Input video : https://drive.google.com/file/d/1OjnFaMEnofQ3uxBp18bfvStkwYUYSC1U/view?usp=sharing
- watch Output video on : https://drive.google.com/file/d/15SHCdhua_MHmCiFbo6SAK701jzx-wXp2/view?usp=sharing

## 📊 Grain Counts by Class (example output on input video)
Grain counts by class:
Class 1: 62
Class 0: 27

## 🚀 How to Run
```bash
# Clone repository
git clone https://github.com/yourusername/Rice-grain-Quality-Inspection.git
cd Rice-grain-Quality-Inspection

# Install requirements
pip install -r requirements.txt

# Run the pipeline
python src/main.py
```

## 🌟 Skills Demonstrated

Computer Vision (OpenCV, YOLOv8)

Object Detection & Classification

Custom Tracking Algorithm Design

Frame-wise Consistency & Data Reporting

End-to-End AI Project Execution

## 👤 Author

Nidhish P
📌 Project completed during internship at SSG Embedded Solutions (Jan–Feb 2024)

🔗 LinkedIn : https://www.linkedin.com/in/nidhish-parke-492883256 | GitHub : https://github.com/Nidhish-p

## ⭐ Acknowledgments

Guidance from Milind Mushrif during the internship

YOLOv8 framework by Ultralytics

## Future Work & Scope

To make the project more robust and closer to real-world application, the following steps are planned:

Expand to All Classes – Extend the classification from 2 classes to all 5 classes present in the dataset.

Balanced Training – Handle class imbalance by applying data augmentation or weighted loss functions.

Model Optimization – Improve accuracy and inference speed for real-time inspection on video streams.

Explainability – Integrate model interpretability tools (e.g., Grad-CAM) to visualize why the model makes certain predictions.

Deployment – Package the model into a deployable web or mobile app interface so non-technical users can use it.

Industry Application – Scale the system for bulk grain quality inspection with integration into automated quality control pipelines.
