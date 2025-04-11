# 🧠 Fake Review Detection Using Deep Learning (LSTM + GPT-2 Augmentation)

This project aims to detect fake reviews using a deep learning approach with a Bidirectional LSTM architecture. The dataset is augmented using GPT-2 generated synthetic reviews for both fake and real classes. The final model is evaluated using multiple visualization tools including ROC curves, confusion matrix, and precision-recall curves.

---

## 📌 Project Highlights

- ✅ Deep learning model using **Bidirectional LSTM**
- ✅ Dataset augmented using **GPT-2** (via HuggingFace Transformers)
- ✅ Performance visualized with **matplotlib** and **seaborn**
- ✅ Compared using metrics like **accuracy**, **confusion matrix**, **ROC-AUC**, and **Precision-Recall**
- ✅ End-to-end pipeline: Data loading → Preprocessing → Training → Evaluation

---

## 📂 Dataset

The base dataset used is:

- **Preprocessed Fake Reviews Detection Dataset** (CSV)
- Augmented with **100 synthetic fake reviews** and **100 real reviews** using **GPT-2**

Make sure to include this file in your project root:
```plaintext
Preprocessed Fake Reviews Detection Dataset.csv
```

---

## 🚀 Model Architecture

The model is built using Keras Sequential API:

- `Embedding` layer for word vector representation
- `Bidirectional LSTM` for capturing forward and backward dependencies
- `Dropout` layers to prevent overfitting
- `Dense` layers for classification

---

## 📊 Visualizations

The following performance visualizations are included:

1. **Accuracy & Loss per Epoch**
2. **ROC Curve** with AUC Score
3. **Confusion Matrix**
4. **Precision-Recall Curve**

All graphs are generated using `matplotlib` and `seaborn`.

---

## 📈 Model Performance

| Metric         | Value     |
|----------------|-----------|
| Accuracy       | ~93.5%    |
| ROC AUC        | High (based on curve) |
| Precision & Recall | Evaluated and visualized |
| Traditional Model Baselines |
| - Logistic Regression | ~88% Accuracy |
| - Naive Bayes         | ~82% Accuracy |

---

## 🧪 How to Run

1. Clone this repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install required libraries
```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, you can install dependencies manually:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow transformers
```

3. Run the Jupyter Notebook
```bash
jupyter notebook final5.ipynb
```

Or open with Google Colab.

---

## 🛠️ Project Structure

```
├── final5.ipynb                  # Main notebook with code and visualizations
├── Preprocessed Fake Reviews Detection Dataset.csv
├── Augmented_Fake_Reviews.csv   # (Auto-generated)
├── README.md
└── requirements.txt             # (optional)
```

---

## 🧠 Technologies Used

- Python
- TensorFlow & Keras
- HuggingFace Transformers
- GPT-2
- Scikit-learn
- Matplotlib & Seaborn
- Pandas & NumPy

---

## 📝 Author

**Shivansh Lohani**  
Undergraduate, Computer Science Engineering  
This project was developed as part of the Project Based Learning(pBL) submission.

---

## 📌 Future Enhancements

- Add support for multilingual review detection
- Integrate behavioral features (reviewer history, frequency)
- Deploy model in real-time on a web platform
- Use advanced explainable AI tools (e.g., SHAP, LIME)

---

## 📄 License

This project is open-source and available under the MIT License.

```

---

Let me know if you'd like help making a `requirements.txt` too or need help pushing the repo live on GitHub!
