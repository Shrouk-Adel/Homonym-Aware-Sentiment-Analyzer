# Homonym-Aware-Sentiment-Analyzer

### Understanding Contextual Meaning & Resolving Sentiment Inversion in NLP

This project investigates one of the hardest challenges in Natural Language Processing:
**sentiment misclassification caused by homonyms, negation, and contextual ambiguity**.

Traditional models often fail when sentiment changes based on context. For example:

* **“I hate the selfishness in you” → Negative**
* **“I hate anyone who can hurt you” → Positive**

Although both sentences contain *negative words* (“hate”, “hurt”), their meanings differ.
This project evaluates how different models handle such contextual complexities.

---

# 📌 Project Objective

To build and evaluate a pipeline that detects and resolves **homonym and context-related sentiment errors**, using:

* A traditional **Bidirectional LSTM baseline**
* Two fine-tuned Transformer models:

  * **DistilBERT**
  * **BERT (base)**

The goal is to measure how well these models understand **deep semantic context**, especially in challenging custom-made test cases involving negation and sentiment inversion.

---

# 📊 Tools & Technologies

### **Programming Frameworks**

* **TensorFlow & Keras** → LSTM + DistilBERT fine-tuning
* **PyTorch** → BERT fine-tuning via Hugging Face Trainer

### **NLP Libraries**

* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* Scikit-learn

### **Scientific Stack**

* Pandas, NumPy
* Matplotlib, Seaborn

### **Dataset**

* **Stanford Sentiment Treebank v2 (SST-2)**
  Binary sentiment classification: *positive* or *negative*

### **Models Used**

* `distilbert-base-cased`
* `bert-base-uncased`

---

# 📁 Dataset Description

* Dataset: **SST-2 (Stanford Sentiment Treebank)**
* Task: **Binary classification**
* Inputs: Short text sentences
* Labels: *positive* or *negative*

---

# 🔬 Experiments

## **1. Baseline: Bidirectional LSTM**

### **Approach**

* Tokenized sentences → integer sequences
* Padding to 49 tokens
* Built BiLSTM with:

  * Embedding layer
  * BiLSTM (128 units)
  * Dense (64) + Sigmoid output

### **Results**

* **Validation Accuracy: 83.26%**
* Clear **overfitting**
* Poor performance on contextual inversion sentences

**Custom Case Failure Example:**

> "I hate anyone hurting you" → should be **Positive**, predicted **Negative**

---

## **2. DistilBERT Fine-Tuning (TensorFlow)**

### Method

* Loaded `distilbert-base-cased`
* Tokenized dataset using HF Tokenizer
* Dynamic batch padding
* Trained for 3 epochs (lr=1e-5)

### Results

* **Validation Accuracy: 89.56%**
* Major improvement over LSTM
* Much better understanding of context

### Custom Sentence Evaluation (Selected)

| Sentence                  | True | DistilBERT |
| ------------------------- | ---- | ---------- |
| I love you                | +    | +          |
| I hate anyone hurting you | +    | +          |
| I like rude people        | –    | –          |
| I don’t like rude people  | +    | +          |

---

## **3. BERT Fine-Tuning (PyTorch & Trainer API)**

### Method

* Loaded `bert-base-uncased`
* Tokenized with truncation (128 tokens)
* Used Hugging Face Trainer API
* Trained for 3 epochs (lr=2e-5)

### Results

* **Validation Accuracy: 92.78%**
* **F1 Score: 92.90%**
* Best performing model overall
* Strong contextual understanding
* One error on a custom test case

---

# 🏆 Final Comparison

| Model           | Validation Accuracy |
| --------------- | ------------------- |
| **BiLSTM**      | **83.26%**          |
| **DistilBERT**  | **89.56%**          |
| **BERT (base)** | **92%**             |

BERT provided the strongest performance and best contextual reasoning, though DistilBERT handled some nuanced cases better.

---

# 💡 Key Insights & Reflections

### **Challenges**

* High accuracy does **not** guarantee contextual understanding
* LSTM behaved like a bag-of-words model
* Transformers captured semantic meaning far better

 

# 🚀 Conclusion

This project demonstrates how Transformer-based models significantly outperform traditional LSTM architectures in handling **homonyms, negation, and sentiment inversion**, making them superior for real-world sentiment analysis tasks.

By fine-tuning DistilBERT and BERT, we successfully built a pipeline capable of analyzing subtle contextual cues that baseline models fail to detect.

 
