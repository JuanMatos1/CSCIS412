# CSCIS412 — AI Data Confidentiality & Prompt Injection Detection

## Overview

This project investigates confidentiality risks introduced when AI-powered tools are integrated into organizational environments.

The project focuses on several ways sensitive information can be exposed through AI systems, including:

- Prompt injection attacks
- Accidental data leakage
- Training-data memorization
- Conversational context retention
- Insufficient data-loss prevention controls

The main implementation is a machine-learning prompt injection detector that classifies user prompts as safe or malicious before they are sent to an AI system.

---

## Research Question

**What risks do AI-powered tools pose to organizational data confidentiality when integrated into enterprise networks?**

---

## Main Features

### Prompt Injection Detection

Built a machine-learning classifier using:

- Python
- scikit-learn
- TF-IDF
- Logistic Regression
- pandas
- NumPy
- Jupyter Notebook

The system combines:

- Word-level TF-IDF features
- Character-level TF-IDF features
- Logistic Regression classification

The model evaluates whether a prompt is likely to be:

- Safe
- Malicious / Prompt Injection

---

## Dataset

The final dataset contains more than 42,000 prompts composed of malicious prompt-injection examples and legitimate safe prompts.

The data preparation process includes:

- Text normalization
- Duplicate removal
- Label preparation
- Dataset balancing considerations
- Train/test splitting

---

## Model Evaluation

The classifier achieved approximately **99.9% test accuracy** on the project dataset.

The project does not rely only on accuracy. Additional evaluation includes:

- Confusion matrix
- False-positive analysis
- Misclassification analysis
- Probability threshold testing
- Custom prompt testing

A classification threshold of approximately **0.60** was also tested to study the tradeoff between identifying malicious prompts and incorrectly blocking legitimate prompts.

> High accuracy on this dataset does not guarantee equivalent performance in a real production environment. Dataset imbalance and differences between training data and real-world prompts must be considered.

---

## Data Loss Prevention

The project also includes a simple DLP-style redaction mechanism designed to detect and remove potentially sensitive information before it is processed by an AI system.

Examples of sensitive information considered include:

- Email addresses
- API keys
- Internal identifiers
- Customer information
- Other confidential strings

---

## Additional Security Experiments

### 1. Accidental Data Leakage

A simulated corporate environment was created using Python, pandas, and SQLite.

Synthetic confidential data included examples such as:

- Customer IDs
- Invoice information
- Internal project information
- API keys
- Meeting notes
- Email addresses

The experiment demonstrates how sensitive information entered into an AI assistant could potentially appear in locations such as:

- Chat logs
- Browser/session history
- Telemetry
- Vendor logs
- Application memory

### 2. Training-Data Memorization

A small character n-gram model was trained using synthetic confidential strings.

The experiment demonstrates how a model may reproduce information that appeared in its training data, illustrating the confidentiality risks associated with model memorization.

### 3. AI Assistant Comparison

The project also compares how different AI assistants handle previously provided information.

Systems tested include:

- ChatGPT
- Microsoft Copilot
- Gemini
- DeepSeek

The tests examine how each system responds when progressively asked to recall, combine, or provide previously discussed information.

---

## Technologies Used

- Python
- scikit-learn
- pandas
- NumPy
- SQLite
- Jupyter Notebook
- TF-IDF
- Logistic Regression
- Machine Learning
- Cybersecurity concepts
- Data Loss Prevention concepts

---

## Project Structure

```text
CSCIS412/
│
├── [prompt injection implementation]
├── [data leakage experiment]
├── [training-data memorization experiment]
├── Chats Outcome/
│   ├── ChatGPT/
│   ├── Copilot/
│   ├── Gemini/
│   └── DeepSeek/
│
└── README.md