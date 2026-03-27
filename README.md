# BCI Cursor Control (EEG + CSP + LDA)
Brain-Computer Interface (BCI) prototype that decodes EEG motor imagery and translates it into **continuous cursor control using probabilities**.

This approach enables smoother and more realistic BCI control compared to traditional discrete classification 

---

## 🚀 Overview

This project implements a full BCI pipeline:

- EEG motor imagery classification (left vs right hand)
- Feature extraction with CSP (Common Spatial Patterns)
- Classification with LDA (Linear Discriminant Analysis)
- Continuous control using `predict_proba()`

Instead of discrete predictions, this system generates **smooth control signals**.

---

## 🧠 Key Idea

Most BCI systems use `predict()`.

This project uses `predict_proba()` to create continuous movement:

```python
raw_move = (proba_right - proba_left) - bias
```

---

## 🎛️ Control Parameters

- `speed` → base movement speed  
- `gain` → amplifies the control signal  
- `threshold` → ignores small noisy movements  
- `confidence_min` → filters low-confidence predictions 

These mechanisms improve stability by reducing noise and compensating for model bias.

Example values:

- `speed = 10`
- `gain = 5`
- `threshold = 0.1`
- `confidence_min = 0.55`

---

## ▶️ Installation & Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 🖥️ Features

- Continuous cursor control (not binary)
- Bias correction and confidence filtering
- Interactive visualization with Pygame
- Confusion matrix evaluation

---

## ⚙️ Tech Stack

- Python
- MNE
- Scikit-learn
- Pygame

---

## ⚠️ Limitations

- Offline simulation (no real-time EEG)
- Single subject dataset
- Binary classification (left vs right)

---

## 📄 License

MIT


