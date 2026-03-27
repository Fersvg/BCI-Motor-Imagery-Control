"""
BCI Cursor Control using EEG Motor Imagery

Pipeline:
- CSP feature extraction
- LDA classification
- Probability-based continuous control

This implementation uses predict_proba() to simulate continuous BCI control.
"""

import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import pygame
import sys


# ==========================================
# 1. LOAD DATA
# ==========================================

# EEG motor imagery dataset (left vs right hand)
subject = 1
runs = [4, 8, 12]

print("Downloading data...")
files = eegbci.load_data(subject, runs)

raws = [read_raw_edf(f, preload=True) for f in files]
raw = mne.concatenate_raws(raws)

# Standardize EEG and apply electrode montage
eegbci.standardize(raw)
raw.set_montage('standard_1020')


# ==========================================
# 2. PREPROCESSING
# ==========================================

# Band-pass filter (motor imagery frequencies)
raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')

events, _ = mne.events_from_annotations(raw)

# Class mapping: left vs right hand
event_id = dict(left=2, right=3)

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=1.,
    tmax=4.,
    baseline=None,
    preload=True
)

X = epochs.get_data()
y = epochs.events[:, -1]


# ==========================================
# 3. TRAIN MODEL (CSP + LDA)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define pipeline
csp = CSP(n_components=4, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()

clf = Pipeline([
    ('CSP', csp),
    ('LDA', lda)
])

print("Training model...")
clf.fit(X_train, y_train)


# ==========================================
# 4. EVALUATION (PROBABILITY-BASED)
# ==========================================

# Predict probabilities instead of classes
proba_test = clf.predict_proba(X_test)

# Convert probabilities to class labels
y_pred = np.argmax(proba_test, axis=1) + 2

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (from proba): {acc:.2f}\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig_cm, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Left", "Right"])
disp.plot(ax=ax)
ax.set_title("Confusion Matrix")
plt.show()


# ==========================================
# 5. PYGAME INTERFACE (ANALYSIS MODE)
# ==========================================

pygame.init()

WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BCI Cursor Control - Analysis Mode")

font = pygame.font.SysFont(None, 28)

# Initial cursor position
cursor_x = WIDTH // 2
cursor_y = HEIGHT // 2

# Control parameters
speed = 10
gain = 5
threshold = 0.1
confidence_min = 0.55

# Get class indices
class_list = list(clf.classes_)
idx_left = class_list.index(2)
idx_right = class_list.index(3)

# Compute model bias (systematic tendency)
proba_test = clf.predict_proba(X_test)
bias = np.mean(proba_test[:, idx_right] - proba_test[:, idx_left])

print(f"Detected bias: {bias:.4f}")


# ==========================================
# 6. PRECOMPUTE CURSOR POSITIONS
# ==========================================

positions = [WIDTH // 2] * (len(X_test) + 1)
probas_cache = []

cursor_x = WIDTH // 2

for i in range(len(X_test)):
    sample = X_test[i:i + 1]
    proba = clf.predict_proba(sample)[0]

    proba_left = proba[idx_left]
    proba_right = proba[idx_right]

    # Continuous control signal
    raw_move = (proba_right - proba_left) - bias

    # Confidence filtering
    confidence = max(proba_left, proba_right)
    if confidence < confidence_min:
        raw_move = 0

    # Noise thresholding
    if abs(raw_move) < threshold:
        raw_move = 0

    # Scale movement
    delta = int(raw_move * speed * gain)

    # Update cursor position
    cursor_x += delta
    cursor_x = max(0, min(WIDTH, cursor_x))

    positions[i + 1] = cursor_x
    probas_cache.append((proba_left, proba_right, raw_move, delta))


# ==========================================
# 7. INTERACTIVE NAVIGATION
# ==========================================

index = 0
running = True

while running:

    cursor_x = positions[index]

    if index == 0:
        proba_left, proba_right, raw_move, delta = (0.5, 0.5, 0, 0)
    else:
        proba_left, proba_right, raw_move, delta = probas_cache[index - 1]

    direction = (
        "LEFT" if raw_move < 0 else
        "RIGHT" if raw_move > 0 else
        "NONE"
    )

    # ==========================================
    # DRAW
    # ==========================================

    screen.fill((30, 30, 30))

    # Center reference line
    pygame.draw.line(screen, (200, 200, 200), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 1)

    # Cursor
    pygame.draw.circle(screen, (0, 255, 0), (cursor_x, cursor_y), 12)

    # Probability bars
    bar_max_width = 300

    pygame.draw.rect(
        screen,
        (255, 80, 80),
        (50, 50, int(proba_left * bar_max_width), 25)
    )

    pygame.draw.rect(
        screen,
        (80, 150, 255),
        (WIDTH - 350, 50, int(proba_right * bar_max_width), 25)
    )

    # ==========================================
    # DEBUG METRICS
    # ==========================================

    text1 = font.render(f"Sample: {index}", True, (255, 255, 255))
    text2 = font.render(f"Proba L/R: {proba_left:.2f} / {proba_right:.2f}", True, (255, 255, 255))
    text3 = font.render(f"Bias: {bias:.3f}", True, (255, 255, 255))
    text4 = font.render(f"Raw move: {raw_move:.3f}", True, (255, 255, 255))
    text5 = font.render(f"Delta px: {delta}", True, (255, 255, 255))
    text6 = font.render(f"Direction: {direction}", True, (255, 255, 0))

    screen.blit(text1, (50, 120))
    screen.blit(text2, (50, 150))
    screen.blit(text3, (50, 180))
    screen.blit(text4, (50, 210))
    screen.blit(text5, (50, 240))
    screen.blit(text6, (50, 270))

    pygame.display.flip()

    # ==========================================
    # CONTROLS
    # ==========================================

    waiting = True
    while waiting:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False
                waiting = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_RIGHT and index < len(X_test):
                    index += 1
                    waiting = False

                elif event.key == pygame.K_LEFT and index > 0:
                    index -= 1
                    waiting = False

                elif event.key == pygame.K_ESCAPE:
                    running = False
                    waiting = False


pygame.quit()
sys.exit()