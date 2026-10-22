A 1D convolutional neural network in TensorFlow/Keras that classifies single ECG heartbeats into five categories, reaching 96.4% accuracy on 21,892 unseen test records from the MIT-BIH Arrhythmia dataset.

Stack: Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · Matplotlib/Seaborn

Problem

Each record is one heartbeat: 187 normalised signal values plus a class label. The data is heavily imbalanced, with 82.8% of training beats normal, so a naive model can score high accuracy while missing the rare, clinically important classes.

Label	Class	Meaning
0	N	Normal beat
1	S	Supraventricular ectopic beat
2	V	Ventricular ectopic beat
3	F	Fusion beat
4	Q	Unknown / paced beat
Approach
Data checks: 87,554 training and 21,892 test records, no missing values, feature distributions compared across the two sets.
Class imbalance: undersampled the normal class from 72,471 to 8,000 beats so the model learns the minority classes.
Model: three Conv1D layers (32, 64, 128 filters) → MaxPool → Dropout (0.5) → Dense (512, 1024) → Softmax (5 classes). Optimiser: Adam; loss: sparse categorical cross-entropy.
Evaluation: trained on the balanced set and evaluated on the original, untouched test set, so results reflect the real-world class distribution.
Results (full test set, 21,892 records)

Accuracy: 96.4% · Macro F1: 0.85 · Weighted F1: 0.96

Class	Precision	Recall	F1
N (0)	0.99	0.97	0.98
S (1)	0.56	0.85	0.68
V (2)	0.90	0.96	0.93
F (3)	0.63	0.82	0.71
Q (4)	0.96	0.99	0.98

What the numbers say: recall is strong for every class, so the model catches most abnormal beats. Precision for S and F is low, because undersampling makes the model over-predict the rare classes. That trade-off is often acceptable in screening, where a missed abnormal beat costs more than a false alarm, but it is the main thing to improve.

Next steps
Use class weights or SMOTE instead of discarding normal beats, and compare precision on S and F.
Add a validation split with early stopping on val_loss.
Package inference as a small script or API in a Docker container.
Run it
bash
git clone https://github.com/Sreesekhar-crtl/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

Download mitbih_train.csv and mitbih_test.csv from the ECG Heartbeat Categorization Dataset on Kaggle into a data/ folder (the files are too large for GitHub), then open ecg_classification.ipynb.

Repository structure
├── ecg_classification.ipynb   # full pipeline: EDA, balancing, model, evaluation
├── requirements.txt
├── images/                    # confusion matrix, training curve, class distribution
├── data/                      # not committed — see "Run it"
└── README.md
Data source

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol 20(3):45-50 (2001), via PhysioNet. Pre-processed heartbeat version by Fazeli (Kaggle).
