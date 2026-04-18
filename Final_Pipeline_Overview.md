# Final Project Pipeline: Mental Health Crisis Detection on Reddit

This document provides a simple, concise, step-by-step breakdown of our final machine learning pipeline. It explains exactly what we built, how it works, and why we made our specific technical decisions.

---

## Step 1: Data Preprocessing & Feature Engineering
**What we did:** We took 4 years of raw Reddit posts (2019–2022) and transformed the chaotic text into clean, mathematical representations of human emotion.
- **Deep Learning Text Embeddings:** We passed the raw text through a state-of-the-art Transformer model to generate 384 mathematical dimensions (`win_emb_000` to `win_emb_383`) that represent the deep semantic meaning of what the user is saying.
- **Linguistic/Psychological Features:** We used natural language processing tools (like LIWC) to count specific emotional markers, such as anxiety, sadness, and posting frequency.
- **Temporal Windows:** Instead of looking at single posts, we grouped the text into "time windows," giving the algorithm a timeline of each user's psychological state leading up to a crisis.

**Why:** Machine Learning models cannot read English. They need numbers. This multi-stream approach ensures the model isn't just looking at keyword counts (which miss sarcasm and context), but instead understands the profound psychological undertones of the dialogue.

---

## Step 2: Temporal Data Splitting (The "Vault" Strategy)
**What we did:** 
- We merged **2019, 2020, and 2021** into a single massive dataset for Training and Validation.
- We aggressively locked the **2022 dataset** away in a "vault" entirely untouched by the training process. 

**Why:** To prevent **Data Leakage**. If the model trains on pieces of 2022 data, it might artificially memorize specific 2022 cultural trends. By holding 2022 out entirely as an "Out-of-Time" test set, we simulate reality: we prove our model can successfully generalize to future, unseen human behavior. 

*(Note: We also strictly split the data at the **User Level**, ensuring a single person’s posts weren't accidentally split across both the training and testing sets).*

---

## Step 3: Modeling & Handling Extreme Imbalance
**What we did:** We selected **XGBoost (eXtreme Gradient Boosting)** as our core algorithm. 
Because true mental health crises are incredibly rare (representing only **1.38%** of the dataset), we used native Class Weighting (`scale_pos_weight = 72`) rather than deleting data or synthesizing fake data.

**Why XGBoost?** XGBoost is the global industry standard for high-dimensional tabular data. 
**Why Class Weighting?** We experimented extensively with advanced sampling techniques (like SMOTE and Random Undersampling). However, drawing synthetic lines between 384-dimensional semantic text embeddings creates statistical garbage. Imposing a severe mathematical penalty inside XGBoost (saying: "Missing a Crisis is 72x worse than a False Alarm") proved to be the only scientifically stable way to handle the 1.38% base rate.

---

## Step 4: Hyperparameter Tuning
**What we did:** We used **Optuna**, an advanced optimization framework, to run 50 complex training trials testing hundreds of XGBoost parameter combinations (like tree depth and learning rate). 

**Why:** Tree-based models can easily overfit (memorize the data rather than learning the patterns). Optuna intelligently searches for the perfect balance of "learning" vs "generalizing" to maximize our predictive ability on unseen users.

---

## Step 5: Clinical Evaluation Strategy
**What we did:** We accepted a final model metric of **~50% Recall** and **~5% Precision**. We chose not to artificially inflate Precision at the expense of Recall.

**Why:** In clinical screening and crisis detection, **Recall is King**. 
- A **False Positive** (Low Precision) simply results in an automated Reddit Care message being sent to someone who is sad but not suicidal. The cost is negligible.
- A **False Negative** (Low Recall) means the system fails to identify a person in a severe mental health crisis. The cost is catastrophic. 
Our pipeline embraces the clinical methodology: cast a wide net to catch as many crises as possible, even if it brings in some false alarms. 

---

## Step 6: Interpretability (SHAP)
**What we did:** We applied **SHAP (Shapley Additive exPlanations)** to our finalized Black-Box model. This mathematical tool generated visual charts evaluating the exact impact every feature had on the model's final decision.

**Why:** A high-performing but unexplainable model is useless in the medical/psychological fields. SHAP proves to researchers and stakeholders exactly *which* specific emotional patterns, linguistic cues, and semantic embeddings the XGBoost algorithm relied on to trigger a Crisis flag, proving the model is clinically sound and not just guessing based on random noise.
