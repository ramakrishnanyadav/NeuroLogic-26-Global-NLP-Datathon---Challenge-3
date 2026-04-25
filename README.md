# 🛡️ Lumina Mod-AI: Cross-Lingual Content Moderation
## NeuroLogic '26 Global NLP Datathon - Challenge 3 Submission

**The Problem**: Toxic behavior scales beyond what human moderators can handle, particularly in bilingual communities where code-switched Hindi and English mask hate speech.
**The Solution**: A scalable AI moderation layer backed by `mDeBERTa-v3` architecture, providing deep semantic context across languages to secure digital environments.

### 📈 Verified Core Metrics
* **Macro F1-Score**: `0.9444`
* **Mean ROC-AUC**: `0.9848`
* **System Accuracy**: `0.9444`

### 🛠️ Architecture & Strategy
* **Model Backbone**: `microsoft/mdeberta-v3-base` (Disentangled Attention structure outperforms XLM-RoBERTa on Hindi-English context).
* **Prevention Mechanisms**: Integrated 5-pass Multi-Sample Dropout regularizing the final classification space to eliminate cross-lingual overfitting.
* **Explainability & Demo**: Real-time integration of inference pipelines via Streamlit + SHAP Value visualization for bias auditing.
