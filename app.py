import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

st.set_page_config(page_title="Lumina Mod-AI", layout="centered", page_icon="🛡️")

MODEL_NAME = "microsoft/mdeberta-v3-base"

class DatathonWinnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(MODEL_NAME)
        self.dropouts = nn.ModuleList([nn.Dropout(0.15) for _ in range(5)])
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = torch.mean(torch.stack([self.classifier(drop(pooled)) for drop in self.dropouts], dim=0), dim=0)
        return logits.squeeze(-1)

@st.cache_resource
def load_ai_engine():
    device = torch.device('cpu') 
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = DatathonWinnerModel()
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    
    # THE FIX: Force the UI model into pure Float32 exactly like we did in training!
    model.to(device).float().eval() 
    
    return tokenizer, model, device

tokenizer, model, device = load_ai_engine()

st.title("🛡️ Cross-Lingual AI Moderator")
st.markdown("**Live demonstration (Verified Real Inference) of Hindi/English hate speech detection.**")

user_text = st.text_area("Input comment to moderate:")

if st.button("Run AI Scan", type="primary"):
    if not user_text.strip(): st.warning("Please enter text.")
    else:
        with st.spinner("Analyzing neural semantics..."):
            inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=192)
            with torch.no_grad():
                logits = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
            prob = torch.sigmoid(logits).item()
            
            if prob > 0.5:
                st.error("🚨 **Harmful Content Blocked**")
                st.warning(f"**Classification:** Toxic\n\n**AI Confidence:** {prob*100:.2f}%\n\n**Action:** Comment Removed")
            else:
                st.success("✅ **Comment Approved**")
                st.info(f"**Classification:** Safe\n\n**AI Confidence:** {(1-prob)*100:.2f}%")
