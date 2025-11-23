import streamlit as st
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# ======================================================
# 1. Load Model + Tokenizer + Label Encoder
# ======================================================
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert_model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    with open("bert_label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model()


# ======================================================
# 2. CSS Injection (Gradient Header + Clean Theme)
# ======================================================
def inject_css():
    st.markdown("""
    <style>
        .app-header {
            background: linear-gradient(90deg, #ffcc00, #ff8800);
            padding: 26px;
            border-radius: 14px;
            text-align: center;
            font-size: 38px;
            font-weight: 800;
            color: black !important;
            margin-bottom: 25px;
        }
        .sub-text {
            font-size: 18px;
            opacity: 0.9;
            margin-top: -8px;
        }
        body, .stApp {
            background-color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

inject_css()


# ======================================================
# 3. Header UI
# ======================================================
st.markdown('<div class="app-header">Plagiarism Checker</div>', unsafe_allow_html=True)

st.write("### Check whether your text is plagiarised or not 🔎")
st.write("### Text Detection between **Google / Human / ChatGPT**")

# ======================================================
# 4. Input Text Area
# ======================================================
user_text = st.text_area("Paste your text here:", height=220)


# ======================================================
# 5. Prediction Function
# ======================================================
def predict(text):
    if not text.strip():
        return None

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    model.eval()
    with torch.no_grad():
        output = model(**encoding)

    logits = output.logits
    raw_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    probs = [float(x) for x in raw_probs.tolist()]

    class_labels = label_encoder.classes_
    pred_idx = int(np.argmax(probs))
    pred_label = class_labels[pred_idx]

    percentages = {cls: round(p * 100, 2) for cls, p in zip(class_labels, probs)}

    return pred_label, probs, percentages


# ======================================================
# 6. Analyze Button — with Plagiarism Verdict
# ======================================================
if st.button("Analyze"):
    result = predict(user_text)

    if result:
        pred_class, raw_probs, percentages = result

        st.success(f"Predicted Class: **{pred_class.upper()}**")

        st.write("### 🔍 AI Likelihood")
        for cls in label_encoder.classes_:
            st.write(f"**{cls}** → {percentages[cls]}%")

        # -------------------------------
        # 7. Plagiarism Verdict
        # -------------------------------
        human_pct = percentages.get("human", 0)
        google_pct = percentages.get("google", 0)
        chatgpt_pct = percentages.get("chatgpt", 0)

        st.write("---")
        st.markdown("### 📝 Plagiarism Verdict")

        if human_pct >= 90:
            st.success("✅ **Your text appears to be Human-written (0% plagiarised / 0% AI-generated).**")
        elif human_pct >= 60:
            st.info("⚠️ **Your text is mostly Human but may contain slight AI-like patterns.**")
        elif human_pct >= 40:
            st.warning("⚠️ **Your text shows a mix of AI and Human writing. Possible partial AI content.**")
        else:
            ai_source = "ChatGPT" if chatgpt_pct > google_pct else "Google"
            st.error(f"❌ **Your text is likely plagiarised / AI-generated ({ai_source}).**")

        st.markdown("---")

        # -------------------------------
        # 8. Class Probabilities (JSON)
        # -------------------------------
        st.subheader("📊 Class Probabilities")
        clean_dict = {cls: float(p) for cls, p in zip(label_encoder.classes_, raw_probs)}
        st.json(clean_dict)

        # -------------------------------
        # 9. Progress Bars (0–1 scale)
        # -------------------------------
        for cls, p in clean_dict.items():
            st.write(f"**{cls}**")
            st.progress(float(p))

