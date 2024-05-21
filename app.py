import os
import gdown
import torch
import streamlit as st
from transformers import MBart50TokenizerFast, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model(gdrive_id='1-boA9aNqG3AHdlGszo1rkbbIrzRI-bJ4'):

  model_path = 'mbart50-en-vi'
  if not os.path.exists(model_path):
    # download folder
    gdown.download_folder(id=gdrive_id)
  tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
  return tokenizer, model

@st.cache_data
def inference(
    text,
    max_length=75,
    beam_size=5
    ):
    
    tokenizer, model = load_model()
    with torch.no_grad():

        inputs = tokenizer(text, return_tensors='pt')
        preds = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=beam_size
        )

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
      
    return preds[0]

def main():
  st.title('Machine Translation')
  st.title('Model: MBART50. Dataset: EN-VI')
  text_input = st.text_input("Sentence: ", "I go to school.")
  result = inference(text_input)
  st.success(result) 

if __name__ == '__main__':
     main() 
