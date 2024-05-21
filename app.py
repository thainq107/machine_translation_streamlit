import os
import gdown
import torch
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model(gdrive_id='13BlpbV2l0xBynkR1tYvY68WXAx4jPSkl'):

  model_path = 't5-small-en-vi'
  if not os.path.exists(model_path):
    # download folder
    gdown.download_folder(id=gdrive_id)
  tokenizer = T5Tokenizer.from_pretrained(model_path)
  model = T5ForConditionalGeneration.from_pretrained(model_path)
  model.eval()
  return tokenizer, model

tokenizer, model = load_model()

def inference(
    text,
    max_length=75,
    beam_size=5
    ):
    tokenized_text = tokenizer.encode(text, return_tensors="pt")
    summary_ids = model.generate(
        tokenized_text,
        max_length=max_length, 
        num_beams=beam_size,
        repetition_penalty=2.5, 
        length_penalty=1.0, 
        early_stopping=True
    )
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
      
    return output

def main():
  st.title('Machine Translation')
  st.title('Model: T5_SMALL. Dataset: EN-VI')
  text_input = st.text_input("Sentence: ", "I go to school.")
  result = inference(text_input)
  st.success(result) 

if __name__ == '__main__':
     main() 
