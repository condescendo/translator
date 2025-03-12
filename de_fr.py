import streamlit as st
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title('📚 Translator')
# @st.cache(allow_output_mutation=True)
@st.cache_resource 
def get_model():
    model_name = "Qwen/QwQ-32B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter German Text Text')
button = st.button("Translate to French")

d = {
    
  1:'Toxic',
  0:'Non Toxic'
}

if user_input and button :
    prompt = "Translate the following text from German into French: \"{user_input}\".\nOutput (French):"
    # test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    # output = model(**test_sample)
    # st.write("Logits: ",output.logits)
   #  y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    # st.write("Prediction: ",d[y_pred[0]])
   ## my output
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response
