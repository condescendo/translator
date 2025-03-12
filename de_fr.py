# import streamlit as st
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# st.title('📚 Translator')
# # @st.cache(allow_output_mutation=True)
# # @st.cache_resource 
# def get_model():
#     model_name = "microsoft/Phi-4-mini-instruct"
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return tokenizer,model



# user_input = st.text_area('Enter German  Text')
# button = st.button("Translate to French")

# tokenizer,model = get_model()

# d = {
    
#   1:'Toxic',
#   0:'Non Toxic'
# }

# if user_input and button :
#     prompt = "Translate the following text from German into French: \"{user_input}\".\nOutput (French):"
#     # test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
#     # test_sample
#     # output = model(**test_sample)
#     # st.write("Logits: ",output.logits)
#    #  y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
#     # st.write("Prediction: ",d[y_pred[0]])
#    ## my output
#     messages = [
#         {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=32768
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
    
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     response

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

st.title('📚 Translator')

@st.cache_resource 
def get_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"  # Verified correct model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True  # Required for Phi models
    )
    return tokenizer, model

tokenizer, model = get_model()

user_input = st.text_area('Enter German Text')
button = st.button("Translate to French")

if user_input and button:
    prompt = f"""<|user|>
Translate this German text to French:
{user_input}
<|assistant|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,  # Reduced from 32k to prevent OOM errors
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Clean up the response by removing the prompt
    translated_text = response.split("<|assistant|>")[-1].strip()
    st.write(translated_text)
