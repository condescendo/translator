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

st.title('📚 German to French Translator')

@st.cache_resource 
def load_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    return tokenizer, model

tokenizer, model = load_model()

user_input = st.text_area('Enter German Text:')
button = st.button("Translate")

if button and user_input:
    messages = [
        {"role": "user", "content": f"Translate this German text to French:\n{user_input}"}
    ]
    
    # Use the correct chat template method
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Updated generation parameters
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode without special tokens
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    st.subheader("Translation:")
    st.write(response)
