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
import logging
import os

# Configure environment first
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

st.title('🇩🇪→🇫🇷 Translator')

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        logger.info("Attempting model load...")
        
        # Use a smaller model for testing
        model_name = "Helsinki-NLP/opus-mt-de-fr"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        
        logger.info(f"Model loaded on device: {model.device}")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"CRITICAL LOAD ERROR: {str(e)}")
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

try:
    tokenizer, model = load_model()
except Exception:
    st.stop()

def translate(text):
    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"Error: {str(e)}"

user_input = st.text_input('Enter German text:')
if st.button("Translate") and user_input:
    with st.spinner("Translating..."):
        result = translate(user_input)
        st.subheader("Result:")
        st.code(result)
