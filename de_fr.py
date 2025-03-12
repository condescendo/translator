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
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title('📚 German to French Translator')

@st.cache_resource(show_spinner="Loading translation model...")
def load_model():
    try:
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        logger.info(f"Loading model: {model_name}")
        
        # Force CPU if needed
        if not torch.cuda.is_available():
            logger.warning("Using CPU for inference")
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            model = model.to('cpu')
            
        logger.info("Model loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        raise

try:
    tokenizer, model = load_model()
except Exception:
    st.stop()

user_input = st.text_area('Enter German Text:', placeholder="Type your German text here...")
button = st.button("Translate")

if button and user_input:
    with st.spinner("Translating..."):
        try:
            # Phi-3 specific prompt format
            prompt = f"<|user|>\nTranslate this German text to French:\n{user_input}\n<|assistant|>\n"
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=False
            ).to(model.device)

            # Generation parameters adjusted for reliability
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # Reduced further for stability
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated = response.split("<|assistant|>")[-1].strip()
            
            st.subheader("Translation:")
            st.write(translated)

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            st.error(f"Translation failed: {str(e)}")
