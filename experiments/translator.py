from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
article_en = "The head of the United Nations says there is no military solution in Syria"
model = MBartForConditionalGeneration.from_pretrained("../../llama2/models/Llama2-13b-Language-translate")
tokenizer = MBart50TokenizerFast.from_pretrained("../../llama2/models/Llama2-13b-Language-translate", src_lang="en_XX")

model_inputs = tokenizer(article_en, return_tensors="pt")

# translate from English to Hindi
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => 'संयुक्त राष्ट्र के नेता कहते हैं कि सीरिया में कोई सैन्य समाधान नहीं है'

# translate from English to Chinese
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
)
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
# => '联合国首脑说,叙利亚没有军事解决办法'
