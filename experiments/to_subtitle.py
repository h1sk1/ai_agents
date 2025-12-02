# Read json from json file
import json

# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
#
# article_en = "The head of the United Nations says there is no military solution in Syria"
# model = MBartForConditionalGeneration.from_pretrained("../llama2/models/Llama2-13b-Language-translate")
# tokenizer = MBart50TokenizerFast.from_pretrained("../llama2/models/Llama2-13b-Language-translate", src_lang="en_XX")

SYSTEM_PROMPT = "Translate English to Chinese, do not generate any other unrelated information, every phase is related, under same context: \n\n"

with open("../data/西方的学校也扼杀创造.json", "r") as f:
    result = json.load(f)

    segments = result["segments"]

    all_text = [[]]
    index = 0
    inside_index = 0
    texts = SYSTEM_PROMPT
    for segment in segments:
        if segment["text"] != "":
            index += 1
            text = segment["text"]
            texts += f"{index}. {text}\n"
            # if index == 0 and inside_index == 0:
            #     all_text[index].append(SYSTEM_PROMPT)
            # if inside_index >= 10:
            #     inside_index = 0
            #     index += 1
            #     all_text.append([])
            #     all_text[index].append(SYSTEM_PROMPT)
            # inside_index += 1
            # all_text[index].append(f"{inside_index}. {text}\n")

print(texts)

# for text in all_text:
#     prompt = "".join(text)
    # print(prompt)
    # model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    # generated_tokens = model.generate(
    #     **model_inputs,
    #     forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
    # )
    # translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    # chinese = "".join(translation)
    # print(chinese)
# model_inputs = tokenizer(all_text, return_tensors="pt")
# # 获取 input_ids 张量的形状
# input_ids = model_inputs["input_ids"]
# batch_size, sequence_length = input_ids.shape
#
# print(f"Batch size: {batch_size}")
# print(f"Number of tokens in each sequence: {sequence_length}")
#
# generated_tokens = model.generate(
#     **model_inputs,
#     forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
# )
# translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# # Convert str list to str
# chinese = "".join(translation)
# print(chinese)
