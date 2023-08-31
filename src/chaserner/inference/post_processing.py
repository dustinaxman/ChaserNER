# import torch
# from transformers import BertTokenizerFast
# import json
# from chaserner.model import NERModel
# from chaserner.utils import model_output_to_label_tensor, extract_entities



#
# offsets = tokenized_data['offset_mapping'][0]
#
# # Initialize token labels with a special label (e.g., -100 which is ignored by Hugging Face's models)
# token_labels = [-100] * self.max_length
#
# label_idx = 0
# for token_idx, (start, end) in enumerate(offsets):
#     # Check if this token corresponds to a new word
#     if start == 0 and label_idx < len(ner_label_strings)  and end != 0:
#         token_labels[token_idx] = self.label_to_id[ner_label_strings[label_idx]]
#         label_idx += 1

#only take the labels for the non subwords! exactly condition above




# # tokenized_output = tokenizer("ChatGPT is developed by OpenAI.", return_tensors="pt")
# tokens = tokenizer.convert_ids_to_tokens(tokenized_output["input_ids"][0])
# merged_tokens = merge_subwords(tokens)



# tokenized_data = self.tokenizer(
#             text.split(),
#             padding='max_length',
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors='pt',
#             is_split_into_words=True,
#             return_offsets_mapping=True
#         )



