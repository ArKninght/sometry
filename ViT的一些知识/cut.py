# 将文字裁剪为token
test_text="这是一段示例文字"
print(test_text)
import re
test_text_arr = re.split(r'(\s)', test_text)
print(test_text_arr)
##进行TOKEN化
test_text_arr = re.split(r'([,.，。\w]|\s)', test_text)
print(test_text_arr)
 
preprocessed = [item.strip() for item in test_text_arr if item.strip()]
print(preprocessed)
##将拆分后的单词进行去重
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
    
print(vocab_size)
##为每一个Token生成一个唯一ID
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)