import re
from simpletokenizer import SimpleTokenizerV1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
		raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)


vocab = {token:integer for integer,token in enumerate(all_words)}

print(vocab)


tokenizer = SimpleTokenizerV1(vocab)
text = "hello who is your father,my friend"
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))