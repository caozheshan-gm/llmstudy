import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "how many strawberry"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
print(tokenizer.decode([4919, 867,301,1831,8396]))