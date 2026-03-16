import mayatok_bpe as bpe



import mayatok_bpe as bpe

my_tokenizer = bpe.PyBPETokenizer.load("bpe_tokenizer_py.json")
test = "Hello, world!"
tokens = my_tokenizer.encode(test)
print(tokens)
decoded_text = my_tokenizer.decode(tokens)
print(decoded_text)