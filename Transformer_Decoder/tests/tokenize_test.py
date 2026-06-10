import mayatok as bpe

my_tokenizer =  bpe.get_tokenizer("v2-100k") # or 'mayatok-base' if you wish to use v1 tokenizer
old_tok = bpe.PyBPETokenizer.load(r"C:\Users\Ashwin Rajhans\webdev\MAYA\bpe_tokenizer_py.json")

text = "Hello, world!"

ids_old = old_tok.encode(text)

print("Old decode:", old_tok.decode(ids_old))
print("New decode:", my_tokenizer.decode(ids_old))