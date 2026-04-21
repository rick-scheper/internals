from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os


if not os.path.exists("data/tokenizer-wiki.json"):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    files = [os.path.join(os.getcwd(),f"data/wikitext-103-{split}.txt") for split in ["test", "train", "validation"]]
    tokenizer.train(files, trainer)
    tokenizer.save("data/tokenizer-wiki.json")
else:
    tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")


output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
