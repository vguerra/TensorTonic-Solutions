class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.id_to_word = {id:word for id, word in enumerate(
            [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        )}
        all_words = set([word for sentence in texts for word in sentence.lower().split()])
        self.id_to_word.update(
            {id + 4: word for id, word in enumerate(sorted(all_words))}
        )
        self.word_to_id = {val:key for key, val in self.id_to_word.items()}
        self.vocab_size = len(self.word_to_id)
        print(self.id_to_word)
        print(self.word_to_id)
        print(self.vocab_size)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
                for word in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join([
            self.id_to_word.get(id, self.unk_token) for id in ids])