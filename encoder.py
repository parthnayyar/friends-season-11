class Encoder:
    def __init__(self, vocabulary):
        self.vocab = vocabulary
        self.token_to_id = {token: i for i, token in enumerate(vocabulary)}
        self.id_to_token = {i: token for i, token in enumerate(vocabulary)}
        self.__max_token_len = 0
        for token in self.vocab:
            if len(token) > self.__max_token_len: self.__max_token_len = len(token)

    def encode(self, text):
        encoded_tokens = []
        i = 0
        while i < len(text):
            for j in range(i+self.__max_token_len, i, -1): # maximal munch encoding
                substr = text[i:j]
                if substr in self.token_to_id:
                    encoded_tokens.append(self.token_to_id[substr])
                    i = j
                    break
            else: raise ValueError(f'Unrecognized token starting at position {i} in text: "{text[i:]}"')
        return encoded_tokens
    
    def decode(self, encoding):
        text = ''
        for token in encoding: text += self.id_to_token[token]
        return text
    