from abc import ABC
from abc import abstractmethod
from .tokenizer import AbstractTokenizer
import logging

logger = logging.getLogger(__name__)


class _IceTokenizer(AbstractTokenizer):
    """Hardcoded tokenizer."""

    def __init__(self):
        name = "IceTokenizer"
        super().__init__(name)

        self.tokenizer = None
        try:
            from icetk import icetk

            self.tokenizer = icetk
        except ImportError:
            pass
        self.num_tokens = 150000
        max_len = 80
        self.set_special_tokens(['MASK', 'gMASK', 'sMASK', 'eod', 'sop', 'eop', 'ENC', 'dBLOCK'] + ['<t>'] + [f'<blank_{i}>' for i in range(2, max_len+1)])

        self.special_tokens['eos'] = 20002
        self.special_tokens_decoder[20002] = '</s>'

    def set_special_tokens(self, special_tokens):
        """Add a list of additional tokens to the encoder.
        The additional tokens are indexed starting from the last index of the
        current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, self.num_tokens + i) for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        # for k, v in self.special_tokens.items():
        #     self.tokenizer.decoder[v] = "\u0120" + k
        logger.info("Special tokens {}".format(self.special_tokens))

    def get_special_token(self, token):
        return self.special_tokens[token]

    def IdToToken(self, idx):
        if idx == 0:
            return '[pad]'
        elif idx in self.special_tokens_decoder:
            return f"[{self.special_tokens_decoder[idx]}]"
        else:
            return self.tokenizer.decode([idx])

    def TokenToId(self, token):
        if token == '[pad]':
            return 0
        elif token in self.special_tokens:
            return self.special_tokens[token]
        else:
            return self.tokenizer.encode(token)[0]

    @property
    def vocab_size(self):
        return self.num_tokens + len(self.special_tokens)

    @property
    def vocab(self):
        assert False
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        assert False
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return "".join([self.IdToToken(t) for t in token_ids])
        # return self.tokenizer.decode(token_ids)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.get_special_token('eos')
