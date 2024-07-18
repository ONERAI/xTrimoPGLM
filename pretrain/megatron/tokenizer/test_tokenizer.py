
from typing import Sequence, Tuple, List, Union
from abc import ABC
from abc import abstractmethod
import logging
import itertools
import re

logger = logging.getLogger(__name__)


class ResidueLevelTokenizer(object):
    """
    Tokenizer for Protein Residue Level Tokenization.
    """

    def __init__(self, **kwargs):
        super(ResidueLevelTokenizer, self).__init__()
        self.pad_tok = ['[pad]']
        self._tokens = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
        self.num_residue_tokens = len(self._tokens) + 1
        # maxlen = 66
        # 'pad', 'eos'
        self._special_tokens = ['MASK', 'gMASK', 'sMASK', 'eod', 'sop', 'eop', '</s>']
        self.set_special_tokens(self._special_tokens)

        self.all_toks = self.pad_tok + self._tokens + self._special_tokens  
        self._vocab = {t: i for i, t in enumerate(self.all_toks)}
        logger.info('Building vocab.: {}'.format(self._vocab))

        # self.special_tokens['eos'] = self.special_tokens['</s>']
        # self.special_tokens_decoder[self.special_tokens['eos']] = '</s>'
        
        
        self.command_token = {'[MASK]':'MASK', '[gMASK]': 'gMASK', 'eos':'</s>'}


    
    def set_special_tokens(self, special_tokens):
        """Add a list of additional tokens to the encoder.
        The additional tokens are indexed starting from the last index of the
        current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, self.num_residue_tokens + i) for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        logger.info("Special tokens {}".format(self.special_tokens))

    def __len__(self):
        return len(self._vocab)

    def get_special_token(self, token):
        return self.special_tokens[token]

    def EncodeAsIds(self, text, process_fn=None):
        """convert sequence to idx"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
            processed_text = str(processed_text)
        tokens = [self.TokenToId(c) for c in processed_text]
        return tokens
    
    def IdToToken(self, idx):
        if idx == 0:
            return '[pad]'
        elif idx in self.special_tokens_decoder:
            return f"[{self.special_tokens_decoder[idx]}]"
        else:
            return self.all_toks[idx]

    def TokenToId(self, token):
        if token == '[pad]':
            return 0
        elif token in self.special_tokens:
            return self.special_tokens[token]
        else:
            return self._vocab[token]
    
    def DecodeIds(self, Ids):
        return ''.join([self.IdToToken(tok) for tok in Ids])
    
    def _tokenize(self, text) -> str:
        return text.split()
    
    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.all_toks
                        else [token]
                        for token in tokenized_text
                    )
                )
            )
        no_split_token = self.all_toks
        tokenized_text = split_on_tokens(no_split_token, text)
        # print(tokenized_text)
        return self.convert_tokens_to_ids(tokenized_text)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        # print_rank_0(tokens)
        # print_rank_0(self.vocab)
        for token in tokens:
            ids.append(self._vocab[token])
        return ids
    def get_command(self, tok):
        return self.command_token[tok]




def fill_blanks(raw_text, tokenizer):
    # add MASK
    generation_mask = "[gMASK]"
    if "[MASK]" in raw_text:
        generation_mask = "[MASK]"
    elif "[sMASK]" in raw_text:
        generation_mask = "[sMASK]"
    use_gmask = "[MASK]" not in raw_text and "[sMASK]" not in raw_text

    mask_pattern = r"\[[sg]?MASK\]"
    text_list = re.split(mask_pattern, raw_text)
    print(f"1: {text_list}")
    pattern_list = re.compile(mask_pattern).findall(raw_text)
    print(f"2: {pattern_list}")
    seq = []
    for i in range(len(pattern_list)):
        pattern = pattern_list[i]
        sub_text = text_list[i]
        seq.extend(tokenizer.tokenize(sub_text))
        seq.append(tokenizer.get_command(pattern))

    seq.extend(tokenizer.tokenize(text_list[-1]))

    if "MASK]" not in raw_text:
        seq += [tokenizer.get_command(generation_mask)]
        raw_text += " " + generation_mask
    if not raw_text.endswith("MASK]"):
        seq = seq + [tokenizer.get_command("eos")]
    print("\nInput: {}\n".format(seq))
    
if __name__ == "__main__":
    proc = ResidueLevelTokenizer()
    print(proc._vocab)
    print(proc.special_tokens)
    print(proc.all_toks)
    seqs = 'LAG[MASK]LAG[MASK]IDP[gMASK]'
    # seqs = ["L","[MASK]","L","[gMASK]",'I']
    # print(seqs.split())
    # toks = '123'
    # print(proc.DecodeIds(seqs))
    # from icetk import icetk
    # proc = icetk
    fill_blanks(seqs, proc)
    # print(proc.encode(seqs))