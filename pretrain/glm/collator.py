import random
import copy
import numpy as np
from scipy.stats import poisson
from scipy.linalg import block_diag


class GLMPreprocessor:
    def __init__(
            self,
            tokenizer,
            eod_id,
            tmask_id,
            smask_id,
            gmask_id,
            sop_id,
            eop_id,
            max_seq_length,
            span_seq_length,
            aggregated_samples_per_sequence,
            bert_prob,
            span_prob,
            short_seq_prob,
            single_span_prob,
            mask_ratio,
            average_block_length,
            min_gmask_ratio,
            relative_pos_encoding,
            no_2d_encoding,
            aggregate_gpt_sample,
            adaptive_multitask_encoding,
            adaptive_multitask_encoding_length,
            unified_multitask_encoding,
            rank,
            device_num,
    ):
        self.tokenizer = tokenizer
        self.eod_id = eod_id
        self.smask_id = smask_id
        self.tmask_id = tmask_id
        self.gmask_id = gmask_id
        self.sop_id = sop_id
        self.eop_id = eop_id
        self.eos_id = 34
        self.max_seq_length = max_seq_length
        self.span_seq_length = span_seq_length
        self.aggregated_samples_per_sequence = aggregated_samples_per_sequence
        self.bert_prob = bert_prob
        self.span_prob = span_prob
        self.gpt_prob = 1 - bert_prob - span_prob
        self.short_seq_prob = short_seq_prob
        self.single_span_prob = single_span_prob
        self.mask_ratio = mask_ratio
        self.average_block_length = average_block_length
        self.min_gmask_ratio = min_gmask_ratio
        self.block_length_distribution = [
            poisson.pmf(i, average_block_length) for i in range(1, 40)
        ]
        self.span_length_distribution = [
            poisson.pmf(i, average_block_length) for i in range(2, 11)
        ]
        self.relative_pos_encoding = relative_pos_encoding
        self.no_2d_encoding = no_2d_encoding
        self.aggregate_gpt_sample = aggregate_gpt_sample
        self.adaptive_multitask_encoding = adaptive_multitask_encoding
        self.adaptive_length_distribution = 1 - np.array([
            poisson.cdf(i, adaptive_multitask_encoding_length) for i in range(1, 40)
        ])
        self.unified_multitask_encoding = unified_multitask_encoding
        self.count = 0
        self.rank = rank
        self.device_num = device_num

    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        if '。' in tok:
            return True
        if '？' in tok:
            return True
        if '！' in tok:
            return True
        if '；' in tok:
            return True
        if '…' in tok:
            return True
        if '\n' in tok:
            return True
        return False

    def truncate_input(self, input_ids, rng):
        target_length = rng.randrange(32, len(input_ids))
        return input_ids[:target_length]

    def truncate_span_input(self, input_ids, rng):
        # if len(input_ids)
        remains = max(0, len(input_ids) - self.span_seq_length)
        starts_idx = rng.randrange(0, remains+1)
        return input_ids[starts_idx:starts_idx+self.span_seq_length]

    @staticmethod
    def build_mask_matrix(separator, seq_length, memory_length=0):
        dtype = np.int64
        m = np.ones((seq_length, seq_length), dtype=dtype)
        m = np.tril(m)
        m[:, :separator] = 1
        if memory_length > 0:
            m = np.concatenate(
                (np.ones((seq_length, memory_length), dtype=dtype), m), dim=2
            )
        m = m[np.newaxis, :, :]
        return m

    @staticmethod
    def sample_spans(span_lengths, total_length, rng, offset=0):
        blank_length = total_length - sum(span_lengths)
        m = blank_length - len(span_lengths) + 1
        places = [rng.randrange(m + 1) for _ in range(len(span_lengths))]
        places.sort()
        spans = []
        for place, span_length in zip(places, span_lengths):
            start = offset + place
            end = offset + place + span_length
            spans.append((start, end))
            offset += span_length + 1
        return spans

    def make_block_data(self, input_ids, block_spans, rng, task="bert"):
        position_ids = np.ones(len(input_ids), dtype=int)
        for start, end in block_spans:
            position_ids[start + 1: end] = 0
        position_ids = np.cumsum(position_ids) - 1
        rng.shuffle(block_spans)
        block_spans = [(start, end) for start, end in block_spans]
        (
            target_tokens,
            target_position_ids,
            target_block_position_ids,
            targets,
        ) = ([], [], [], [])
        for start, end in block_spans:
            target_tokens.append([self.sop_id])
            span_tokens = copy.deepcopy(input_ids[start:end])
            target_tokens.append(span_tokens)
            targets.append(input_ids[start:end])
            targets.append([self.eop_id])
            target_position_id = position_ids[start:end]
            target_position_ids.append(target_position_id)
            target_position_ids.append([target_position_id[0]])
            target_block_position_ids.append(
                np.arange(1, end - start + 2, dtype=int)
            )
        block_spans.sort(key=lambda x: x[0])
        source_tokens, source_position_ids, local_spans = [], [], []
        last, current_length = 0, 0
        for start, end in block_spans:
            # if task == "generation":
            #     mask_id = self.gmask_id
            # elif task == "bert":
            #     mask_id = self.tmask_id
            # else:
            mask_id = self.smask_id
            local_spans.append((current_length, current_length + start - last))
            source_tokens.append(input_ids[last:start])
            source_tokens.append([mask_id])
            source_position_ids.append(position_ids[last:start])
            source_position_ids.append([position_ids[start]])
            current_length += start - last + 1
            last = end
        if last < len(input_ids):
            local_spans.append(
                (current_length, current_length + len(input_ids) - last)
            )
            source_tokens.append(input_ids[last:])
            source_position_ids.append(position_ids[last:])
        source_length = sum(map(len, source_tokens))
        tokens = np.concatenate(source_tokens + target_tokens)
        targets = np.concatenate(source_tokens + targets)
        loss_masks = np.ones(len(tokens), dtype=int)
        loss_masks[:source_length] = 0
        position_ids = np.concatenate(source_position_ids + target_position_ids)
        block_position_ids = np.concatenate(
            [np.zeros(source_length, dtype=int)] + target_block_position_ids
        )
        position_ids = np.stack([position_ids, block_position_ids], axis=0)
        return tokens, targets, loss_masks, position_ids, source_length

    def generate_blank_data(self, input_ids, masked_lengths, rng, task="bert"):
        rng.shuffle(masked_lengths)
        block_spans = self.sample_spans(masked_lengths, len(input_ids), rng)
        if len(block_spans) < len(masked_lengths):
            return None
        data = self.make_block_data(input_ids, block_spans, rng, task=task)
        return data

    def pad_batch(self, tokens, targets, loss_masks, position_ids, max_seq_length=None):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if len(tokens) >= max_seq_length:
            tokens = tokens[: max_seq_length]
            targets = targets[: max_seq_length]
            loss_masks = loss_masks[: max_seq_length]
            position_ids = position_ids[:, : max_seq_length]
        else:
            tokens = np.concatenate(
                (
                    tokens,
                    np.zeros(max_seq_length - len(tokens), dtype=int),
                )
            )
            targets = np.concatenate(
                (
                    targets,
                    np.zeros(max_seq_length - len(targets), dtype=int),
                )
            )
            loss_masks = np.concatenate(
                (
                    loss_masks,
                    np.zeros(
                        max_seq_length - len(loss_masks), dtype=int
                    ),
                )
            )
            position_ids = np.concatenate(
                (
                    position_ids,
                    np.zeros(
                        (2, max_seq_length - position_ids.shape[1]),
                        dtype=int,
                    ),
                ),
                axis=1,
            )
        return tokens, targets, loss_masks, position_ids

    def _build_relative_pos_encoding(self, position_ids, division):
        position_ids, block_position_ids = position_ids[0], position_ids[1]
        seq_length = position_ids.shape[0]
        relative_pos = np.zeros((seq_length, seq_length), dtype=np.float16)
        relative_pos[:, :division] = -np.abs(position_ids[:, None] - position_ids[None, :division])
        arange = np.arange(seq_length - division)
        relative_pos[division:, division:] = np.tril(arange[None, :] - arange[:, None])
        return relative_pos

    def _pack_samples(self, sequences):
        tokens, targets, loss_masks, position_ids, division = zip(*sequences)
        tokens = np.concatenate(tokens, axis=-1)
        targets = np.concatenate(targets, axis=-1)
        loss_masks = np.concatenate(loss_masks, axis=-1)
        if self.relative_pos_encoding:
            position_ids = block_diag(*position_ids)
        else:
            position_ids = np.concatenate(position_ids, axis=-1)
        division = list(division)
        division.extend([np.array([0]) for _ in range(2-len(division))])
        # division.extend([np.array([0]) for _ in range(8-len(division))])
        division = np.concatenate(division, axis=-1)
        return tokens, targets, loss_masks, position_ids, division

    def get_input_data(self, input_ids, index=None):
        if index is None:
            rng = random.Random(self.count * self.device_num + self.rank)
        else:
            rng = random.Random(random.Random(index).randint(0, 2 ** 32 - 1))
        self.count += 1
        # found_sentence_end = False
        # for tok in input_ids:
        #     if self.contains_sentence_end(tok):
        #         found_sentence_end = True
        #         break
        aggregated_samples = rng.choices(
               # [1,2,4,8],
                #weights=[0.1, 0.4, 0.4, 0.1],
                [1],
                weights=[1.0],
            )[0]
        # input_ids = self.truncate_span_input(input_ids, rng)
        assert self.max_seq_length % aggregated_samples == 0
        assert len(input_ids) % aggregated_samples == 0
        input_length = len(input_ids) // aggregated_samples
        task_rand = rng.random()
        # print(f"task_rand: {task_rand} bert: {self.bert_prob}/{self.mask_ratio} span: {self.span_prob} gpt: {self.gpt_prob}")
        if task_rand < self.bert_prob:
            sequences = []
            # input_length = len(input_ids)
            # current_input_ids = input_ids
            for i in range(aggregated_samples):
                current_input_ids = input_ids[input_length * i: input_length * (i + 1)]     
                if rng.random() < self.short_seq_prob:
                    current_input_ids = self.truncate_input(current_input_ids, rng)
                
                # # for special spans
                # current_input_ids = [idx % 100 for idx in current_input_ids]
                
                masked_lengths = int(len(current_input_ids) * self.mask_ratio)
                cand_maked_pos = [idx for idx, token in enumerate(current_input_ids)]
                rng.shuffle(cand_maked_pos)

                target_tokens = copy.deepcopy(current_input_ids)
                source_tokens = copy.deepcopy(current_input_ids)
                loss_masks = np.zeros(len(target_tokens), dtype=int)
                position_ids = np.arange(len(target_tokens), dtype=int)
                for pos in cand_maked_pos[:masked_lengths]:
                    loss_masks[pos] = 1
                    if rng.random() < 0.8:  # 80%
                        source_tokens[pos] = self.tmask_id # make token mask
                    elif rng.random() < 0.5:  # 10%
                        index = rng.randint(1, len(self.tokenizer.normal_vocab)) # random index in vocabulary
                        # print(pos, index)
                        source_tokens[pos] = index # replace
                block_position_ids = np.concatenate(
                    [np.zeros(len(target_tokens), dtype=int)]
                )
                
                position_ids = np.stack([position_ids, block_position_ids], axis=0)
                # breakpoint()
                tokens, targets, loss_masks, position_ids = self.pad_batch(
                    source_tokens, target_tokens, loss_masks, position_ids,
                    max_seq_length=self.max_seq_length // aggregated_samples
                )
                
                division = len(target_tokens)
                if self.relative_pos_encoding:
                    position_ids = self._build_relative_pos_encoding(position_ids, division)
                elif self.no_2d_encoding:
                    position_ids = position_ids[0]
                division = np.array([division], dtype=int)
                # attention_mask = np.zeros([1, len(tokens), len(tokens)])
                # last = 0
                # for i in range(len(target_tokens)):
                #     if target_tokens[i] == self.eos_id:
                #         attention_mask[:, last: i + 1, last: i + 1] = 1
                #         last = i + 1
                # attention_mask[:, last:, last:] = 1
                # attention_mask = attention_mask < 0.5
                sequences.append((tokens, targets, loss_masks, position_ids, division))
                # breakpoint()
            return *self._pack_samples(sequences), 0
        
        
        
        # if task_rand < self.bert_prob:
        #     sequences = []
        #     # input_length = len(input_ids)
        #     # current_input_ids = input_ids
        #     for i in range(aggregated_samples):
        #         current_input_ids = input_ids[input_length * i: input_length * (i + 1)]     
        #         if rng.random() < self.short_seq_prob:
        #             current_input_ids = self.truncate_input(current_input_ids, rng)
        #         # # for special spans
        #         # current_input_ids = [idx % 100 for idx in current_input_ids]
                
        #         # masked_lengths = int(len(current_input_ids) * self.mask_ratio)
        #         # cand_maked_pos = [i for i, token in enumerate(current_input_ids)]
        #         masked_lengths, masked_count = [], 0
        #         while masked_count < int(self.mask_ratio * len(current_input_ids)):
        #             if rng.random() < 0.6:
        #                 masked_lengths.append(1)
        #                 masked_count += 1
        #             else:
        #                 block_length = rng.choices(
        #                     range(2, len(self.span_length_distribution) + 2),
        #                     weights=self.span_length_distribution,
        #                 )[0]
        #                 masked_lengths.append(block_length)
        #                 masked_count += block_length
        #                 # breakpoint()
        #         block_spans = self.sample_spans(masked_lengths, len(current_input_ids), rng)
        #         # breakpoint()
        #         rng.shuffle(block_spans)
                
        #         target_tokens = copy.deepcopy(current_input_ids)
        #         source_tokens = copy.deepcopy(current_input_ids)
        #         loss_masks = np.zeros(len(target_tokens), dtype=int)
        #         position_ids = np.arange(len(target_tokens), dtype=int)
        #         # breakpoint()
        #         for start, end in block_spans:
        #             loss_masks[start:end] = 1
        #             if rng.random() < 0.8:  # 80%
        #                 # print('mask')
        #                 source_tokens[start:end] = self.tmask_id # make token mask
        #             elif rng.random() < 0.5:  # 10%
        #                 # print('replace')
        #                 for replace_index in range(start, end):
        #                     index = rng.randint(1, len(self.tokenizer.normal_vocab)) # random index in vocabulary
        #                     source_tokens[replace_index] = index # replace
        #             # breakpoint()
        #         block_position_ids = np.concatenate(
        #             [np.zeros(len(target_tokens), dtype=int)]
        #         )
                
        #         position_ids = np.stack([position_ids, block_position_ids], axis=0)
        #         # breakpoint()
        #         tokens, targets, loss_masks, position_ids = self.pad_batch(
        #             source_tokens, target_tokens, loss_masks, position_ids,
        #             max_seq_length=self.max_seq_length // aggregated_samples
        #         )
                
        #         division = len(target_tokens)
        #         if self.relative_pos_encoding:
        #             position_ids = self._build_relative_pos_encoding(position_ids, division)
        #         elif self.no_2d_encoding:
        #             position_ids = position_ids[0]
        #         division = np.array([division], dtype=int)
        #         sequences.append((tokens, targets, loss_masks, position_ids, division))
        #         # breakpoint()
        #     return *self._pack_samples(sequences), 0

        elif task_rand < self.bert_prob + self.span_prob:
            sequences = []

            # current_input_ids = input_ids
            for i in range(aggregated_samples):
                current_input_ids = input_ids[input_length * i: input_length * (i + 1)]            
                if rng.random() < self.short_seq_prob:
                    current_input_ids = self.truncate_input(current_input_ids, rng)
                single_span = rng.random() < self.single_span_prob
                if single_span:
                    masked_lengths = [
                        rng.choices(
                            range(1, len(self.block_length_distribution) + 1),
                            weights=self.block_length_distribution,
                        )[0]
                    ]
                else:
                    masked_lengths, masked_count = [], 0
                    while masked_count < int(self.mask_ratio * len(current_input_ids)):
                        block_length = rng.choices(
                            range(1, len(self.block_length_distribution) + 1),
                            weights=self.block_length_distribution,
                        )[0]
                        masked_lengths.append(block_length)
                        masked_count += block_length
                tokens, targets, loss_masks, position_ids, division = self.generate_blank_data(
                    current_input_ids, masked_lengths, rng, task="span"
                )
                tokens, targets, loss_masks, position_ids = self.pad_batch(
                    tokens, targets, loss_masks, position_ids,
                    max_seq_length=self.max_seq_length // aggregated_samples
                )
                if self.relative_pos_encoding:
                    position_ids = self._build_relative_pos_encoding(position_ids, division)
                elif self.no_2d_encoding:
                    position_ids = position_ids[0]
                division = np.array([division], dtype=int)
                sequences.append((tokens, targets, loss_masks, position_ids, division))
            return *self._pack_samples(sequences), 1            
        else:
            sequences = []
            # input_length = len(input_ids)
            # aggregated_samples_per_sequence = 1
            # for i in range(aggregated_samples_per_sequence):
            # current_input_ids = input_ids
            # if rng.random() < 0.5:
            #    current_input_ids = current_input_ids[::-1]
            for i in range(aggregated_samples):
                current_input_ids = input_ids[input_length * i: input_length * (i + 1)] 
                if rng.random() < self.short_seq_prob:
                    current_input_ids = self.truncate_input(current_input_ids, rng)
                generation_length = rng.randint(
                    int(self.min_gmask_ratio * len(current_input_ids)), len(current_input_ids)
                )
                division = len(current_input_ids) - generation_length
                source_tokens, target_tokens = (
                    current_input_ids[:division],
                    current_input_ids[division:],
                )
                target_masks = np.ones(len(target_tokens), dtype=int)
                tokens = np.concatenate(
                    (
                        source_tokens,
                        [self.gmask_id, self.sop_id],
                        target_tokens[:-1],
                    )
                )
                targets = np.concatenate(
                    (source_tokens, [self.gmask_id], target_tokens)
                )
                loss_masks = np.concatenate(
                    (np.zeros(len(source_tokens) + 1, dtype=int), target_masks)
                )
                position_ids = np.arange(
                    len(source_tokens) + len(target_tokens) + 1, dtype=int
                )
                position_ids[len(source_tokens) + 1:] = len(source_tokens)
                block_position_ids = np.concatenate(
                    (
                        np.zeros(len(source_tokens), dtype=int),
                        np.arange(len(target_tokens) + 1, dtype=int),
                    )
                )
                position_ids = np.stack([position_ids, block_position_ids], axis=0)
                division = division + 1
                tokens, targets, loss_masks, position_ids = self.pad_batch(
                    tokens, targets, loss_masks, position_ids,
                    max_seq_length=self.max_seq_length // aggregated_samples
                )
                if self.relative_pos_encoding:
                    position_ids = self._build_relative_pos_encoding(position_ids, division)
                elif self.no_2d_encoding:
                    position_ids = np.arange(len(tokens), dtype=int)
                # attention_mask = self.build_mask_matrix(division, self.max_seq_length)
                division = np.array([division], dtype=int)
                sequences.append((tokens, targets, loss_masks, position_ids, division))
            return *self._pack_samples(sequences), 2

    def _get_single_multitask_data(self, text, target, max_seq_length):
        if len(text) + len(target) + 2 > max_seq_length:
            text_length = max(int(0.25 * max_seq_length), max_seq_length - len(target) - 2)
            text = text[:text_length]
        if len(text) + len(target) + 2 > max_seq_length:
            target = target[:max_seq_length - len(text) - 2]
        dtype = text.dtype
        if self.mask_id in text:
            assert self.unified_multitask_encoding
            mask_position = np.where(self.mask_id)[0][0]
            tokens = np.concatenate((text, [self.sop_id], target))
            targets = np.concatenate((text, target, [self.eop_id]))
            loss_masks = np.concatenate((np.zeros(len(text), dtype=dtype), np.ones(len(target) + 1, dtype=dtype)))
            position_ids = np.arange(len(tokens), dtype=dtype)
            position_ids[len(text):] = mask_position
            position_ids = np.stack([position_ids, position_ids])
            division = len(text)
            tokens, targets, loss_masks, position_ids = self.pad_batch(tokens, targets, loss_masks, position_ids,
                                                                       max_seq_length=max_seq_length)
            return tokens, targets, loss_masks, position_ids[0], np.array([division], dtype=dtype)
        tokens = np.concatenate((text, [self.mask_id, self.sop_id], target))
        targets = np.concatenate((text, [self.mask_id], target, [self.eop_id]))
        loss_masks = np.concatenate((np.zeros(len(text) + 1, dtype=dtype), np.ones(len(target) + 1, dtype=dtype)))
        position_ids = np.arange(len(tokens), dtype=dtype)
        position_ids[len(text) + 1:] = len(text)
        block_position_ids = np.concatenate((np.zeros(len(text), dtype=dtype), np.arange(len(target) + 2, dtype=dtype)))
        position_ids = np.stack([position_ids, block_position_ids])
        tokens, targets, loss_masks, position_ids = self.pad_batch(tokens, targets, loss_masks, position_ids,
                                                                   max_seq_length=max_seq_length)
        division = len(text) + 1
        if self.relative_pos_encoding:
            position_ids = self._build_relative_pos_encoding(position_ids, division)
        elif self.no_2d_encoding:
            position_ids = np.arange(len(tokens), dtype=dtype)
            if self.adaptive_multitask_encoding:
                rng = random.Random(random.Random(np.sum(tokens) + np.sum(targets)).randint(0, 2 ** 32 - 1))
                if len(target) < len(self.adaptive_length_distribution) \
                        and rng.random() < self.adaptive_length_distribution[len(target)]:
                    position_ids[len(text) + 1:] = len(text)
                else:
                    position_ids = np.concatenate((np.arange(len(text) + 1, dtype=dtype),
                                                   np.arange(len(text), len(text) + len(target) + 1, dtype=dtype)))
                    position_ids = np.concatenate((position_ids,
                                                   np.zeros(max_seq_length - len(position_ids), dtype=dtype)))
            elif self.unified_multitask_encoding:
                position_ids = np.concatenate((np.arange(len(text) + 1, dtype=dtype),
                                         np.arange(len(text), len(text) + len(target) + 1, dtype=dtype)))
                position_ids = np.concatenate((position_ids,
                                        np.zeros(max_seq_length - len(position_ids), dtype=dtype)))
        # attention_mask = self.build_mask_matrix(len(text) + 1, max_seq_length)
        return tokens, targets, loss_masks, position_ids, np.array([division], dtype=dtype)

    def get_multitask_data(self, texts, targets):
        if self.aggregated_samples_per_sequence > 1:
            assert self.max_seq_length % self.aggregated_samples_per_sequence == 0
            sequences = []
            for text, target in zip(texts, targets):
                data = self._get_single_multitask_data(text, target, self.max_seq_length // self.aggregated_samples_per_sequence)
                sequences.append(data)
            return self._pack_samples(sequences)
        else:
            return self._get_single_multitask_data(texts[0], targets[0], self.max_seq_length)

    def get_greedily_aggregated_multitask_data(self, texts, targets):
        sequences, length = [], 0
        for idx, (text, target) in enumerate(zip(texts, targets)):
            cur_length = self.max_seq_length - length if idx + 1 == len(texts) else len(text) + len(target) + 2
            tokens, targets, loss_masks, position_ids, division = \
                self._get_single_multitask_data(text, target, max_seq_length=cur_length)
            division  = np.array([division, [cur_length]], dtype=np.long)
            sequences.append((tokens, targets, loss_masks, position_ids, division))
            length += cur_length
        return self._pack_samples(sequences)



def debug_block_data(data, special_tokens):
    tokens, targets, loss_masks, position_ids, attention_mask = data
    # block_position_ids = position_ids[1]
    # position_ids_ = position_ids[0]
    sep = int(attention_mask[0, 0].sum())
    text, last_segment = "", []
    id_to_token = {idx: token for token, idx in special_tokens.items()}
    for i, token_id in enumerate(tokens[:sep].tolist()):
        if token_id in [special_tokens["mask"], special_tokens["gmask"], special_tokens["tmask"]]:
            if last_segment:
                text += " ".join(last_segment)
                last_segment = []
            text += f" [{position_ids[i]}, {id_to_token[token_id]}]"
        else:
            last_segment.append(str(token_id))
    if last_segment:
        text += " ".join(last_segment)
    print(text)
    last_index = None
    for i in range(sep, tokens.shape[0]):
        if tokens[i] == special_tokens["sop"]:
            if last_index is not None:
                print(
                    tokens[last_index:i].tolist(),
                    "|",
                    targets[last_index:i].tolist(),
                    position_ids[last_index:i].tolist(),
                    # position_ids_[last_index:i].tolist(),
                    # block_position_ids[last_index:i].tolist(),
                )
            last_index = i
    if last_index is not None:
        end_index = last_index
        for i in range(last_index, tokens.shape[0]):
            if loss_masks[i] != 0:
                end_index = i
        print(
            tokens[last_index:end_index + 1].tolist(),
            "|",
            targets[last_index:end_index + 1].tolist(),
            position_ids[last_index:end_index + 1].tolist(),
            # position_ids_[last_index:end_index + 1].tolist(),
            # block_position_ids[last_index:end_index + 1].tolist(),
        )



def main():
    from SwissArmyTransformer import get_tokenizer

    tokenizer = get_tokenizer(tokenizer_type="icetk-glm-130B")
    aggregated_samples_per_sequence = 4
    max_seq_length = 2304
    single_length = max_seq_length // aggregated_samples_per_sequence
    special_tokens = {"eod": 100000, "mask": 100001, "gmask": 100002, "smask": 100003, "sop": 100004, "eop": 100005}
    collator = GLMPreprocessor(
        tokenizer=tokenizer,
        eod_id=special_tokens["eod"],
        mask_id=special_tokens["mask"],
        gmask_id=special_tokens["gmask"],
        smask_id=special_tokens["smask"],
        sop_id=special_tokens["sop"],
        eop_id=special_tokens["eop"],
        max_seq_length=max_seq_length,
        aggregated_samples_per_sequence=aggregated_samples_per_sequence,
        sent_prob=0.5,
        gpt_prob=0.2,
        short_seq_prob=0.02,
        single_span_prob=0.02,
        mask_ratio=0.15,
        average_block_length=3,
        min_gmask_ratio=0.2,
        relative_pos_encoding=False,
        no_2d_encoding=True,
        rank=1,
        device_num=2,
        aggregate_gpt_sample=False,
        adaptive_multitask_encoding=False,
        adaptive_multitask_encoding_length=False,
        unified_multitask_encoding=False,
    )
    with open("test.txt") as file:
        text = file.read()
    input_ids = tokenizer.tokenize(text)
    # Remove sentence end
    # input_ids = [tok for tok in input_ids if not collator.contains_sentence_end(tok)]
    if len(input_ids) < 512:
        input_ids = input_ids + [0] * (512 - len(input_ids))
    else:
        input_ids = input_ids[:512]
    input_ids = np.array(input_ids * 4, dtype=int)
    for _ in range(10):
        (
            tokens_,
            targets_,
            loss_masks_,
            position_ids_,
            attention_mask_,
            task_type
        ) = collator.get_input_data(input_ids)
        if len(attention_mask_) > 1:
            for i in range(aggregated_samples_per_sequence):
                debug_block_data(
                    (tokens_[i * single_length: (i + 1) * single_length],
                     targets_[i * single_length: (i + 1) * single_length],
                     loss_masks_[i * single_length: (i + 1) * single_length],
                     position_ids_[i * single_length: (i + 1) * single_length], collator.build_mask_matrix(
                        attention_mask_[i], single_length)),
                    special_tokens
                )
        else:
            debug_block_data(
                (tokens_, targets_, loss_masks_, position_ids_,
                 collator.build_mask_matrix(attention_mask_[0], max_seq_length)), special_tokens)
        breakpoint()
        print()
    texts, targets = [np.arange(256), np.arange(256, 512), np.arange(512, 768), np.arange(768, 1024)], [
        np.arange(1024, 1024 + 64), np.arange(1024 + 64, 1024 + 128), np.arange(1024 + 128, 1024 + 192),
        np.arange(1024 + 192, 1024 + 256)]
    tokens_, targets_, loss_masks_, position_ids_, attention_mask_ = collator.get_multitask_data(texts, targets)
    for i in range(aggregated_samples_per_sequence):
        debug_block_data((tokens_[i * single_length: (i + 1) * single_length],
                          targets_[i * single_length: (i + 1) * single_length],
                          loss_masks_[i * single_length: (i + 1) * single_length],
                          position_ids_[i * single_length: (i + 1) * single_length],
                          collator.build_mask_matrix(attention_mask_[i], single_length)), special_tokens)
        print()
        breakpoint()
    text, target = np.arange(1024), np.arange(1024, 1024 + 64)
    tokens_, targets_, loss_masks_, position_ids_, attention_mask_ = collator.get_multitask_data(text, target)
    debug_block_data((tokens_, targets_, loss_masks_, position_ids_, attention_mask_), special_tokens)
    print()
    text, target = np.arange(256), np.arange(1024, 1024 + 1024)
    tokens_, targets_, loss_masks_, position_ids_, attention_mask_ = collator.get_multitask_data(text, target)
    debug_block_data((tokens_, targets_, loss_masks_, position_ids_, attention_mask_), special_tokens)


if __name__ == "__main__":
    main()



    # def get_input_data(self, input_ids, index=None):
    #     if index is None:
    #         rng = random.Random(self.count * self.device_num + self.rank)
    #     else:
    #         rng = random.Random(random.Random(index).randint(0, 2 ** 32 - 1))
    #     self.count += 1
    #     found_sentence_end = False
    #     for tok in input_ids:
    #         if self.contains_sentence_end(tok):
    #             found_sentence_end = True
    #             break
    #     task_rand = rng.random()
    #     if found_sentence_end and task_rand < self.sent_prob:
    #         sequences = []
    #         assert self.max_seq_length % self.aggregated_samples_per_sequence == 0
    #         assert len(input_ids) % self.aggregated_samples_per_sequence == 0
    #         input_length = len(input_ids) // self.aggregated_samples_per_sequence
    #         for i in range(self.aggregated_samples_per_sequence):
    #             current_input_ids = input_ids[input_length * i: input_length * (i + 1)]
    #             if rng.random() < self.short_seq_prob:
    #                 current_input_ids = self.truncate_input(current_input_ids, rng)
    #             sentence_spans = []
    #             last_index = 0
    #             for i in range(len(current_input_ids)):
    #                 if self.contains_sentence_end(current_input_ids[i]):
    #                     if last_index < i + 1:
    #                         sentence_spans.append((last_index, i + 1))
    #                     last_index = i + 1
    #                 elif current_input_ids[i] == self.eod_id:  # Sentence cannot cross document boundary
    #                     last_index = i + 1
    #             single_span = rng.random() < self.single_span_prob
    #             if single_span:
    #                 if sentence_spans:
    #                     block_spans = rng.sample(sentence_spans, 1)
    #                 else:
    #                     block_spans = []
    #             else:
    #                 rng.shuffle(sentence_spans)
    #                 block_spans, block_length = [], 0
    #                 for start, end in sentence_spans:
    #                     block_spans.append((start, end))
    #                     block_length += end - start
    #                     if block_length >= int(self.mask_ratio * len(current_input_ids)):
    #                         break
    #             tokens, targets, loss_masks, position_ids, division = self.make_block_data(
    #                 current_input_ids, block_spans, rng, task="sentence"
    #             )
    #             tokens, targets, loss_masks, position_ids = self.pad_batch(
    #                 tokens, targets, loss_masks, position_ids,
    #                 max_seq_length=self.max_seq_length // self.aggregated_samples_per_sequence
    #             )
    #             if self.relative_pos_encoding:
    #                 position_ids = self._build_relative_pos_encoding(position_ids, division)
    #             elif self.no_2d_encoding:
    #                 position_ids = position_ids[0]
    #             division = np.array([division], dtype=int)
    #             sequences.append((tokens, targets, loss_masks, position_ids, division))
    #         return *self._pack_samples(sequences), 3
    #     elif task_rand < self.bert_prob + self.sent_prob:
    #         sequences = []
    #         assert self.max_seq_length % self.aggregated_samples_per_sequence == 0
    #         assert len(input_ids) % self.aggregated_samples_per_sequence == 0
    #         input_length = len(input_ids) // self.aggregated_samples_per_sequence
    #         for i in range(self.aggregated_samples_per_sequence):
    #             current_input_ids = input_ids[input_length * i: input_length * (i + 1)]
    #             if rng.random() < self.short_seq_prob:
    #                 current_input_ids = self.truncate_input(current_input_ids, rng)
    #             single_span = rng.random() < self.single_span_prob
    #             if single_span:
    #                 masked_lengths = [
    #                     rng.choices(
    #                         range(1, len(self.block_length_distribution) + 1),
    #                         weights=self.block_length_distribution,
    #                     )[0]
    #                 ]
    #             else:
    #                 masked_lengths, masked_count = [], 0
    #                 while masked_count < int(self.mask_ratio * len(current_input_ids)):
    #                     block_length = rng.choices(
    #                         range(1, len(self.block_length_distribution) + 1),
    #                         weights=self.block_length_distribution,
    #                     )[0]
    #                     masked_lengths.append(block_length)
    #                     masked_count += block_length
    #             tokens, targets, loss_masks, position_ids, division = self.generate_blank_data(
    #                 current_input_ids, masked_lengths, rng, task="bert"
    #             )
    #             tokens, targets, loss_masks, position_ids = self.pad_batch(
    #                 tokens, targets, loss_masks, position_ids,
    #                 max_seq_length=self.max_seq_length // self.aggregated_samples_per_sequence
    #             )
    #             if self.relative_pos_encoding:
    #                 position_ids = self._build_relative_pos_encoding(position_ids, division)
    #             elif self.no_2d_encoding:
    #                 position_ids = position_ids[0]
    #             division = np.array([division], dtype=int)
    #             sequences.append((tokens, targets, loss_masks, position_ids, division))
    #         return *self._pack_samples(sequences), 0
    #     else:
    #         sequences = []
    #         if self.aggregate_gpt_sample:
    #             assert self.max_seq_length % self.aggregated_samples_per_sequence == 0
    #             assert len(input_ids) % self.aggregated_samples_per_sequence == 0
    #             input_length = len(input_ids) // self.aggregated_samples_per_sequence
    #             aggregated_samples_per_sequence = self.aggregated_samples_per_sequence
    #         else:
    #             input_length = len(input_ids)
    #             aggregated_samples_per_sequence = 1
    #         for i in range(aggregated_samples_per_sequence):
    #             current_input_ids = input_ids[input_length * i: input_length * (i + 1)]
    #             generation_length = rng.randint(
    #                 int(self.min_gmask_ratio * len(current_input_ids)), len(current_input_ids)
    #             )
    #             division = len(current_input_ids) - generation_length
    #             source_tokens, target_tokens = (
    #                 current_input_ids[:division],
    #                 current_input_ids[division:],
    #             )
    #             target_masks = np.ones(len(target_tokens), dtype=int)
    #             tokens = np.concatenate(
    #                 (
    #                     source_tokens,
    #                     [self.gmask_id, self.sop_id],
    #                     target_tokens[:-1],
    #                 )
    #             )
    #             targets = np.concatenate(
    #                 (source_tokens, [self.gmask_id], target_tokens)
    #             )
    #             loss_masks = np.concatenate(
    #                 (np.zeros(len(source_tokens) + 1, dtype=int), target_masks)
    #             )
    #             position_ids = np.arange(
    #                 len(source_tokens) + len(target_tokens) + 1, dtype=int
    #             )
    #             position_ids[len(source_tokens) + 1:] = len(source_tokens)
    #             block_position_ids = np.concatenate(
    #                 (
    #                     np.zeros(len(source_tokens), dtype=int),
    #                     np.arange(len(target_tokens) + 1, dtype=int),
    #                 )
    #             )
    #             position_ids = np.stack([position_ids, block_position_ids], axis=0)
    #             division = division + 1
    #             tokens, targets, loss_masks, position_ids = self.pad_batch(
    #                 tokens, targets, loss_masks, position_ids,
    #                 max_seq_length=self.max_seq_length // aggregated_samples_per_sequence
    #             )
    #             if self.relative_pos_encoding:
    #                 position_ids = self._build_relative_pos_encoding(position_ids, division)
    #             elif self.no_2d_encoding:
    #                 position_ids = np.arange(len(tokens), dtype=int)
    #             # attention_mask = self.build_mask_matrix(division, self.max_seq_length)
    #             division = np.array([division], dtype=int)
    #             sequences.append((tokens, targets, loss_masks, position_ids, division))
    #         return *self._pack_samples(sequences), 1
