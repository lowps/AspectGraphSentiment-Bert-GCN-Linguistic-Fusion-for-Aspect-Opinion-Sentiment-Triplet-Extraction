import math
import torch
import numpy as np
from collections import OrderedDict, defaultdict
from transformers import BertTokenizer
import pickle
import argparse
import os
import sys



sentiment2id = {"negative": 3, "neutral": 4, "positive": 5}

label = [
    "N",
    "B-A",
    "I-A",
    "A",
    "B-O",
    "I-O",
    "O",
    "negative",
    "neutral",
    "positive",
]

label2id, id2label = OrderedDict(), OrderedDict()
for i, v in enumerate(label):
    label2id[v] = i
    id2label[i] = v


def get_spans(tags: str) -> list[list[int]]:
    """
    Extracts spans from a list of BIO (Beginning, Inside, Outside) tags.

    This function identifies continuous spans of entities based on the BIO tagging scheme.
    - 'B' denotes the beginning of an entity.
    - 'I' denotes a continuation inside the entity.
    - 'O' denotes outside the entity.

    Args:
        tags (str): A string of space-separated BIO tags (e.g., "B-PER I-PER O B-LOC O").

    Returns:
        list[list[int]]: A list of spans, where each span is represented as a list of two integers [start, end],
              indicating the start and end indices of the identified entities.

    Example:
        >>> get_spans("B-PER I-PER O B-LOC O")
        [[0, 1], [3, 3]]
        or
        tag = "All\\O I\\O can\\O say\\O is\\O $\\O 2\\O pints\\O during\\O happy\\O hour\\O and\\O the\\O some\\O of\\O the\\O cheapest\\O oysters\\B you\\O ll\\O find\\O in\\O the\\O city\\O ,\\O though\\O the\\O quality\\O is\\O some\\O of\\O the\\O best\\O .\\O"
        >>> get_spans(tag)
        [[17, 17]]

    """
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith("B"):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith("O"):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_evaluate_spans(
    tags: list, length: int, token_range: list[list[int]]
) -> list:
    """
    Extracts evaluation spans from tokenized input based on a tagging scheme.

    This function identifies continuous spans based on specific tag values and
    returns the start and end indices of each span. It works with a custom tagging
    scheme where:
    - `1` indicates the beginning of an entity or span.
    - `0` indicates the end or outside of an entity.
    - `-1` indicates that the token should be ignored.

    Args:
        tags (list): A list of tags associated with each token, where the value represents the
                     state of the token (e.g., 1 for start of span, 0 for end, -1 for ignored).
        length (int): The total length of the input sequence or number of tokens.
        token_range (list of tuples) or (list of lists): A list of tuples, where each tuple `(l, r)` represents
                                      the range of a token (e.g., (start_index, end_index)).

    Returns:
        list: A list of spans, where each span is a list of two integers [start, end],
              representing the start and end indices of the identified entity or span.

    Example:
        >>> get_evaluate_spans([1, 0, -1, 1, 0], 5, [(0, 2), (3, 5)])
        [[0, 1], [3, 4]]
    """
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):
    """
    Instance Class: Processes a sentence and its metadata into structured data
    for a machine learning model that accepts tensor inputs. It handles tokenization,
    tagging, and feature generation for aspects, opinions and their relationships.
    The output includes tensors for tokens, tags, and word-pair features, ready
    for model input.

    """

    def __init__(
        self,
        tokenizer,
        sentence_pack,
        post_vocab,
        deprel_vocab,
        postag_vocab,
        synpost_vocab,
        args,
    ):
        self.id = sentence_pack["id"]
        self.sentence = sentence_pack["sentence"]
        self.tokens = self.sentence.strip().split()
        self.postag = sentence_pack["postag"]
        self.head = sentence_pack["head"]
        self.deprel = sentence_pack["deprel"]
        self.sen_length = len(self.tokens)
        # Maps each token to its start and end positions in BERT tokens EX:"The sushi", token_range is [[1, 1], [2, 2]]
        self.token_range = []
        # BERT token IDs for the sentence
        self.bert_tokens = tokenizer.encode(self.sentence)

        self.length = len(self.bert_tokens)
        self.bert_tokens_padding = torch.zeros(args.max_sequence_len).long()
        self.aspect_tags = torch.zeros(args.max_sequence_len).long()
        self.opinion_tags = torch.zeros(args.max_sequence_len).long()
        # Matrix for relationships between tokens EX: aspect-opinion pairs
        self.tags = torch.zeros(
            args.max_sequence_len, args.max_sequence_len
        ).long()
        # Symmetric version of self.tags
        self.tags_symmetry = torch.zeros(
            args.max_sequence_len, args.max_sequence_len
        ).long()

        # Mask for valid tokens (1 for tokens, 0 for padding)
        self.mask = torch.zeros(args.max_sequence_len)

        for i in range(self.length):
            self.bert_tokens_padding[i] = self.bert_tokens[i]
        self.mask[: self.length] = 1

        token_start = 1
        for (
            i,
            w,
        ) in enumerate(self.tokens):
            token_end = token_start + len(
                tokenizer.encode(w, add_special_tokens=False)
            )
            self.token_range.append([token_start, token_end - 1])
            token_start = token_end
        # self.token_range[-1][-1]: Accesses the last element of the last [start, end] pair. And +2 accounts for the 2x special tokens added "[CLS]" & "[SEP]"
        # Essentially indexing the range from 1st token to last token including the 2x special tokens.
        assert self.length == self.token_range[-1][-1] + 2

        # Colon ":" is used for slicing lists, arrays, sequence-like objects.
        # Slices all elements from index "self.length" until the end of the list/tensor
        self.aspect_tags[self.length :] = -1
        self.aspect_tags[0] = -1
        self.aspect_tags[self.length - 1] = -1

        self.opinion_tags[self.length :] = -1
        self.opinion_tags[0] = -1
        self.opinion_tags[self.length - 1] = -1

        self.tags[:, :] = -1
        self.tags_symmetry[:, :] = -1
        for i in range(1, self.length - 1):
            for j in range(i, self.length - 1):
                self.tags[i][j] = 0

        # Processing Triples: Processes aspect-opinion-sentiment triples from sentence_pack
        # Steps: Extracts spans for aspects and opinions using "get_spans()",
        # Set tags for aspect and opinion in self.tags, self.aspect_tags and
        # self.opinion_tags, & Handle sentiment labels for aspect-opinion pairs.
        for triple in sentence_pack["triples"]:
            aspect = triple["target_tags"]
            opinion = triple["opinion_tags"]
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            # set tag for aspect
            for l, r in aspect_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end + 1):
                    for j in range(i, end + 1):
                        if j == start:
                            self.tags[i][j] = label2id["B-A"]
                        elif j == i:
                            self.tags[i][j] = label2id["I-A"]
                        else:
                            self.tags[i][j] = label2id["A"]

                # self.aspect_tags attribute
                for i in range(l, r + 1):
                    set_tag = 1 if i == l else 2
                    al, ar = self.token_range[i]
                    self.aspect_tags[al] = set_tag
                    self.aspect_tags[al + 1 : ar + 1] = -1
                    # mask positions of sub words
                    self.tags[al + 1 : ar + 1, :] = -1
                    self.tags[:, al + 1 : ar + 1] = -1

            # set tag for opinion
            for l, r in opinion_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end + 1):
                    for j in range(i, end + 1):
                        if j == start:
                            self.tags[i][j] = label2id["B-O"]
                        elif j == i:
                            self.tags[i][j] = label2id["I-O"]
                        else:
                            self.tags[i][j] = label2id["O"]

                # self.opinion_tags attribute
                for i in range(l, r + 1):
                    set_tag = 1 if i == l else 2
                    pl, pr = self.token_range[i]
                    self.opinion_tags[pl] = set_tag
                    self.opinion_tags[pl + 1 : pr + 1] = -1
                    self.tags[pl + 1 : pr + 1, :] = -1
                    self.tags[:, pl + 1 : pr + 1] = -1

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar + 1):
                        for j in range(pl, pr + 1):
                            sal, sar = self.token_range[i]
                            spl, spr = self.token_range[j]
                            self.tags[sal : sar + 1, spl : spr + 1] = -1
                            if args.task == "pair":
                                if i > j:
                                    self.tags[spl][sal] = 7
                                else:
                                    self.tags[sal][spl] = 7
                            elif args.task == "triplet":
                                if i > j:
                                    self.tags[spl][sal] = label2id[
                                        triple["sentiment"]
                                    ]
                                else:
                                    self.tags[sal][spl] = label2id[
                                        triple["sentiment"]
                                    ]

        for i in range(1, self.length - 1):
            for j in range(i, self.length - 1):
                self.tags_symmetry[i][j] = self.tags[i][j]
                self.tags_symmetry[j][i] = self.tags_symmetry[i][j]

        """1) Generate position index of the word pair"""
        self.word_pair_position = torch.zeros(
            args.max_sequence_len, args.max_sequence_len
        ).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_position[row][col] = (
                            post_vocab.stoi.get(
                                abs(row - col), post_vocab.unk_index
                            )
                        )

        """2. generate deprel index of the word pair"""
        self.word_pair_deprel = torch.zeros(
            args.max_sequence_len, args.max_sequence_len
        ).long()
        for i in range(len(self.tokens)):
            start = self.token_range[i][0]
            end = self.token_range[i][1]
            for j in range(start, end + 1):
                s, e = (
                    self.token_range[self.head[i] - 1]
                    if self.head[i] != 0
                    else (0, 0)
                )
                for k in range(s, e + 1):
                    self.word_pair_deprel[j][k] = deprel_vocab.stoi.get(
                        self.deprel[i]
                    )
                    self.word_pair_deprel[k][j] = deprel_vocab.stoi.get(
                        self.deprel[i]
                    )
                    self.word_pair_deprel[j][j] = deprel_vocab.stoi.get("self")

        """3. generate POS tag index of the word pair"""
        self.word_pair_pos = torch.zeros(
            args.max_sequence_len, args.max_sequence_len
        ).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_pos[row][col] = postag_vocab.stoi.get(
                            tuple(sorted([self.postag[i], self.postag[j]]))
                        )

        """4. generate synpost index of the word pair"""
        self.word_pair_synpost = torch.zeros(
            args.max_sequence_len, args.max_sequence_len
        ).long()
        tmp = [[0] * len(self.tokens) for _ in range(len(self.tokens))]
        for i in range(len(self.tokens)):
            j = self.head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = defaultdict(list)
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        word_level_degree = [
            [4] * len(self.tokens) for _ in range(len(self.tokens))
        ]

        for i in range(len(self.tokens)):
            node_set = set()
            word_level_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    if k not in node_set:
                        word_level_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                word_level_degree[i][g] = 3
                                node_set.add(g)

        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_synpost[row][col] = (
                            synpost_vocab.stoi.get(
                                word_level_degree[i][j],
                                synpost_vocab.unk_index,
                            )
                        )


def load_data_instances(
    sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args
):
    instances = list()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    for sentence_pack in sentence_packs:
        instances.append(
            Instance(
                tokenizer,
                sentence_pack,
                post_vocab,
                deprel_vocab,
                postag_vocab,
                synpost_vocab,
                args,
            )
        )
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances) / args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []
        tags_symmetry = []
        word_pair_position = []
        word_pair_deprel = []
        word_pair_pos = []
        word_pair_synpost = []

        for i in range(
            index * self.args.batch_size,
            min((index + 1) * self.args.batch_size, len(self.instances)),
        ):
            sentence_ids.append(self.instances[i].id)
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)
            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)
            tags_symmetry.append(self.instances[i].tags_symmetry)

            word_pair_position.append(self.instances[i].word_pair_position)
            word_pair_deprel.append(self.instances[i].word_pair_deprel)
            word_pair_pos.append(self.instances[i].word_pair_pos)
            word_pair_synpost.append(self.instances[i].word_pair_synpost)

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)
        tags = torch.stack(tags).to(self.args.device)
        tags_symmetry = torch.stack(tags_symmetry).to(self.args.device)

        word_pair_position = torch.stack(word_pair_position).to(
            self.args.device
        )
        word_pair_deprel = torch.stack(word_pair_deprel).to(self.args.device)
        word_pair_pos = torch.stack(word_pair_pos).to(self.args.device)
        word_pair_synpost = torch.stack(word_pair_synpost).to(self.args.device)

        return (
            sentence_ids,
            sentences,
            bert_tokens,
            lengths,
            masks,
            sens_lens,
            token_ranges,
            aspect_tags,
            tags,
            word_pair_position,
            word_pair_deprel,
            word_pair_pos,
            word_pair_synpost,
            tags_symmetry,
        )


def main():
    pass


if __name__ == "__main__":

    import inspect
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from i_src.i_prepare_vocab import VocabHelp

    def parse_args():
        parser = argparse.ArgumentParser(description="configurations")

        parser.add_argument(
            "--max_sequence_len",
            type=int,
            default=102,
            help="max length of a sentence",
        )

        parser.add_argument(
            "--task",
            type=str,
            default="triplet",
            choices=["triplet"],
            help="option: pair, triplet",
        )
        args = parser.parse_args()
        return args

    sentence_pack = {
        "id": "339",
        "sentence": "The sushi was awful !",
        "postag": ["DT", "NN", "VBD", "JJ", "."],
        "head": [2, 4, 4, 0, 4],
        "deprel": ["det", "nsubj", "cop", "root", "punct"],
        "triples": [
            {
                "uid": "339-0",
                "target_tags": "The\\O sushi\\B was\\O awful\\O !\\O",
                "opinion_tags": "The\\O sushi\\O was\\O awful\\B !\\O",
                "sentiment": "negative",
            }
        ],
    }
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sentence_pack["sentence"]

    def load_vocab(vocab_path: str):
        """
        Load a vocabulary from a file.

        Args:
           vocab_path (str): The file path to the vocabulary file, which is expected to be a serialized object.

        Returns:
            object: The deserialized vocabulary object loaded from the specified file.

        This method loads a vocabulary object from a file using `pickle`. The file should contain
        a serialized version of the vocabulary, which is deserialized and returned.
        """

        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    path1 = "/Users/ericklopez/Desktop/AspectSentimentTripletExtraction/empirical/data/D1/res14/vocab_deprel.vocab"
    path2 = "/Users/ericklopez/Desktop/AspectSentimentTripletExtraction/empirical/data/D1/res14/vocab_post.vocab"
    path3 = "/Users/ericklopez/Desktop/AspectSentimentTripletExtraction/empirical/data/D1/res14/vocab_postag.vocab"
    path4 = "/Users/ericklopez/Desktop/AspectSentimentTripletExtraction/empirical/data/D1/res14/vocab_synpost.vocab"

    vd = load_vocab(path1)
    vp = load_vocab(path2)
    vpt = load_vocab(path3)
    vs = load_vocab(path4)

    args = parse_args()

    class TestInstance(object):
        """
        Instance Class: Processes a sentence and its metadata into structured data
        for a machine learning model that accepts tensor inputs. It handles tokenization,
        tagging, and feature generation for aspects, opinions and their relationships.
        The output includes tensors for tokens, tags, and word-pair features, ready
        for model input.

        """

        def __init__(
            self,
            tokenizer,
            sentence_pack,
            post_vocab,
            deprel_vocab,
            postag_vocab,
            synpost_vocab,
            args,
        ):
            self.id = sentence_pack["id"]
            self.sentence = sentence_pack["sentence"]
            self.tokens = self.sentence.strip().split()
            self.postag = sentence_pack["postag"]
            self.head = sentence_pack["head"]
            self.deprel = sentence_pack["deprel"]
            self.sen_length = len(self.tokens)
            # Maps each token to its start and end positions in BERT tokens EX:"The sushi", token_range is [[1, 1], [2, 2]]
            self.token_range = []
            # BERT token IDs for the sentence
            # self.sentende is length 5, then after this its length 8 because
            # 1.Adds [CLS] at the beginning.
            # 2.Tokenizes each word (possibly splitting into subwords).
            # 3.Adds [SEP] at the end.

            self.bert_tokens = tokenizer.encode(self.sentence)

            self.length = len(self.bert_tokens)
            self.bert_tokens_padding = torch.zeros(
                args.max_sequence_len
            ).long()
            self.aspect_tags = torch.zeros(args.max_sequence_len).long()
            self.opinion_tags = torch.zeros(args.max_sequence_len).long()
            # Matrix for relationships between tokens EX: aspect-opinion pairs
            self.tags = torch.zeros(
                args.max_sequence_len, args.max_sequence_len
            ).long()
            # Symmetric version of self.tags
            self.tags_symmetry = torch.zeros(
                args.max_sequence_len, args.max_sequence_len
            ).long()

            # Mask for valid tokens (1 for tokens, 0 for padding)
            self.mask = torch.zeros(args.max_sequence_len)

            # Enables self.mask to know the length is 8 and to add 1's for tokens
            # And 0 for the remaining default = 102, producing a sparse matrix.
            # self.bert_tokens_padding@@@@@@
            for i in range(self.length):
                # mapping all the numerical tokens within "self.bert_tokens" and
                # in this example its able to mape it from range "self.length"
                # which is 8 thus range(0,8)
                # EX:
                # bert_tokens: [101, 1996, 10514,
                # bert_tokens_padding: tensor([  101,  1996, 10514,  0, 0, 0])
                # 0's up to default 102 is reached

                self.bert_tokens_padding[i] = self.bert_tokens[i]
            self.mask[: self.length] = 1

            token_start = 1
            # only 1x word in "w" unless a word tokenized into subwords like "'sush', '##i'",
            # keep that in mind in respects to len()
            for (
                i,
                w,
            ) in enumerate(self.tokens):
                token_end = token_start + len(
                    tokenizer.encode(w, add_special_tokens=False)
                )
                # self.token_range@@@@@@
                # token_range: This is a mapping that connects each token in the
                # original sentence "self.tokens" to its corresponding positions
                # in the BERT tokenized sequence "self.bert_tokens" and this is
                # necessary because BERT tokenization split words into subword
                # units, and the token_range helps keep TRACK where each original
                # token starts and ends in the BERT token sequence
                # EX:
                # ['[CLS]', 'The', 'sush', '##i', 'was', 'awful', '!', '[SEP]']
                # token_range MAPPING: [[1, 1], [2, 3], [4, 4], [5, 5], [6, 6]]
                # "The" → BERT tokens at index [1, 1] ("The").
                # "sushi" → BERT tokens at indices [2, 3] ("sush" and "##i").
                # "was" → BERT token at index [4, 4] ("was").
                # "awful" → BERT token at index [5, 5] ("awful").
                # "!" → BERT token at index [6, 6] ("!").

                self.token_range.append([token_start, token_end - 1])
                # token_start gets updated here to token_end
                token_start = token_end
            assert self.length == self.token_range[-1][-1] + 2

            self.aspect_tags[self.length :] = -1
            self.aspect_tags[0] = -1
            self.aspect_tags[self.length - 1] = -1

            self.opinion_tags[self.length :] = -1
            self.opinion_tags[0] = -1
            self.opinion_tags[self.length - 1] = -1

            self.tags[:, :] = -1
            self.tags_symmetry[:, :] = -1

            for i in range(1, self.length - 1):
                for j in range(i, self.length - 1):
                    self.tags[i][j] = 0

            # Processing Triples: Processes aspect-opinion-sentiment triples from sentence_pack
            # Steps: Extracts spans for aspects and opinions using "get_spans()",
            # Set tags for aspect and opinion in self.tags, self.aspect_tags and
            # self.opinion_tags, & Handle sentiment labels for aspect-opinion pairs.
            for triple in sentence_pack["triples"]:
                aspect = triple["target_tags"]
                opinion = triple["opinion_tags"]
                aspect_span = get_spans(aspect)
                opinion_span = get_spans(opinion)

                # set tag for aspect
                for l, r in aspect_span:
                    start = self.token_range[l][0]
                    end = self.token_range[r][1]
                    for i in range(start, end + 1):
                        for j in range(i, end + 1):
                            if j == start:
                                self.tags[i][j] = label2id["B-A"]
                            elif j == i:
                                self.tags[i][j] = label2id["I-A"]
                            else:
                                self.tags[i][j] = label2id["A"]

                    # self.aspect_tags attribute
                    for i in range(l, r + 1):
                        set_tag = 1 if i == l else 2
                        al, ar = self.token_range[i]
                        self.aspect_tags[al] = set_tag
                        self.aspect_tags[al + 1 : ar + 1] = -1
                        # mask positions fo subwords
                        self.tags[al + 1 : ar + 1, :] = -1
                        self.tags[:, al + 1 : ar + 1] = -1

                # set tag for opinion
                for l, r in opinion_span:
                    start = self.token_range[l][0]
                    end = self.token_range[r][1]
                    for i in range(start, end + 1):
                        for j in range(i, end + 1):
                            if j == start:
                                self.tags[i][j] = label2id["B-O"]
                            elif j == i:
                                self.tags[i][j] = label2id["I-O"]
                            else:
                                self.tags[i][j] = label2id["O"]

                    # self.opinion_tags attribute
                    for i in range(l, r + 1):
                        set_tag = 1 if i == l else 2
                        pl, pr = self.token_range[i]
                        self.opinion_tags[pl] = set_tag
                        self.opinion_tags[pl + 1 : pr + 1] = -1
                        self.tags[pl + 1 : pr + 1, :] = -1
                        self.tags[:, pl + 1 : pr + 1] = -1

                for al, ar in aspect_span:
                    for pl, pr in opinion_span:
                        for i in range(al, ar + 1):
                            for j in range(pl, pr + 1):
                                sal, sar = self.token_range[i]
                                spl, spr = self.token_range[j]
                                self.tags[sal : sar + 1, spl : spr + 1] = -1
                                if args.task == "pair":
                                    if i > j:
                                        self.tags[spl][sal] = 7
                                    else:
                                        self.tags[sal][spl] = 7
                                elif args.task == "triplet":
                                    if i > j:
                                        self.tags[spl][sal] = label2id[
                                            triple["sentiment"]
                                        ]
                                    else:
                                        self.tags[sal][spl] = label2id[
                                            triple["sentiment"]
                                        ]

            for i in range(1, self.length - 1):
                for j in range(i, self.length - 1):
                    self.tags_symmetry[i][j] = self.tags[i][j]
                    self.tags_symmetry[j][i] = self.tags_symmetry[i][j]

            # 1) Generate position index of the word pair
            self.word_pair_position = torch.zeros(
                args.max_sequence_len, args.max_sequence_len
            ).long()
            for i in range(len(self.tokens)):
                start, end = self.token_range[i][0], self.token_range[i][1]
                for j in range(len(self.tokens)):
                    s, e = self.token_range[j][0], self.token_range[j][1]
                    for row in range(start, end + 1):
                        for col in range(s, e + 1):
                            self.word_pair_position[row][col] = (
                                post_vocab.stoi.get(
                                    abs(row - col), post_vocab.unk_index
                                )
                            )

            # 2. generate deprel index of the word pair
            self.word_pair_deprel = torch.zeros(
                args.max_sequence_len, args.max_sequence_len
            ).long()
            for i in range(len(self.tokens)):
                start = self.token_range[i][0]
                end = self.token_range[i][1]
                for j in range(start, end + 1):
                    s, e = (
                        self.token_range[self.head[i] - 1]
                        if self.head[i] != 0
                        else (0, 0)
                    )
                    for k in range(s, e + 1):
                        self.word_pair_deprel[j][k] = deprel_vocab.stoi.get(
                            self.deprel[i]
                        )
                        self.word_pair_deprel[k][j] = deprel_vocab.stoi.get(
                            self.deprel[i]
                        )
                        self.word_pair_deprel[j][j] = deprel_vocab.stoi.get(
                            "self"
                        )

            # 3. generate POS tag index of the word pair
            self.word_pair_pos = torch.zeros(
                args.max_sequence_len, args.max_sequence_len
            ).long()
            for i in range(len(self.tokens)):
                start, end = self.token_range[i][0], self.token_range[i][1]
                for j in range(len(self.tokens)):
                    s, e = self.token_range[j][0], self.token_range[j][1]
                    for row in range(start, end + 1):
                        for col in range(s, e + 1):
                            self.word_pair_pos[row][col] = (
                                postag_vocab.stoi.get(
                                    tuple(
                                        sorted(
                                            [self.postag[i], self.postag[j]]
                                        )
                                    )
                                )
                            )

            # 4. generate synpost index of the word pair
            self.word_pair_synpost = torch.zeros(
                args.max_sequence_len, args.max_sequence_len
            ).long()
            tmp = [[0] * len(self.tokens) for _ in range(len(self.tokens))]
            for i in range(len(self.tokens)):
                j = self.head[i]
                if j == 0:
                    continue
                tmp[i][j - 1] = 1
                tmp[j - 1][i] = 1

            tmp_dict = defaultdict(list)
            for i in range(len(self.tokens)):
                for j in range(len(self.tokens)):
                    if tmp[i][j] == 1:
                        tmp_dict[i].append(j)

            word_level_degree = [
                [4] * len(self.tokens) for _ in range(len(self.tokens))
            ]

            for i in range(len(self.tokens)):
                node_set = set()
                word_level_degree[i][i] = 0
                node_set.add(i)
                for j in tmp_dict[i]:
                    if j not in node_set:
                        word_level_degree[i][j] = 1
                        node_set.add(j)
                    for k in tmp_dict[j]:
                        if k not in node_set:
                            word_level_degree[i][k] = 2
                            node_set.add(k)
                            for g in tmp_dict[k]:
                                if g not in node_set:
                                    word_level_degree[i][g] = 3
                                    node_set.add(g)

            for i in range(len(self.tokens)):
                start, end = self.token_range[i][0], self.token_range[i][1]
                for j in range(len(self.tokens)):
                    s, e = self.token_range[j][0], self.token_range[j][1]
                    for row in range(start, end + 1):
                        for col in range(s, e + 1):
                            self.word_pair_synpost[row][col] = (
                                synpost_vocab.stoi.get(
                                    word_level_degree[i][j],
                                    synpost_vocab.unk_index,
                                )
                            )

    object1 = TestInstance(tokenizer, sentence_pack, vp, vd, vpt, vs, args)

    print("@@@@@@@STARTING NOW@@@@@@@")
    print("\n")

    listx = [
        "aspect_tags",
        "bert_tokens",
        "bert_tokens_padding",
        "deprel",
        "head",
        "id",
        "length",
        "mask",
        "opinion_tags",
        "postag",
        "sen_length",
        "sentence",
        "tags",
        "tags_symmetry",
        "token_range",
        "tokens",
        "word_pair_deprel",
        "word_pair_pos",
        "word_pair_position",
        "word_pair_synpost",
    ]
    # sort the list alphabetically
    sorted_listx = sorted(listx)

    for x in sorted_listx:
        print(f"{x}: {getattr(object1, x)}")

    print(len(object1.tags_symmetry))
    print("@@@@")
    for num, item1 in enumerate(object1.__dict__):
        num = num + 1
        print(f"{num}) {item1}\n")
    thing = object1.token_range
    
    
