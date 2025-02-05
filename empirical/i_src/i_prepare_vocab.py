import json
import tqdm
import pickle
import argparse
import numpy as np
from collections import Counter
from collections import defaultdict


class VocabHelp(object):
    def __init__(self, counter, specials=["<pad>", "<unk>"]):
        """
        Initialize a VocabHelp object for managing vocabulary.

        Args:
            counter (collections.Counter): A counter object mapping words to their frequencies.
            specials (list of str, optional): A list of special tokens to be included in the vocabulary.
                Defaults to ['<pad>', '<unk>'].

        Attributes:
            pad_index (int): Index of the <pad> token in the vocabulary, default is 0.
            unk_index (int): Index of the <unk> token in the vocabulary, default is 1.
            itos (list): A list of tokens in the vocabulary, sorted first by the given specials,
                then by frequency (descending), and alphabetically for ties.
            stoi (dict): A dictionary mapping each token (str) to its index (int) in the vocabulary.
        """
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        # words_and_frequencies is a tuple
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        """
        Check equality between two VocabHelp objects.

         Args:
             other (VocabHelp): Another instance of the VocabHelp class to compare against.

         Returns:
             bool: True if both objects have the same 'stoi' (string-to-index) and 'itos' (index-to-string) attributes,
                 False otherwise.

        This method compares two VocabHelp instances by their 'stoi' and 'itos' attributes.
        If both are identical in terms of token-to-index and index-to-token mappings,
        the objects are considered equal.
        """
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        """
        Return the number of tokens in the vocabulary.

        Returns:
            int: The total number of tokens in the vocabulary, which is the length of the 'itos' list.

        This method returns the number of tokens present in the vocabulary, which is represented
        by the 'itos' (index-to-string) list. The length of this list corresponds to the total
        number of tokens, including special tokens like <pad> and <unk>.
        """
        return len(self.itos)

    def extend(self, v):
        """
        Extend the current vocabulary with words from another VocabHelp instance.

        Args:
            v (VocabHelp): Another instance of the VocabHelp class whose tokens are to be added to the current vocabulary.

        Returns:
            VocabHelp: The updated VocabHelp object with the extended vocabulary.

        This method adds tokens from another VocabHelp instance to the current vocabulary.
        It checks if each word is already in the vocabulary (based on 'stoi'). If not, the word is added
        to the 'itos' list, and a corresponding index is assigned in the 'stoi' dictionary.
        """
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
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

    def save_vocab(self, vocab_path):
        """
        Serializes the Python object (`self`) and writes it to a specified file.

        Args:
            vocab_path (str): The path where the serialized object will be stored.

        This method opens the file at `vocab_path` in binary write mode and uses the
        `pickle` module to serialize the current object (`self`) and save it to the file.
        This operation creates a binary file containing the serialized version of the object,
        which can later be loaded or deserialized using `pickle.load()`.

        Returns:
            None: This method does not return anything, as the primary operation is side-effect based
            (saving the serialized object to a file).

        NOTE:
            Serialization: The process of converting a Python object into a
                byte stream so it can be saved to a file or transmitted. In
                this case, the object is serialized with pickle.dump() into
                the file.
            Deserialization: This would occur when reading the object back
                from the file using pickle.load().

        """

        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


def parse_args():
    """
    Parse command-line arguments for the vocabulary preparation process.

    Returns:
        Namespace: The parsed command-line arguments as an `argparse.Namespace` object,
                   which includes the following attributes:
                   - `data_dir`: The directory containing the data (default: '../data/D1/res14').
                   - `vocab_dir`: The directory to output the prepared vocabulary (default: '../data/D1/res14').
                   - `lower`: A flag indicating whether to lowercase all words (default: False).

    This function uses `argparse` to handle command-line arguments for a vocabulary preparation
    task. It provides options for specifying data and vocabulary directories, and optionally
    for lowercasing all words in the data.
    """
    parser = argparse.ArgumentParser(description="Prepare vocab.")
    parser.add_argument(
        "--data_dir",
        default="/Users/ericklopez/Desktop/AspectSentimentTripletExtraction/empirical/data/D1/res14",
        help="data directory.",
    )
    parser.add_argument(
        "--vocab_dir",
        default="/Users/ericklopez/Desktop/test_pytorch/empirical/data/D1/res14",
        help="Output vocab directory.",
    )
    parser.add_argument(
        "--lower", default=False, help="If specified, lowercase all words."
    )
    args = parser.parse_args()
    return args


def load_tokens(filename):
    """
    Load tokens, dependency relations, and part-of-speech tags from a JSON file.

    Args:
        filename (str): The path to the JSON file containing sentence data with
                         tokens, dependency relations, and part-of-speech tags.

    Returns:
        tuple: A tuple containing:
            - `tokens` (list): A list of all tokens from the sentences in the file.
            - `deprel` (list): A list of dependency relations corresponding to each token.
            - `postag` (list): A list of tuples containing all pairwise combinations of part-of-speech tags.
            - `postag_ca` (list): A list of part-of-speech tags for each token, as they appear in the input data.
            - `max_len` (int): The maximum sentence length (number of tokens) across all sentences in the file.

    This method processes a JSON file containing sentences with their associated dependency relations
    and part-of-speech tags. It extracts and stores the tokens, dependency relations, and generates
    pairwise combinations of part-of-speech tags. It also calculates the maximum sentence length in the data.
    """
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        deprel = []
        postag = []
        postag_ca = []

        max_len = 0
        for d in data:
            sentence = d["sentence"].split()
            # NOTE to self, extend vs append
            # extend,  extends an existing list with each item from the list individaully
            # >>>[1, 2, 3, 4, 5, 6]
            # append, appends the actual list [4,5] within an existing list
            # >>>[1, 2, 3, 4, [5, 6]]
            tokens.extend(sentence)
            deprel.extend(d["deprel"])
            postag_ca.extend(d["postag"])
            # postag.extend(d['postag'])
            n = len(d["postag"])
            tmp_pos = []
            for i in range(n):
                for j in range(n):
                    tup = tuple(sorted([d["postag"][i], d["postag"][j]]))
                    tmp_pos.append(tup)
            postag.extend(tmp_pos)

            max_len = max(len(sentence), max_len)
    print(
        "{} tokens from {} examples loaded from {}.".format(
            len(tokens), len(data), filename
        )
    )
    return tokens, deprel, postag, postag_ca, max_len


def main():
    args = parse_args()

    # input files
    train_file = args.data_dir + "/train.json"
    test_file = args.data_dir + "/test.json"
    dev_file = args.data_dir + "/dev.json"

    # output files
    # token
    vocab_tok_file = args.vocab_dir + "/vocab_tok.vocab"
    # position
    vocab_post_file = args.vocab_dir + "/vocab_post.vocab"
    # deprel
    vocab_deprel_file = args.vocab_dir + "/vocab_deprel.vocab"
    # postag
    vocab_postag_file = args.vocab_dir + "/vocab_postag.vocab"
    # syn_post
    vocab_synpost_file = args.vocab_dir + "/vocab_synpost.vocab"

    # Load files
    print("Loading Files...")
    (
        train_tokens,
        train_deprel,
        train_postag,
        train_postag_ca,
        train_max_len,
    ) = load_tokens(train_file)

    dev_tokens, dev_deprel, dev_postag, dev_postag_ca, dev_max_len = (
        load_tokens(dev_file)
    )

    test_tokens, test_deprel, test_postag, test_postag_ca, test_max_len = (
        load_tokens(test_file)
    )

    # Lowercase Tokens
    if args.lower:
        train_tokens, dev_tokens, test_tokens = [
            [t.lower() for t in tokens]
            for tokens in (train_tokens, dev_tokens, test_tokens)
        ]

    # Counters
    token_counter = Counter(train_tokens + dev_tokens + test_tokens)
    deprel_counter = Counter(train_deprel + dev_deprel + test_deprel)
    postag_counter = Counter(train_postag + dev_postag + test_postag)
    postag_ca_counter = Counter(
        train_postag_ca + dev_postag_ca + test_postag_ca
    )

    deprel_counter["self"] = 1

    max_len = max(train_max_len, dev_max_len, test_max_len)
    post_counter = Counter(list(range(0, max_len)))
    syn_post_counter = Counter(list(range(0, 5)))

    # Build vocab
    print("Building Vocabulary...")
    token_vocab = VocabHelp(token_counter, specials=["<pad>", "<unk>"])
    post_vocab = VocabHelp(post_counter, specials=["<pad>", "<unk>"])
    deprel_vocab = VocabHelp(deprel_counter, specials=["<pad>", "<unk>"])
    postag_vocab = VocabHelp(postag_counter, specials=["<pad>", "<unk>"])
    syn_post_vocab = VocabHelp(syn_post_counter, specials=["<pad>", "<unk>"])

    print(
        f"token_vocab: {len(token_vocab)}\n post_vocab: {len(post_vocab)}\n syn_post_vocab: {len(syn_post_vocab)}\n deprel_vocab: {len(deprel_vocab)}\n postag_vocab: {len(postag_vocab)}"
    )

    print("Serializing object via pickle.dump(), converting")
    post_vocab.save_vocab(vocab_post_file)
    deprel_vocab.save_vocab(vocab_deprel_file)
    postag_vocab.save_vocab(vocab_postag_file)
    syn_post_vocab.save_vocab(vocab_synpost_file)
    print("All  Finish")


if __name__ == "__main__":
    pass

# =============================================================================
#     filename = "/Users/ericklopez/Desktop/test_pytorch/empirical/data/D1/res14/train.json"
#
#     try:
#         pass
#
#     except Exception as e:
#         print(f"Better Luck Next Time... {e}")
# =============================================================================
