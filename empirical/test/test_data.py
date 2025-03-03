import unittest
from unittest.mock import MagicMock, patch
import torch
from transformers import BertTokenizer
from .ii_data import get_spans, get_evaluate_spans, Instance


# Mock implementation of get_spans
def mock_get_spans(tags):
    """
    Mock implementation of get_spans that mimics the real function's behavior.
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


class TestInstance(unittest.TestCase):
    def setUp(self):
        # Mock tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Mock sentence_pack
        self.sentence_pack = {
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

        # Mock args
        self.args = MagicMock()
        self.args.max_sequence_len = 10

    # Patch get_spans
    @patch("ii_data.get_spans", side_effect=mock_get_spans)
    def test_instance_initialization(self, mock_get_spans):
        # Create an instance of the Instance class
        instance = Instance(
            self.tokenizer,
            self.sentence_pack,
            post_vocab=None,
            deprel_vocab=None,
            postag_vocab=None,
            synpost_vocab=None,
            args=self.args,
        )

        # Test attributes
        self.assertEqual(instance.id, "339")
        self.assertEqual(instance.sentence, "The sushi was awful !")
        self.assertEqual(instance.tokens, ["The", "sushi", "was", "awful", "!"])
        self.assertEqual(instance.postag, ["DT", "NN", "VBD", "JJ", "."])
        self.assertEqual(instance.head, [2, 4, 4, 0, 4])
        self.assertEqual(instance.deprel, ["det", "nsubj", "cop", "root", "punct"])
        self.assertEqual(instance.sen_length, 5)
        self.assertEqual(instance.length, len(instance.bert_tokens))
        self.assertEqual(
            instance.bert_tokens_padding.tolist()[:6], instance.bert_tokens
        )
        self.assertEqual(instance.mask.tolist(), [1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(instance.token_range, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        self.assertEqual(
            instance.aspect_tags.tolist(), [-1, 0, 1, 0, 0, -1, -1, -1, -1, -1]
        )
        self.assertEqual(
            instance.opinion_tags.tolist(), [-1, 0, 0, 0, 1, -1, -1, -1, -1, -1]
        )

        # Test triples
        self.assertEqual(len(instance.triples), 1)
        triple = instance.triples[0]
        self.assertEqual(triple["uid"], "339-0")
        self.assertEqual(triple["target_tags"], "The\\O sushi\\B was\\O awful\\O !\\O")
        self.assertEqual(triple["opinion_tags"], "The\\O sushi\\O was\\O awful\\B !\\O")
        self.assertEqual(triple["sentiment"], "negative")

        # Verify that get_spans was called correctly
        mock_get_spans.assert_any_call("The\\O sushi\\B was\\O awful\\O !\\O")
        mock_get_spans.assert_any_call("The\\O sushi\\O was\\O awful\\B !\\O")

        # Verify the spans returned by get_spans
        aspect_spans = mock_get_spans("The\\O sushi\\B was\\O awful\\O !\\O")
        opinion_spans = mock_get_spans("The\\O sushi\\O was\\O awful\\B !\\O")
        self.assertEqual(aspect_spans, [[1, 1]])  # "sushi" is the aspect
        self.assertEqual(opinion_spans, [[3, 3]])  # "awful" is the opinion


if __name__ == "__main__":
    unittest.main()
