import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
AVAILABLE = all(importlib.util.find_spec(x) for x in ("torch", "tokenizers", "numpy"))

@unittest.skipUnless(AVAILABLE, "optional transport dependencies")
class InterfaceTests(unittest.TestCase):
    def fixture(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        class Source:
            pieces = ["▁play", "ing"] + ["▁unused" + str(i) for i in range(10)]
            def __len__(self): return len(self.pieces)
            def convert_ids_to_tokens(self, ids): return [self.pieces[i] for i in ids]
            def decode(self, ids, skip_special_tokens=True):
                return "".join(self.pieces[i].removeprefix("▁") for i in ids)
        class Target:
            eos_token_id = 0
            backend_tokenizer = Tokenizer(WordLevel(
                {"<eos>": 0, "<unk>": 1, " play": 2, "ing": 3, " playing": 4},
                unk_token="<unk>"))
        table = torch.nn.Embedding.from_pretrained(torch.arange(20).reshape(5, 4).float())
        return torch, Source(), Target(), torch.eye(12), table

    def test_composition_and_cache(self):
        from migro.transport import boundary_lexicon, ComposedTransport, ValidVocabularyTransport
        torch, source, target, source_table, target_table = self.fixture()
        lexicon = boundary_lexicon(source, target, "A")
        full = ComposedTransport(source_table, target_table, lexicon, source, target, "A")
        simple = ValidVocabularyTransport(source_table, target_table, lexicon)
        z = source_table[:2]
        full.cache_enabled = True
        result = full(z)
        self.assertEqual(tuple(result.shape), (1, 4))
        self.assertEqual(tuple(simple(z).shape), (2, 4))
        torch.testing.assert_close(result[0], target_table.weight[4])
        torch.testing.assert_close(simple(z), target_table.weight[[2, 3]])
        full.clear_query_cache()
        torch.testing.assert_close(result, full(z), rtol=0, atol=0)
        self.assertNotIn("host_embeddings.weight", full.state_dict())

    def test_zero_residual_identity_and_nonzero_adapter(self):
        import torch
        from migro.calibration import ResidualInterface, BoundaryAdapter
        torch.manual_seed(7)
        x = torch.randn(5, 8)
        adapter = ResidualInterface(8)
        torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
        with torch.no_grad():
            adapter.up.weight.fill_(0.1)
        self.assertFalse(torch.equal(adapter(x), x))
        boundary = BoundaryAdapter(8)
        torch.testing.assert_close(boundary(x[:2], x[2:], x[-1]), x[-1], rtol=0, atol=0)
