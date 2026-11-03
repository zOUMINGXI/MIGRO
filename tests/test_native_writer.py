import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

@unittest.skipUnless(importlib.util.find_spec("torch"), "optional transport dependency")
class NativeWriterTests(unittest.TestCase):
    def test_causal_session_links_without_host_updates(self):
        import torch
        from migro.native_writer import NativeBindingWriter
        from migro.observations import observation_text
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0
            def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]
            def decode(self, ids, skip_special_tokens=True): return "".join(chr(c) for c in ids)
            def apply_chat_template(self, messages, **kwargs): return "\n".join(m["content"] for m in messages)
            def __call__(self, text, **kwargs):
                return dict(input_ids=self.encode(text), offset_mapping=[(i, i+1) for i in range(len(text))])
        class Layer(torch.nn.Module):
            def forward(self, x): return (torch.tanh(x),)
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 8)
                self.layers = torch.nn.ModuleList([Layer(), Layer()])
            def forward(self, input_ids, **kwargs):
                x = self.embedding(input_ids)
                for layer in self.layers: x = layer(x)[0]
                return x
        class Host(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Decoder()
        tok = Tokenizer()
        class State:
            def __init__(self):
                self.meta, self.ids = [], []
                self.core = SimpleNamespace(codec=SimpleNamespace(tokenizer=tok))
                self.mem = SimpleNamespace(read_slot=lambda i, flag: torch.tensor(self.ids[i]))
            def append_many(self, obs, writer, links):
                for row, link in zip(obs, links):
                    self.meta.append(dict(row, writer=writer, reference=link))
                    self.ids.append(tok.encode(observation_text(row)))
        host = Host().eval().requires_grad_(False)
        initial = {k: v.clone() for k, v in host.state_dict().items()}
        bridge = SimpleNamespace(host=host, tokenizer=tok, device=torch.device("cpu"), context_limit=4096)
        state = State()
        writer = NativeBindingWriter(state, bridge, "source")
        def obs(body, session):
            return dict(speaker="Ada", body=body, session=session, date="unknown",
                        role="observed_human", dialogue_id=body)
        self.assertEqual(writer.observe([obs("I visited Kyoto.", "s1"),
                                         obs("I liked that trip.", "s1")]), [None, 0])
        self.assertEqual(writer.observe([obs("A new session.", "s2")]), [None])
        self.assertEqual(writer.forwards, 2)
        self.assertEqual(writer.trace[1]["candidate_indices"], [0])
        self.assertEqual(writer.trace[2]["candidate_indices"], [])
        self.assertEqual(len(host.model.layers[writer.layer]._forward_hooks), 0)
        for name, tensor in host.state_dict().items():
            torch.testing.assert_close(tensor, initial[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in host.parameters()))
