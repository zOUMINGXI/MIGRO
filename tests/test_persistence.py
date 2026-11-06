import os
import importlib.util
import unittest

@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ["numpy","torch","zstandard","pyppmd","brotli"]), "Runtime dependencies required")
class PersistenceTests(unittest.TestCase):
    def test_native_ans_matches_python_and_roundtrips(self):
        import numpy as np
        from migro.backend import rd_semantic_ans as codec
        if not codec.NATIVE:
            self.skipTest("Build coding kernels first")
        rng = np.random.default_rng(7)
        cdf = codec.make_cdf(np.arange(1, 258))
        ids = rng.integers(0, 257, 1200)
        blob = codec.encode(ids, cdf)
        self.assertEqual(blob, codec.python_encode(ids, cdf))
        np.testing.assert_array_equal(codec.decode(blob, len(ids), cdf), ids)

    def test_causal_content_restores_unicode_and_metadata(self):
        from migro.backend.causal_content_codec import candidate, decode, BINARY
        if not BINARY.exists():
            self.skipTest("Build coding kernels first")
        rows = [{"body": "Ada moved to 京都.\n", "scope": "a", "revision": 1},
                {"body": "Ada moved to Osaka.\n", "scope": "a", "revision": 2},
                {"body": "Different session: café.", "scope": "b", "offset": 11}]
        blob, _ = candidate(rows, bits=12)
        self.assertEqual(decode(blob), rows)

    def test_arithmetic_stream_restores_tokens(self):
        from migro.backend.causal_neural_content import Encoder, Decoder, TOTAL
        bounds = [0, TOTAL//4, 3*TOTAL//4, TOTAL]
        tokens = [0, 1, 2, 1, 1, 0, 2] * 100
        encoder = Encoder()
        for token in tokens:
            encoder.put(bounds[token], bounds[token+1])
        decoder = Decoder(encoder.finish()); recovered = []
        for _ in tokens:
            value = decoder.count()
            token = next(i for i in range(3) if bounds[i] <= value < bounds[i+1])
            recovered.append(token); decoder.consume(bounds[token], bounds[token+1])
        self.assertEqual(recovered, tokens)

    @unittest.skipUnless(os.environ.get("MIGRO_TEST_ASSETS") == "1", "Shared checkpoint required")
    def test_cpu_state_handoff_append_and_archive_restore(self):
        import torch
        from migro.memory import MemoryStore, load_core
        torch.set_num_threads(4); torch.set_grad_enabled(False)
        core = load_core("cpu")
        rows = [{"speaker": "Ada", "role": "observed_human", "body": body,
                 "session": "s1", "date": "1 January, 2026", "dialogue_id": f"s1:{i}"}
                for i, body in enumerate(["I moved to Kyoto.", "My favorite color is blue.",
                                          "I now live in Osaka."])]
        store = MemoryStore.create(core); store.observe(rows[:2])
        original = store.state.serialize(); receiver = store.handoff()
        old_ids = [receiver.state.mem.read_slot(i, False).tolist() for i in range(2)]
        receiver.observe(rows[2:])
        self.assertEqual(store.state.serialize(), original)
        self.assertEqual(receiver.observations, 3)
        self.assertEqual(old_ids, [receiver.state.mem.read_slot(i, False).tolist() for i in range(2)])
        receiver.compact(); restored = MemoryStore.from_snapshot(core, receiver.snapshot())
        self.assertEqual(restored.state.meta, receiver.state.meta)
        for i in range(3):
            self.assertTrue(torch.equal(restored.state.mem.read_slot(i, False),
                                        receiver.state.mem.read_slot(i, False)))

if __name__ == "__main__":
    unittest.main()
