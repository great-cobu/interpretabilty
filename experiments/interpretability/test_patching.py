"""Small-model correctness checks; no downloads required."""
import unittest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from patching import forward


class PatchingTests(unittest.TestCase):
    def test_interventions(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=16,
            n_positions=16, vocab_size=30, bos_token_id=0, eos_token_id=0, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)).eval()
        x = torch.tensor([[1, 2, 3], [1, 4, 3]])
        base, cache = forward(model, x, capture=True)
        control, _ = forward(model, x, (0, [2, 2], cache[0]))
        torch.testing.assert_close(base, control, rtol=0, atol=0)
        other = x.clone(); other[:, 1] = 5
        other_logits, _ = forward(model, other)
        self.assertFalse(torch.equal(base, other_logits))
        restored, _ = forward(model, other, (1, [2, 2], cache[1]))
        torch.testing.assert_close(base, restored)
        self.assertTrue(all(not block._forward_hooks for block in model.transformer.h))


if __name__ == '__main__':
    unittest.main()
