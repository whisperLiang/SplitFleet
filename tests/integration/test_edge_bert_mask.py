"""The capture-stable mask must preserve full native BERT semantics."""

import pytest
import torch

from experiments.unified_multitask.edge_models import BertTextClassifier


def test_full_bert_additive_mask_matches_native_padded_and_unpadded_training():
    pytest.importorskip("transformers")
    torch.set_num_threads(1)
    model = BertTextClassifier({})
    ids = torch.arange(2, 10).unsqueeze(0).repeat(2, 1)
    for padded in (False, True):
        mask = torch.ones_like(ids)
        if padded:
            mask[:, 5:] = 0
        for training in (False, True):
            model.train(training)
            torch.manual_seed(8232)
            with torch.no_grad():
                native = model.model(input_ids=ids, attention_mask=mask, return_dict=False)[0]
            torch.manual_seed(8232)
            with torch.no_grad():
                wrapped = model(ids, mask)["logits"]
            torch.testing.assert_close(native, wrapped, rtol=2e-4, atol=2e-6)
