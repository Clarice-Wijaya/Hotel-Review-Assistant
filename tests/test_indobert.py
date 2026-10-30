import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from hra.indobert import AspectModel, class_weights, loss_fn, predict  # noqa: E402


@pytest.fixture
def tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "kamar", "bersih", "kotor"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    cfg = BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32
    )
    return AspectModel(BertModel(cfg)), tokenizer


def test_output_shape(tiny):
    model, tok = tiny
    ids, probs = predict(model, tok, ["kamar bersih", "kamar kotor kamar"], max_length=16)
    assert ids.shape == (2, 7)
    assert probs.shape == (2, 7, 4)
    assert np.allclose(probs.sum(-1), 1, atol=1e-5)


def test_loss_goes_down(tiny):
    model, tok = tiny
    enc = tok(["kamar bersih", "kamar kotor"], padding=True, return_tensors="pt")
    y = torch.zeros((2, 7), dtype=torch.long)
    y[0, 1], y[1, 1] = 1, 2
    w = torch.ones((7, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = loss_fn(model(**enc), y, w).item()
    for _ in range(30):
        loss = loss_fn(model(**enc), y, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first / 2


def test_rare_classes_get_more_weight():
    y = np.zeros((100, 7), dtype=int)
    y[:5, 0] = 3
    w = class_weights(y)
    assert w[0, 3] > w[0, 0]


def test_save_and_load(tiny, tmp_path):
    model, tok = tiny
    model.save(tmp_path, tok, max_length=16)
    loaded, tok2, max_length = AspectModel.load(tmp_path)
    a = predict(model, tok, ["kamar bersih"], 16)[1]
    b = predict(loaded, tok2, ["kamar bersih"], 16)[1]
    assert np.allclose(a, b, atol=1e-6)
    assert max_length == 16
