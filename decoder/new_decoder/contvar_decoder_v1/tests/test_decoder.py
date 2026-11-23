import pytest
import torch

from contvar_decoder.model import LABELS, PairedGODecoder, classification_loss, deterministic, load_decoder, reverse_loss

torch.set_num_threads(1)


def inputs():
    generator = torch.Generator().manual_seed(9)
    wt = {s: torch.randn(6, 8, generator=generator) for s in ("mut", "global")}
    var = {s: torch.randn(6, 8, generator=generator) for s in ("mut", "global")}
    go = torch.randn(6, 5, generator=generator)
    return wt, var, go


def model(head="multiclass", mode="joint", **kwargs):
    return PairedGODecoder(scale_dims={"mut": 8, "global": 8}, go_dim=5,
                           width=8, heads=2, head=head, pair_mode=mode, **kwargs)


@pytest.mark.parametrize("head", ["parallel", "multiclass"])
@pytest.mark.parametrize("mode", ["score_difference", "joint"])
def test_all_four_learn_classification(head, mode):
    torch.manual_seed(3)
    net = model(head, mode)
    wt, var, go = inputs()
    opt = torch.optim.Adam(net.parameters(), lr=.005)
    with deterministic(net):
        before = classification_loss(net(wt, var, go), torch.tensor([0, 1, 2, 0, 1, 2]), head).item()
        for _ in range(25):
            opt.zero_grad()
            out = net(wt, var, go)
            loss = classification_loss(out, torch.tensor([0, 1, 2, 0, 1, 2]), head)
            loss.backward()
            opt.step()
        after = classification_loss(net(wt, var, go), torch.tensor([0, 1, 2, 0, 1, 2]), head).item()
    assert after < before * .8


@pytest.mark.parametrize("head", ["parallel", "multiclass"])
def test_score_subtraction_is_odd_and_identical_inputs_have_zero_direction(head):
    net = model(head, "score_difference", dropout=.8).train()
    wt, var, go = inputs()
    out, rev = net(wt, var, go), net(var, wt, go)
    torch.testing.assert_close(out["d"], -rev["d"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(net(wt, wt, go)["d"], torch.zeros(6), atol=1e-6, rtol=0)
    torch.testing.assert_close(out["side_scores"]["wt"], net(wt, wt, go)["side_scores"]["wt"])
    if head == "multiclass":
        torch.testing.assert_close(out["scores"], rev["scores"][:, [1, 0, 2]], atol=1e-6, rtol=1e-5)


def test_reverse_penalty_backpropagates_and_dropout_state_is_restored():
    net = model(dropout=.8).train()
    wt, var, go = inputs()
    with deterministic(net):
        a = net(wt, var, go)
        b = net(wt, var, go)
        torch.testing.assert_close(a["d"], b["d"], atol=0, rtol=0)
        loss = reverse_loss(a, net(var, wt, go))
    assert net.training
    assert loss.item() > 0
    loss.backward()
    assert net.class_head.bias.grad.abs().sum() > 0


def test_future_scale_can_be_missing_and_masked_nan_never_affects_predictions():
    torch.manual_seed(4)
    net = PairedGODecoder(scale_dims={"mut": 8, "local_window": 4, "pocket": 3, "global": 8},
                          go_dim=5, width=8, heads=2).eval()
    wt, var, go = inputs()
    missing = net(wt, var, go)
    available = torch.tensor([[True, False, False, True]] * 6)
    wt["local_window"] = torch.full((6, 4), float("nan"))
    var["local_window"] = torch.full((6, 4), float("nan"))
    wt["pocket"], var["pocket"] = torch.randn(6, 3), torch.randn(6, 3)
    masked = net(wt, var, go, available)
    torch.testing.assert_close(missing["logits"], masked["logits"], atol=1e-6, rtol=1e-5)
    assert torch.equal(masked["attention"]["wt"][:, 1:3], torch.zeros(6, 2))
    with pytest.raises(ValueError, match="at least one"):
        net(wt, var, go, torch.zeros(6, 4, dtype=torch.bool))


def test_shared_reader_and_go_query_both_use_the_inputs():
    net = model().eval()
    wt, var, go = inputs()
    original = net(wt, var, go)["logits"]
    assert not torch.allclose(original, net(wt, wt, go)["logits"])
    assert not torch.allclose(original, net(wt, var, -go)["logits"])


def test_reverse_loss_zero_does_not_prove_model_is_informative():
    zero = {"d": torch.zeros(5)}
    assert reverse_loss(zero, zero).item() == 0
    assert reverse_loss({"d": torch.ones(5)}, {"d": torch.ones(5)}).item() == 4


def test_checkpoint_reproduces_predictions_and_rejects_wrong_label_order(tmp_path):
    net = model().eval()
    wt, var, go = inputs()
    checkpoint = dict(format="contvar_go_decoder_starter_v1", model_spec=net.spec,
                      state_dict=net.state_dict(), labels=list(LABELS))
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    loaded, _ = load_decoder(path)
    torch.testing.assert_close(net(wt, var, go)["scores"], loaded(wt, var, go)["scores"], atol=0, rtol=0)
    checkpoint["labels"] = ["NEUTRAL", "LOF", "GOF"]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="label order"):
        load_decoder(path)
