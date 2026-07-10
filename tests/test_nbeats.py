"""Tests for the hand-built N-BEATS building block."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.nbeats.nbeats_block import NBeatsBlock


@pytest.fixture
def block() -> NBeatsBlock:
    torch.manual_seed(7)
    return NBeatsBlock(
        input_size=12,
        output_size=6,
        hidden_layer_widths=[16, 8],
        theta_dim=4,
    )


def test_forward_returns_expected_backcast_and_forecast_shapes(block: NBeatsBlock) -> None:
    x = torch.randn(5, 12)

    backcast, forecast = block(x)

    assert backcast.shape == (5, 12)
    assert forecast.shape == (5, 6)
    assert torch.isfinite(backcast).all()
    assert torch.isfinite(forecast).all()


def test_each_batch_item_is_processed_independently(block: NBeatsBlock) -> None:
    x = torch.randn(3, 12)

    batch_backcast, batch_forecast = block(x)
    single_backcast, single_forecast = block(x[:1])

    torch.testing.assert_close(batch_backcast[:1], single_backcast)
    torch.testing.assert_close(batch_forecast[:1], single_forecast)


def test_backward_reaches_input_and_every_parameter(block: NBeatsBlock) -> None:
    x = torch.randn(4, 12, requires_grad=True)

    backcast, forecast = block(x)
    (backcast.square().mean() + forecast.square().mean()).backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient for {name}"


def test_backcast_can_be_removed_from_input_to_form_next_residual(block: NBeatsBlock) -> None:
    x = torch.randn(2, 12)

    backcast, _ = block(x)
    residual = x - backcast

    assert residual.shape == x.shape
    torch.testing.assert_close(residual + backcast, x)


def test_block_supports_no_hidden_layers() -> None:
    block = NBeatsBlock(
        input_size=8,
        output_size=3,
        hidden_layer_widths=[],
        theta_dim=2,
    )

    backcast, forecast = block(torch.randn(4, 8))

    assert backcast.shape == (4, 8)
    assert forecast.shape == (4, 3)


def test_dtype_conversion_applies_to_outputs(block: NBeatsBlock) -> None:
    block = block.double()
    x = torch.randn(2, 12, dtype=torch.float64)

    backcast, forecast = block(x)

    assert backcast.dtype == torch.float64
    assert forecast.dtype == torch.float64


def test_state_dict_round_trip_preserves_predictions(block: NBeatsBlock) -> None:
    x = torch.randn(2, 12)
    expected_backcast, expected_forecast = block(x)

    restored = NBeatsBlock(
        input_size=12,
        output_size=6,
        hidden_layer_widths=[16, 8],
        theta_dim=4,
    )
    restored.load_state_dict(block.state_dict())
    actual_backcast, actual_forecast = restored(x)

    torch.testing.assert_close(actual_backcast, expected_backcast)
    torch.testing.assert_close(actual_forecast, expected_forecast)


def test_wrong_input_width_is_rejected(block: NBeatsBlock) -> None:
    with pytest.raises(RuntimeError):
        block(torch.randn(2, 11))
