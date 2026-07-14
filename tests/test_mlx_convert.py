"""PyTorch-to-MLX checkpoint conversion checks."""

from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten

from config import TernaryConfig
from mlx_convert import convert_pytorch_checkpoint, map_pytorch_key
from mlx_model import MLXBitNet, MLXBitNetConfig
from mlx_optim import CMUD
from mlx_train import load_checkpoint
from model import BitNetDeep
from optim import build_cmud
from train import build_arg_parser


@pytest.mark.parametrize("use_rfmoe", [False, True])
def test_pytorch_checkpoint_converts_weights_hashes_optimizer_and_outputs(tmp_path, use_rfmoe) -> None:
    torch.manual_seed(9)
    source_config = TernaryConfig(
        vocab_size=512 if not use_rfmoe else 32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        intermediate_size=16,
        block_size=1,
        path_window_size=4,
        infini_memory_dim=4,
        use_hadamard=False,
        use_engram=True,
        engram_layer_ids=(0,),
        engram_vocab_size=17,
        engram_num_heads=2,
        engram_head_dim=2,
        use_rfmoe=use_rfmoe,
        rfmoe_num_experts=2,
        rfmoe_expert_dim=4,
        rfmoe_rank=2,
        mtp_depth=1,
    )
    source_model = BitNetDeep(source_config)
    source_optimizer = build_cmud(
        source_model,
        lr=1e-3,
        fallback_lr=3e-4,
        weight_decay=0.01,
        eight_bit=True,
    )
    source_scheduler = torch.optim.lr_scheduler.LambdaLR(source_optimizer, lr_lambda=lambda _: 1.0)
    tokens = torch.tensor([[1, 2, 3, 4]])
    segments = torch.zeros_like(tokens)
    source_model(tokens, segment_ids=segments).sum().backward()
    source_optimizer.step()
    source_scheduler.step()
    source_optimizer.zero_grad(set_to_none=True)

    source_path = tmp_path / "source.pt"
    source_optimizer_state = source_optimizer.state_dict()
    parameter_names = {id(parameter): name for name, parameter in source_model.named_parameters()}
    for saved_group, live_group in zip(source_optimizer_state["param_groups"], source_optimizer.param_groups):
        saved_group["param_names"] = [parameter_names[id(parameter)] for parameter in live_group["params"]]
    model_state = dict(reversed(list(source_model.state_dict().items())))
    training_args = vars(build_arg_parser().parse_args([]))
    training_args.update(
        {
            "total_tokens": 4,
            "stage1_ratio": 0.0,
            "stage1_weight_mix_start": 0.25,
            "stage1_activation_mix_start": 0.0,
            "stage1_activation_bits": 8,
            "final_activation_bits": 4,
        }
    )
    torch.save(
        {
            "model": model_state,
            "optimizer": source_optimizer_state,
            "scheduler": source_scheduler.state_dict(),
            "trainer_state": {
                "step": 1,
                "tokens_processed": 4,
                "samples_processed": 1,
                "best_val_loss": 2.0,
            },
            "model_config": asdict(source_config),
            "args": training_args,
        },
        source_path,
    )
    converted_path = convert_pytorch_checkpoint(source_path, tmp_path / "mlx")

    metadata = json.loads(converted_path.with_suffix(".json").read_text(encoding="utf-8"))
    config_values = metadata["model_config"]
    config_values["engram_layer_ids"] = tuple(config_values["engram_layer_ids"])
    converted_model = MLXBitNet(MLXBitNetConfig(**config_values))
    converted_optimizer = CMUD(**metadata["optimizer_config"])
    converted_optimizer.init(converted_model.trainable_parameters())
    trainer_state = load_checkpoint(converted_path, converted_model, converted_optimizer)

    bf16_model = MLXBitNet(MLXBitNetConfig(**config_values))
    bf16_model.set_dtype(mx.bfloat16)
    bf16_optimizer = CMUD(**metadata["optimizer_config"])
    bf16_optimizer.init(bf16_model.trainable_parameters())
    load_checkpoint(converted_path, bf16_model, bf16_optimizer)
    assert dict(tree_flatten(bf16_model.parameters()))["embedding.weight"].dtype == mx.bfloat16

    source_multipliers = source_model.layers[0].engram.multipliers.cpu().numpy()
    converted_multipliers = np.array(converted_model.blocks[0].engram.multipliers)
    assert np.array_equal(converted_multipliers, source_multipliers)
    source_hashes = source_model.layers[0].engram.hash_ids(tokens, segment_ids=segments)
    converted_hashes = converted_model.blocks[0].engram.hash_ids(mx.array(tokens.numpy()), mx.array(segments.numpy()))
    mx.eval(converted_hashes)
    assert np.array_equal(np.array(converted_hashes), source_hashes.cpu().numpy())

    source_model.eval()
    with torch.no_grad():
        expected = source_model(tokens, segment_ids=segments).cpu().numpy()
    actual = converted_model(mx.array(tokens.numpy()), mx.array(segments.numpy()))
    mx.eval(actual)
    assert np.allclose(np.array(actual), expected, rtol=2e-3, atol=2e-3)
    assert trainer_state["step"] == 1

    converted_optimizer_state = dict(tree_flatten(converted_optimizer.state))
    assert converted_optimizer_state["states.0.step"].item() == 1
    assert any(key.endswith("momentum_buffer") for key in converted_optimizer_state)
    if not use_rfmoe:
        assert "states.1.embedding.weight.exp_avg_q" in converted_optimizer_state

    for group_index, group in enumerate(source_optimizer_state["param_groups"]):
        for parameter_id, source_name in zip(group["params"], group["param_names"]):
            source_slot = source_optimizer_state["state"].get(parameter_id, {})
            target_name, squeeze = map_pytorch_key(source_name)
            prefix = f"states.{group_index}.{target_name}."
            for slot_name in ("momentum_buffer", "exp_avg", "exp_avg_q", "exp_avg_scale"):
                if slot_name not in source_slot:
                    continue
                expected_slot = source_slot[slot_name].detach().float().cpu().numpy()
                if squeeze and slot_name in {"momentum_buffer", "exp_avg"}:
                    expected_slot = expected_slot.squeeze(1)
                assert np.allclose(
                    np.array(converted_optimizer_state[prefix + slot_name]),
                    expected_slot,
                    rtol=1e-6,
                    atol=1e-6,
                )

    batch = mx.array(tokens.numpy())
    segment_batch = mx.array(segments.numpy())
    source_model.train()
    source_model(tokens, segment_ids=segments).sum().backward()
    source_optimizer.step()

    loss_and_grad = nn.value_and_grad(
        converted_model,
        lambda input_ids, segment_ids: converted_model(input_ids, segment_ids).sum(),
    )
    loss, gradients = loss_and_grad(batch, segment_batch)
    converted_optimizer.update(converted_model, gradients)
    mx.eval(loss, converted_model.state, converted_optimizer.state)
    assert mx.isfinite(loss).item()
    converted_parameters = dict(tree_flatten(converted_model.parameters()))
    for source_name in ("embed_tokens.weight", "layers.0.infini_attn.qkv.weight"):
        target_name, _ = map_pytorch_key(source_name)
        assert np.allclose(
            np.array(converted_parameters[target_name]),
            source_model.state_dict()[source_name].detach().cpu().numpy(),
            rtol=3e-3,
            atol=3e-3,
        )
