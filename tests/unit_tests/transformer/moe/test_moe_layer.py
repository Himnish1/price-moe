# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestMoELayerInit:
    def setup_method(self, method):
        pass

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_te_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        moe_layer = MoELayer(self.transformer_config, submodules.mlp.submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_legacy_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            add_bias_linear=False,
        )
        transformer_layer_submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        moe_layer = MoELayer(self.transformer_config, transformer_layer_submodules.mlp.submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.skip(
        "Late init of parallel_state was broken after parallel states refactor MR2988."
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["alltoall", "allgather"])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2)])
    def test_moe_with_late_initialize(
        self, moe_token_dispatcher_type, grouped_gemm, tp_size, ep_size
    ):
        num_moe_experts = 4
        hidden_size = 12
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            add_bias_linear=False,
            moe_grouped_gemm=grouped_gemm,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )

        # Fake initialization as NeMo does
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        moe_layer = MoELayer(transformer_config, submodules.mlp.submodules).cuda()

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        input_data = torch.randn(
            16, 4, hidden_size, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        output = moe_layer(input_data)

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestInterleaveTransformerBlock:

    @pytest.mark.parametrize("moe_layer_freq", [2, eval("[0,1,1,1]"), eval("[0]*2+[1]*2")])
    def test_interleave_transformer_block(self, moe_layer_freq):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            moe_layer_freq=moe_layer_freq,
            moe_ffn_hidden_size=256,
            use_cpu_initialization=True,
            num_moe_experts=2,
            add_bias_linear=False,
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_decoder_block_spec(self.transformer_config, False)
        )

        # Check if the moe layer is interleaved correctly
        if isinstance(self.transformer_config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % self.transformer_config.moe_layer_freq == 0) else 0
                for i in range(self.transformer_config.num_layers)
            ]
        else:
            moe_layer_pattern = self.transformer_config.moe_layer_freq

        for i, layer in enumerate(self.parallel_transformer_block.layers):
            is_moe_layer = isinstance(layer.mlp, MoELayer)
            assert is_moe_layer == moe_layer_pattern[i]

        # Test forward pass
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerFP16:
    """Test MoE layer with FP16 precision."""

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2), (4, 2)])
    def test_moe_layer_fp16_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, tp_size, ep_size
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,  # Use SequentialMLP for fp16 test
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp16=True,
            params_dtype=torch.float16,
        )

        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )

        moe_layer = MoELayer(transformer_config, submodules.mlp.submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
            requires_grad=True,
        )

        # Forward pass
        output, _ = moe_layer(hidden_states)

        assert output.dtype == torch.float16, f"Expected fp16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.float16
        ), f"Expected fp16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerRecompute:
    """Test MoE layer with recompute enabled (activation checkpointing).

    Tests both code paths:
    - fp8=False: uses tensor_parallel.checkpoint
    - fp8=True: uses te_checkpoint (requires TE >= 1.7.0)
    """

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("with_padding_mask", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (4, 2)])
    @pytest.mark.parametrize("fp8", [False, True])
    def test_moe_layer_recompute_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, with_padding_mask, tp_size, ep_size, fp8
    ):
        """Test MoE layer forward and backward pass with recompute enabled.

        When fp8=False, uses tensor_parallel.checkpoint.
        When fp8=True, uses te_checkpoint (requires TE >= 1.7.0).
        """
        # Skip fp8 tests if TE version is not sufficient
        if fp8 and not is_te_min_version("1.7.0.dev0"):
            pytest.skip("FP8 MoE recompute requires TE 1.7.0 and later.")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            # Enable recompute for MoE layer
            recompute_granularity="selective",
            recompute_modules=["moe"],
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp8=fp8,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        # Use TE spec for fp8, local spec otherwise
        if fp8:
            transformer_layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=False
            )
        else:
            transformer_layer_submodules = get_gpt_layer_local_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=False
            )

        moe_layer = MoELayer(transformer_config, transformer_layer_submodules.mlp.submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Create padding mask if needed: shape [batch_size, sequence_length]
        padding_mask = None
        if with_padding_mask:
            padding_mask = torch.ones(
                micro_batch_size,
                sequence_length,
                device=torch.cuda.current_device(),
                dtype=torch.bool,
            )
            # Mark last 4 tokens as padding for each batch
            padding_mask[:, -4:] = False

        output, _ = moe_layer(hidden_states, padding_mask=padding_mask)

        assert output.dtype == torch.bfloat16, f"Expected bf16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass - this is where recompute/checkpoint is actually used
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.bfloat16
        ), f"Expected bf16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerRoutingStatsLogging:
    """Tests for MoELayer WandB routing stats logging helpers."""

    class _DummyLayer:
        _get_step = MoELayer._get_step

    @pytest.mark.internal
    def test_get_step_handles_tensor_and_missing_values(self):
        layer = self._DummyLayer()

        layer.router = SimpleNamespace(cp_steps=torch.tensor([150]))
        assert layer._get_step() == 150

        # scalar (0-d) tensor, as registered by CapacityPricedRouter
        layer.router = SimpleNamespace(cp_steps=torch.tensor(200))
        assert layer._get_step() == 200

        layer.router = SimpleNamespace(cp_steps=torch.tensor([]))
        assert layer._get_step() is None

        layer.router = SimpleNamespace(cp_steps="200")
        assert layer._get_step() == 200

        layer.router = SimpleNamespace(cp_steps="abc")
        assert layer._get_step() is None

        layer.router = None
        assert layer._get_step() is None

    @pytest.mark.internal
    def test_log_routing_stats_to_wandb(self, monkeypatch):
        layer = self._DummyLayer()
        layer.layer_number = 2
        layer.router = SimpleNamespace(cp_steps=torch.tensor([100]))

        wandb_log = Mock()
        fake_wandb = SimpleNamespace(run=object(), log=wandb_log)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        routing_map = torch.tensor(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=torch.int64,
        )

        MoELayer._log_routing_stats(layer, routing_map)

        wandb_log.assert_called_once()
        args, kwargs = wandb_log.call_args
        log_dict = args[0]

        assert kwargs == {"step": 100}
        assert log_dict["routing/layer_2/gini"] == pytest.approx(1.0 / 6.0, rel=1e-6)
        assert log_dict["routing/layer_2/entropy"] == pytest.approx(0.94639463, rel=1e-6)
        assert log_dict["routing/layer_2/usage_expert_0"] == 2.0
        assert log_dict["routing/layer_2/usage_expert_1"] == 1.0
        assert log_dict["routing/layer_2/usage_expert_2"] == 1.0

    @pytest.mark.internal
    def test_log_routing_stats_respects_log_interval(self, monkeypatch):
        layer = self._DummyLayer()
        layer.layer_number = 1
        layer.router = SimpleNamespace(cp_steps=torch.tensor([101]))

        wandb_log = Mock()
        fake_wandb = SimpleNamespace(run=object(), log=wandb_log)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        routing_map = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
        MoELayer._log_routing_stats(layer, routing_map)

        wandb_log.assert_not_called()

    @pytest.mark.internal
    def test_log_routing_stats_skips_when_step_is_unavailable(self, monkeypatch):
        layer = self._DummyLayer()
        layer.layer_number = 1
        layer.router = SimpleNamespace()

        wandb_log = Mock()
        fake_wandb = SimpleNamespace(run=object(), log=wandb_log)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        routing_map = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
        MoELayer._log_routing_stats(layer, routing_map)

        wandb_log.assert_not_called()
    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cp_router_forward_captures_step_for_moe_routing_logging(self, monkeypatch):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        try:
            config = TransformerConfig(
                num_layers=1,
                hidden_size=12,
                num_attention_heads=4,
                num_moe_experts=4,
                use_cpu_initialization=True,
                moe_token_dispatcher_type="alltoall",
                moe_router_load_balancing_type="none",
                moe_router_topk=1,
                moe_capacity_priced_routing=True,
                moe_cp_log_interval=1000,
                add_bias_linear=False,
                bf16=True,
                params_dtype=torch.bfloat16,
            )
            submodules = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
            moe_layer = MoELayer(config, submodules.mlp.submodules).cuda()
            moe_layer.train()
            moe_layer.set_layer_number(1)

            wandb_log = Mock()
            fake_wandb = SimpleNamespace(run=object(), log=wandb_log)
            monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

            with torch.no_grad():
                moe_layer.router.cp_steps.fill_(49)

            hidden_states = torch.randn(
                8,
                2,
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )

            moe_layer(hidden_states)
            assert int(moe_layer.router.cp_steps.item()) == 50
            wandb_log.assert_not_called()

            moe_layer(hidden_states)
            assert int(moe_layer.router.cp_steps.item()) == 51

            routing_calls = [
                call
                for call in wandb_log.call_args_list
                if any(str(k).startswith("routing/layer_1/") for k in call[0][0].keys())
            ]
            assert len(routing_calls) == 1

            args, kwargs = routing_calls[0]
            log_dict = args[0]
            assert kwargs == {"step": 50}
            assert "routing/layer_1/gini" in log_dict
            assert "routing/layer_1/entropy" in log_dict
            assert any(k.startswith("routing/layer_1/usage_expert_") for k in log_dict)
        finally:
            Utils.destroy_model_parallel()
