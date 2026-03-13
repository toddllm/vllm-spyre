import torch

from vllm.v1.outputs import SamplerOutput

from vllm_spyre.v1.worker.spyre_input_batch import SamplingRequestState
from vllm_spyre.v1.worker.spyre_model_runner import ChunkedPrefillModelRunner


def _make_request(req_id: str, padding_blocks: int) -> SamplingRequestState:
    request = SamplingRequestState(
        sampling_params=None,  # type: ignore[arg-type]
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
    )
    request.padding_blocks = padding_blocks
    return request


def test_sampled_output_uses_current_step_req_ids():
    runner = object.__new__(ChunkedPrefillModelRunner)
    runner.block_size = 64
    runner.tkv = 512
    runner.requests = {
        "stale": _make_request("stale", padding_blocks=1),
        "current": _make_request("current", padding_blocks=2),
    }

    # Simulate stale batch metadata that should not leak into the output map.
    runner.get_req_id_to_index = lambda is_prefill: {"stale": 0, "current": 1}

    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[42]], dtype=torch.int64),
        logprobs_tensors=None,
    )

    model_output = ChunkedPrefillModelRunner.sampled_output(
        runner,
        sampler_output,
        is_prefill=False,
        req_ids=["current"],
    )

    assert model_output.req_ids == ["current"]
    assert model_output.req_id_to_index == {"current": 0}
    assert model_output.sampled_token_ids == [[42]]
    assert model_output.tkv == 512
    assert model_output.left_padding == {"current": 128}
