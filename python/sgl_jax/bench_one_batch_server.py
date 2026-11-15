"""
Benchmark the latency of running a single batch with a server.

This script launches a server and uses the HTTP interface.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

Usage:
python3 -m sgl_jax.bench_one_batch_server --model meta-llama/Meta-Llama-3.1-8B --batch-size 1 16 64 --input-len 1024 --output-len 8

python3 -m sgl_jax.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8
python3 -m sgl_jax.bench_one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --show-report --profile --profile-by-stage
"""

import argparse
import dataclasses
import itertools
import json
import multiprocessing
import os
import time

import numpy as np
import requests

from sgl_jax.bench_serving import (
    get_tokenizer,
    sample_random_requests,
    sample_generated_shared_prefix_requests,
)
from sgl_jax.profiler import run_profile
from sgl_jax.srt.entrypoints import http_server
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import is_in_ci, write_github_step_summary


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: tuple[int] = (1,)
    input_len: tuple[int] = (1024,)
    output_len: tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    client_stream_interval: int = 1
    input_len_step_percentage: float = 0.0
    result_filename: str = "result.jsonl"
    base_url: str = ""
    skip_warmup: bool = False
    skip_flush: bool = False
    show_report: bool = False
    profile: bool = False
    profile_by_stage: bool = False
    api_type: str = "native"  # "native" or "openai"
    dataset_name: str = "random"
    gsp_num_groups: int = 1
    gsp_prompts_per_group: int = 1
    gsp_system_prompt_len: int = 1024
    gsp_question_len: int = 100
    gsp_output_len: int = 16
    gsp_range_ratio: float = 1.0
    seed: int = 1
    gsp_send_routing_key: bool = False
    gsp_question_seed: int = -1  # <0: unset (single random stream)
    gsp_num_turns: int = 1
    gsp_fast_prepare: bool = False
    gsp_ordered: bool = False
    gsp_real_corpus_dir: str = ""
    save_responses_dir: str = ""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument("--batch-size", type=int, nargs="+", default=BenchArgs.batch_size)
        parser.add_argument("--input-len", type=int, nargs="+", default=BenchArgs.input_len)
        parser.add_argument("--output-len", type=int, nargs="+", default=BenchArgs.output_len)
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--client-stream-interval",
            type=int,
            default=BenchArgs.client_stream_interval,
        )
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument("--result-filename", type=str, default=BenchArgs.result_filename)
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument(
            "--skip-flush",
            action="store_true",
            help="Do not POST /flush_cache before each case (keeps radix + HiCache host state "
            "from previous runs, e.g. to measure prefix reuse across invocations).",
        )
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--api-type",
            type=str,
            default=BenchArgs.api_type,
            choices=["native", "openai"],
            help="API type to use: 'native' for /generate or 'openai' for /v1/completions",
        )
        parser.add_argument("--dataset-name", type=str, default=BenchArgs.dataset_name)
        parser.add_argument("--gsp-num-groups", type=int, default=BenchArgs.gsp_num_groups)
        parser.add_argument("--gsp-prompts-per-group", type=int, default=BenchArgs.gsp_prompts_per_group)
        parser.add_argument("--gsp-system-prompt-len", type=int, default=BenchArgs.gsp_system_prompt_len)
        parser.add_argument("--gsp-question-len", type=int, default=BenchArgs.gsp_question_len)
        parser.add_argument("--gsp-output-len", type=int, default=BenchArgs.gsp_output_len)
        parser.add_argument("--gsp-range-ratio", type=float, default=BenchArgs.gsp_range_ratio)
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
        parser.add_argument("--gsp-send-routing-key", action="store_true", default=BenchArgs.gsp_send_routing_key)
        parser.add_argument(
            "--gsp-question-seed",
            type=int,
            default=BenchArgs.gsp_question_seed,
            help="Seed questions separately from system prompts (same --seed => same system prompts across runs).",
        )
        parser.add_argument("--gsp-num-turns", type=int, default=BenchArgs.gsp_num_turns)
        parser.add_argument("--gsp-fast-prepare", action="store_true", default=BenchArgs.gsp_fast_prepare)
        parser.add_argument("--gsp-ordered", action="store_true", default=BenchArgs.gsp_ordered)
        parser.add_argument("--gsp-real-corpus-dir", type=str, default=BenchArgs.gsp_real_corpus_dir)
        parser.add_argument("--save-responses-dir", type=str, default=BenchArgs.save_responses_dir)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        # Use default from dataclass if argument isn't present in args
        return cls(**{attr: attr_type(getattr(args, attr, getattr(cls, attr))) for attr, attr_type in attrs})


def launch_server_internal(server_args):
    try:
        http_server.launch(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process(server_args: ServerArgs):
    proc = multiprocessing.Process(target=launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"
    timeout = 600

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(f"{base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                return proc, base_url
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")


def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    stream_interval: int,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer,
    profile: bool = False,
    profile_by_stage: bool = False,
    api_type: str = "native",
    bench_args: BenchArgs = None,
):
    if not (bench_args is not None and getattr(bench_args, "skip_flush", False)):
        requests.post(url + "/flush_cache")

    # Determine whether to use text or input_ids based on API type
    return_text = api_type == "openai"
    
    if bench_args is not None and getattr(bench_args, "dataset_name", "") == "generated-shared-prefix":
        # Override prompts_per_group so we can test different batch sizes in one run
        dynamic_prompts_per_group = max(1, batch_size // bench_args.gsp_num_groups)
        bench_args.gsp_prompts_per_group = dynamic_prompts_per_group
        
        input_requests = sample_generated_shared_prefix_requests(
            num_groups=bench_args.gsp_num_groups,
            prompts_per_group=dynamic_prompts_per_group,
            system_prompt_len=bench_args.gsp_system_prompt_len,
            question_len=bench_args.gsp_question_len,
            output_len=bench_args.gsp_output_len,
            range_ratio=getattr(bench_args, "gsp_range_ratio", 1.0),
            tokenizer=tokenizer,
            args=bench_args,
        )
        batch_size = len(input_requests)
        input_len = bench_args.gsp_system_prompt_len + bench_args.gsp_question_len
        output_len = bench_args.gsp_output_len
    else:
        input_requests = sample_random_requests(
            input_len=input_len,
            output_len=output_len,
            num_prompts=batch_size,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path="",
            random_sample=True,
            return_text=return_text,
        )

    use_structured_outputs = False
    if use_structured_outputs:
        texts = []
        for _ in range(batch_size):
            texts.append(
                "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                * 50
                + "Assistant:"
            )
        json_schema = "$$ANY$$"
    else:
        json_schema = None

    profile_link = None
    if profile:
        profile_link: str = run_profile(url, 3, ["CPU", "GPU"], None, None, profile_by_stage)

    pre_info = requests.get(url + "/get_server_info").json()
    pre_state = pre_info["internal_states"][0] if pre_info.get("internal_states") else {}
    pre_cum_accept_len = pre_state.get("cum_spec_accept_length", 0)
    pre_cum_accept_cnt = pre_state.get("cum_spec_accept_count", 0)

    generated_texts = [""] * batch_size
    completion_tokens_per_req = [0] * batch_size
    cached_tokens_per_req = [0] * batch_size
    req_ttfts = [0.0] * batch_size
    req_finish_times = [0.0] * batch_size
    ttft_first = 0.0

    tic = time.perf_counter()

    if api_type == "openai":
        # Use OpenAI API - send requests as a batch with n parameter
        # Convert prompts to a single request with n > 1 if batch_size > 1
        if batch_size == 1:
            request_data = {
                "model": "default",
                "prompt": input_requests[0].prompt,
                "max_tokens": output_len,
                "temperature": temperature,
                "logprobs": 1 if return_logprob else None,
                "stream": True,
            }
        else:
            # For batch, we'll send the first prompt with n=batch_size
            # Note: This assumes all prompts should be the same, which may not be ideal
            # For different prompts, we'd need to send separate requests
            request_data = {
                "model": "default",
                "prompt": [req.prompt for req in input_requests],
                "max_tokens": output_len,
                "temperature": temperature,
                "logprobs": 1 if return_logprob else None,
                "stream": True,
            }

        response = requests.post(
            url + "/v1/completions",
            json=request_data,
            stream=True,
        )

        # Parse OpenAI streaming response
        ttft = 0.0
        first_token_received = False
        total_chunks = 0
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                if "error" in data:
                    raise RuntimeError(f"Request has failed. {data}.")

                # Track first token time (across all choices in batch)
                if data.get("choices") and len(data["choices"]) > 0:
                    # Check if any choice has generated text
                    for choice in data["choices"]:
                        if choice.get("text") and not first_token_received:
                            ttft = time.perf_counter() - tic
                            ttft_first = ttft
                            first_token_received = True
                            break
                    total_chunks += 1
    else:
        # Use native API
        json_data = {
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": True,
                "json_schema": json_schema,
                "stream_interval": stream_interval,
            },
            "return_logprob": return_logprob,
            "stream": True,
        }
        if getattr(bench_args, "dataset_name", "") == "generated-shared-prefix":
            json_data["text"] = [req.prompt for req in input_requests]
        else:
            json_data["input_ids"] = [req.prompt for req in input_requests]

        response = requests.post(
            url + "/generate",
            json=json_data,
            stream=True,
            timeout=3600,
        )

        # The TTFT of the last request in the batch (matching 85f4dd1 reference doc)
        ttft = 0.0
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                if "error" in data:
                    raise RuntimeError(f"Request has failed. {data}.")

                assert (
                    data["meta_info"]["finish_reason"] is None
                    or data["meta_info"]["finish_reason"]["type"] == "length"
                )
                now_rel = time.perf_counter() - tic
                idx = data.get("index", 0)
                if 0 <= idx < batch_size:
                    if "text" in data:
                        generated_texts[idx] = data["text"]
                    completion_tokens_per_req[idx] = data["meta_info"].get("completion_tokens", 0)
                    cached_tokens_per_req[idx] = data["meta_info"].get("cached_tokens", 0)
                    if req_ttfts[idx] == 0.0 and completion_tokens_per_req[idx] >= 1:
                        req_ttfts[idx] = now_rel
                    req_finish_times[idx] = now_rel
                comp_toks = data["meta_info"].get("completion_tokens", 0)
                if comp_toks >= 1 and ttft_first == 0.0:
                    ttft_first = now_rel
                if comp_toks == 1:
                    ttft = now_rel
        if ttft == 0.0 and ttft_first > 0.0:
            ttft = ttft_first

    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / ttft
    output_throughput = batch_size * output_len / (latency - ttft)
    true_decode_window_throughput = (
        batch_size * output_len / (latency - ttft_first) if (latency - ttft_first) > 0 else 0.0
    )
    overall_throughput = batch_size * (input_len + output_len) / latency

    server_info = requests.get(url + "/get_server_info").json()
    post_state = server_info["internal_states"][0] if server_info.get("internal_states") else {}
    cum_acc_length = post_state.get("avg_spec_accept_length", None)
    post_cum_accept_len = post_state.get("cum_spec_accept_length", 0)
    post_cum_accept_cnt = post_state.get("cum_spec_accept_count", 0)
    delta_len = post_cum_accept_len - pre_cum_accept_len
    delta_cnt = post_cum_accept_cnt - pre_cum_accept_cnt
    acc_length = (delta_len / delta_cnt) if delta_cnt > 0 else cum_acc_length
    spec_num_draft_tokens = int(server_info.get("speculative_num_draft_tokens", 4) or 4)
    spec_num_steps = int(server_info.get("speculative_num_steps", 3) or 3)
    acc_rate = (acc_length / spec_num_draft_tokens) if (acc_length is not None and spec_num_draft_tokens > 0) else None
    draft_token_acc_rate = (
        max(0.0, (acc_length - 1.0) / spec_num_steps)
        if (acc_length is not None and spec_num_steps > 0)
        else None
    )
    mean_verify_step_ms = (
        (latency - ttft) * 1000.0 / (output_len / acc_length)
        if (acc_length is not None and acc_length > 0 and output_len > 0)
        else None
    )
    last_gen_throughput = post_state.get("last_gen_throughput", 0.0)

    valid_ttfts = [t for t in req_ttfts if t > 0] or [ttft]
    ttft_mean = float(np.mean(valid_ttfts))
    ttft_p50 = float(np.percentile(valid_ttfts, 50))
    ttft_p90 = float(np.percentile(valid_ttfts, 90))
    ttft_p99 = float(np.percentile(valid_ttfts, 99))

    tpots_ms = [
        (fin - first_t) * 1000.0 / max(1, output_len - 1)
        for first_t, fin in zip(req_ttfts, req_finish_times)
        if first_t > 0 and fin > first_t
    ]
    tpot_mean_ms = float(np.mean(tpots_ms)) if tpots_ms else (1000.0 / (output_throughput / batch_size))
    tpot_p50_ms = float(np.percentile(tpots_ms, 50)) if tpots_ms else tpot_mean_ms
    tpot_p90_ms = float(np.percentile(tpots_ms, 90)) if tpots_ms else tpot_mean_ms
    cached_tokens_mean = float(np.mean(cached_tokens_per_req)) if cached_tokens_per_req else 0.0
    cache_hit_rate = cached_tokens_mean / input_len if input_len > 0 else 0.0
    num_groups_val = getattr(bench_args, "gsp_num_groups", 1) if bench_args is not None else 1
    prompts_per_group_val = max(1, batch_size // num_groups_val)

    print(f"batch size: {batch_size} (num_groups: {num_groups_val}, prompts_per_group: {prompts_per_group_val})")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"ttft: {ttft:.2f} s (ttft_first: {ttft_first:.2f} s, p50: {ttft_p50:.2f} s, p90: {ttft_p90:.2f} s)")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s (true decode window: {true_decode_window_throughput:.2f} tok/s)")
    if acc_length is not None:
        print(
            f"spec accept length (batch): {acc_length:.2f} / {spec_num_draft_tokens} "
            f"(acc_rate: {acc_rate * 100:.1f}%, draft_only_rate: {draft_token_acc_rate * 100:.1f}%, "
            f"cumulative: {cum_acc_length:.2f}, steps: {delta_cnt}, step_ms: {mean_verify_step_ms:.2f} ms)"
        )

    save_dir = getattr(bench_args, "save_responses_dir", "") if bench_args is not None else ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        resp_path = os.path.join(save_dir, f"responses_bs{batch_size}.json")
        records = []
        for i in range(batch_size):
            prompt_str = input_requests[i].prompt if isinstance(input_requests[i].prompt, str) else ""
            records.append(
                {
                    "request_index": i,
                    "prompt_len_tokens": input_requests[i].prompt_len,
                    "output_len_target": input_requests[i].output_len,
                    "completion_tokens": completion_tokens_per_req[i],
                    "cached_tokens": cached_tokens_per_req[i],
                    "req_ttft_s": round(req_ttfts[i], 4),
                    "req_finish_s": round(req_finish_times[i], 4),
                    "prompt_tail_600_chars": prompt_str[-600:] if prompt_str else "",
                    "generated_text": generated_texts[i],
                }
            )
        with open(resp_path, "w", encoding="utf-8") as f_resp:
            json.dump(
                {
                    "run_name": run_name,
                    "num_groups": num_groups_val,
                    "prompts_per_group": prompts_per_group_val,
                    "batch_size": batch_size,
                    "input_len": input_len,
                    "output_len": output_len,
                    "latency_s": round(latency, 4),
                    "ttft_s": round(ttft, 4),
                    "ttft_first_s": round(ttft_first, 4),
                    "ttft_mean_s": round(ttft_mean, 4),
                    "ttft_p50_s": round(ttft_p50, 4),
                    "ttft_p90_s": round(ttft_p90, 4),
                    "ttft_p99_s": round(ttft_p99, 4),
                    "input_throughput_tok_s": round(input_throughput, 2),
                    "output_throughput_tok_s": round(output_throughput, 2),
                    "true_decode_window_throughput_tok_s": round(true_decode_window_throughput, 2),
                    "overall_throughput_tok_s": round(overall_throughput, 2),
                    "last_gen_throughput_tok_s": round(last_gen_throughput, 2),
                    "batch_acc_length": round(acc_length, 4) if acc_length is not None else None,
                    "acc_rate": round(acc_rate, 4) if acc_rate is not None else None,
                    "draft_token_acc_rate": round(draft_token_acc_rate, 4) if draft_token_acc_rate is not None else None,
                    "cum_acc_length": round(cum_acc_length, 4) if cum_acc_length is not None else None,
                    "decode_verify_steps": int(delta_cnt),
                    "mean_verify_step_ms": round(mean_verify_step_ms, 2) if mean_verify_step_ms is not None else None,
                    "tpot_mean_ms": round(tpot_mean_ms, 2),
                    "tpot_p50_ms": round(tpot_p50_ms, 2),
                    "tpot_p90_ms": round(tpot_p90_ms, 2),
                    "cached_tokens_mean": round(cached_tokens_mean, 1),
                    "cache_hit_rate": round(cache_hit_rate, 4),
                    "responses": records,
                },
                f_resp,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved {len(records)} generated responses to {resp_path}")

    if result_filename:
        with open(result_filename, "a") as fout:
            res = {
                "run_name": run_name,
                "num_groups": num_groups_val,
                "prompts_per_group": prompts_per_group_val,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "latency": round(latency, 4),
                "ttft": round(ttft, 4),
                "ttft_first": round(ttft_first, 4),
                "ttft_mean": round(ttft_mean, 4),
                "ttft_p50": round(ttft_p50, 4),
                "ttft_p90": round(ttft_p90, 4),
                "ttft_p99": round(ttft_p99, 4),
                "input_throughput": round(input_throughput, 2),
                "output_throughput": round(output_throughput, 2),
                "true_decode_window_throughput": round(true_decode_window_throughput, 2),
                "overall_throughput": round(overall_throughput, 2),
                "last_gen_throughput": round(last_gen_throughput, 2),
                "acc_length": round(acc_length, 4) if acc_length is not None else None,
                "acc_rate": round(acc_rate, 4) if acc_rate is not None else None,
                "draft_token_acc_rate": round(draft_token_acc_rate, 4) if draft_token_acc_rate is not None else None,
                "cum_acc_length": round(cum_acc_length, 4) if cum_acc_length is not None else None,
                "decode_verify_steps": int(delta_cnt),
                "mean_verify_step_ms": round(mean_verify_step_ms, 2) if mean_verify_step_ms is not None else None,
                "itl_ms": round(1000.0 / (output_throughput / batch_size), 2) if output_throughput > 0 else None,
                "tpot_mean_ms": round(tpot_mean_ms, 2),
                "tpot_p50_ms": round(tpot_p50_ms, 2),
                "tpot_p90_ms": round(tpot_p90_ms, 2),
                "cached_tokens_mean": round(cached_tokens_mean, 1),
                "cache_hit_rate": round(cache_hit_rate, 4),
            }
            fout.write(json.dumps(res) + "\n")

    return (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        profile_link if profile else None,
    )


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(server_args)

    server_info = requests.get(base_url + "/get_server_info").json()
    if "tokenizer_path" in server_info:
        tokenizer_path = server_info["tokenizer_path"]
    elif "prefill" in server_info:
        tokenizer_path = server_info["prefill"][0]["tokenizer_path"]
    tokenizer = get_tokenizer(tokenizer_path)

    # warmup
    if not bench_args.skip_warmup:
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        run_one_case(
            base_url,
            batch_size=16,
            input_len=1024,
            output_len=16,
            temperature=bench_args.temperature,
            return_logprob=bench_args.return_logprob,
            stream_interval=bench_args.client_stream_interval,
            input_len_step_percentage=bench_args.input_len_step_percentage,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
            api_type=bench_args.api_type,
            bench_args=bench_args,
        )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    # benchmark
    result = []
    bench_result = []
    try:
        for bs, il, ol in itertools.product(
            bench_args.batch_size, bench_args.input_len, bench_args.output_len
        ):
            result.append(
                run_one_case(
                    base_url,
                    bs,
                    il,
                    ol,
                    temperature=bench_args.temperature,
                    return_logprob=bench_args.return_logprob,
                    stream_interval=bench_args.client_stream_interval,
                    input_len_step_percentage=bench_args.input_len_step_percentage,
                    run_name=bench_args.run_name,
                    result_filename=bench_args.result_filename,
                    tokenizer=tokenizer,
                    api_type=bench_args.api_type,
                    bench_args=bench_args,
                )
            )

        if bench_args.profile:
            try:
                for bs, il, ol in itertools.product(
                    bench_args.batch_size, bench_args.input_len, bench_args.output_len
                ):
                    bench_result.append(
                        (
                            run_one_case(
                                base_url,
                                bs,
                                il,
                                ol,
                                temperature=bench_args.temperature,
                                return_logprob=bench_args.return_logprob,
                                stream_interval=bench_args.client_stream_interval,
                                input_len_step_percentage=bench_args.input_len_step_percentage,
                                run_name=bench_args.run_name,
                                result_filename=bench_args.result_filename,
                                tokenizer=tokenizer,
                                profile=bench_args.profile,
                                profile_by_stage=bench_args.profile_by_stage,
                                api_type=bench_args.api_type,
                                bench_args=bench_args,
                            )[-1],
                        )
                    )
                result = [t1[:-1] + t2 for t1, t2 in zip(result, bench_result)]
            except Exception as e:
                print(f"Error profiling, there will be no profile trace dump: {e}")
    finally:
        if proc:
            kill_process_tree(proc.pid)

    print(f"\nResults are saved to {bench_args.result_filename}")

    if not bench_args.show_report:
        return

    if bench_args.dataset_name == "generated-shared-prefix":
        eff_input_len = bench_args.gsp_system_prompt_len + bench_args.gsp_question_len
        eff_output_len = bench_args.gsp_output_len
        summary = f"\nInput lens: ({eff_input_len},). Output lens: ({eff_output_len},).\n"
    else:
        eff_input_len = bench_args.input_len[0] if bench_args.input_len else 1024
        eff_output_len = bench_args.output_len[0] if bench_args.output_len else 16
        summary = f"\nInput lens: {bench_args.input_len}. Output lens: {bench_args.output_len}.\n"

    summary += "| batch size | latency (s) | ttft (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) | req cost ($) | e2e req cost ($) |"

    if bench_args.profile:
        summary += " profile |"

    summary += "\n"
    summary += "| ---------- | ----------- | -------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ | ------------ | ---------------- |"

    if bench_args.profile:
        summary += "-------------|"
    summary += "\n"

    actual_tp_size = server_info.get("tp_size", server_args.tp_size) or 32
    num_tpu7x_chips = (actual_tp_size // 2) if actual_tp_size > 1 else 16
    tpu7x_chip_hourly_cost = 12.0  # $12.00/hour per TPU v7x chip (On-Demand)
    hourly_cost = num_tpu7x_chips * tpu7x_chip_hourly_cost  # $192.00/hour for 16 TPU v7x chips

    for (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        trace_link,
    ) in result:
        input_util = 0.7
        accept_length = round(acc_length, 2) if acc_length is not None else "n/a"
        input_cost_per_1m = 1e6 / (input_throughput * input_util) / 3600 * hourly_cost
        output_cost_per_1m = 1e6 / output_throughput / 3600 * hourly_cost
        req_cost = (eff_input_len / 1e6) * input_cost_per_1m + (eff_output_len / 1e6) * output_cost_per_1m
        e2e_req_cost = (latency / batch_size) / 3600 * hourly_cost
        line = (
            f"| {batch_size} | "
            f"{latency:.2f} | "
            f"{ttft:.2f} | "
            f"{input_throughput:.2f} | "
            f"{output_throughput:.2f} | "
            f"{accept_length} | "
            f"{1 / (output_throughput / batch_size) * 1000:.2f} | "
            f"{input_cost_per_1m:.2f} | "
            f"{output_cost_per_1m:.2f} | "
            f"{req_cost:.4f} | "
            f"{e2e_req_cost:.4f} |"
        )
        if trace_link:
            line += f" [Profile]({trace_link}) |"
        line += "\n"
        summary += line

    # print metrics table
    print(summary)

    if is_in_ci():
        write_github_step_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)
