"""Run with `python -m dllm.pipelines.tie.eval --model tie --tasks gsm8k_cot`."""

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

if __name__ == "__main__":
    print("dLLM loaded. Importing the lm-eval adapter.", flush=True)

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import torch

from dllm.core.eval.running_metrics import batch_running_accuracy
from dllm.utils import ModelArguments, get_model, get_tokenizer

from .action_fusion import ACTION_CHECKPOINTS, ActionFusionDiagnostic
from .commit_actions import CommitExperiment
from .commit_support import MODE as COMMIT_SUPPORT_MODE, REPLICATION_MODE, CommitSupport
from .exact_text_fusion import MODE as EXACT_TEXT_FUSION_MODE, ExactTextFusion
from .final_answer_fusion import load_saved_answers, score_saved_answers
from .peer_block_proposals import BLOCK_STARTS, PeerBlockProposals
from .periodic_js_ensemble import MODE as PERIODIC_JS_MODE, PeriodicJSEnsemble
from .periodic_js_ensemble import ROUTING_CONFIDENCE_THRESHOLD, SELECTIVE_MODE as SELECTIVE_JS_MODE
from .sampler import TIEMember, TIESampler, TIESamplerConfig
from .shared_audit import CHECKPOINTS, SharedCanvasAudit
from .short_action_fusion import ShortActionFusionDiagnostic, load_full_rollouts

COMMIT_MODES = {
    "diagnostic", "schedule_diagnostic", "exact", "llada_verifier", "text",
    "positions", "joint", "ctca",
}
ACTION_FUSION_MODES = {"action_fusion", "action_fusion_wide"}
SHORT_ACTION_FUSION_MODE = "action_fusion_short"
FRESH_ACTION_FUSION_MODE = "action_fusion_short_fresh"
SHORT_ACTION_FUSION_MODES = {SHORT_ACTION_FUSION_MODE, FRESH_ACTION_FUSION_MODE}
FRESH_GSM8K_TASK = "gsm8k_cot_train_dev"
PEER_BLOCK_MODE = "peer_block_proposals"
FINAL_ANSWER_FUSION_MODES = {"final_answer_fusion", "final_answer_fusion_extract"}
POSITION_STUDY_MODES = {COMMIT_SUPPORT_MODE, REPLICATION_MODE}
JS_ENSEMBLE_MODES = {PERIODIC_JS_MODE, SELECTIVE_JS_MODE}
TRAIN_DEV_MODES = {FRESH_ACTION_FUSION_MODE} | POSITION_STUDY_MODES | JS_ENSEMBLE_MODES

@register_model("tie")
class TIEEvalHarness(LM):
    def __init__(
        self,
        mode="relay",
        llada="GSAI-ML/LLaDA-8B-Instruct",
        dream="Dream-org/Dream-v0-Instruct-7B",
        device="cuda:0",
        llada_device=None,
        dream_device=None,
        dtype="bfloat16",
        batch_size=1,
        max_batch_size=None,
        max_new_tokens=256,
        answer_tokens=16,
        block_size=16,
        steps=128,
        relay_interval=16,
        trace_path=None,
        policy_path=None,
        doc_ids_path=None,
        multi_model="llada",
        sample_count=2,
        progress_log_interval=0,
        audit_checkpoints=None,
        source_trace_path=None,
        source_samples_path=None,
        routing_confidence_threshold=ROUTING_CONFIDENCE_THRESHOLD,
        js_owner="llada",
    ):
        super().__init__()
        modes = {"relay", "post_generation", "llada", "dream", "multi", "shared_audit", PEER_BLOCK_MODE}
        modes |= (
            COMMIT_MODES | ACTION_FUSION_MODES | FINAL_ANSWER_FUSION_MODES
            | SHORT_ACTION_FUSION_MODES | POSITION_STUDY_MODES | JS_ENSEMBLE_MODES | {EXACT_TEXT_FUSION_MODE}
        )
        if mode not in modes:
            raise ValueError(f"Unsupported TIE experiment mode: {mode}")
        if multi_model not in {"llada", "dream"} or int(sample_count) < 1:
            raise ValueError("Select a native model and a positive sample count")
        if mode in JS_ENSEMBLE_MODES and js_owner not in {"llada", "dream"}:
            raise ValueError("js_owner must be llada or dream")
        self.mode = mode
        self.routing_confidence_threshold = None
        if mode == SELECTIVE_JS_MODE:
            self.routing_confidence_threshold = float(routing_confidence_threshold)
            if not 0 <= self.routing_confidence_threshold <= 1:
                raise ValueError("routing_confidence_threshold must be between 0 and 1")
        uses_action_checkpoints = mode in ACTION_FUSION_MODES | SHORT_ACTION_FUSION_MODES
        configured_checkpoints = ACTION_CHECKPOINTS if uses_action_checkpoints else CHECKPOINTS
        if mode == PEER_BLOCK_MODE:
            configured_checkpoints = BLOCK_STARTS
        if audit_checkpoints is not None:
            if tuple(map(int, audit_checkpoints.split(":"))) != configured_checkpoints:
                raise ValueError("audit_checkpoints must match the configured schedule")
        task = FRESH_GSM8K_TASK if mode in TRAIN_DEV_MODES else "gsm8k_cot"
        self.doc_ids = json.loads(Path(doc_ids_path).read_text())[task] if doc_ids_path else None
        if mode in TRAIN_DEV_MODES and not self.doc_ids:
            raise ValueError(f"{mode} needs the fixed development split")
        self.saved_answers = None
        self.full_rollouts = None
        if mode in FINAL_ANSWER_FUSION_MODES:
            if not (source_trace_path and source_samples_path and self.doc_ids):
                raise ValueError("Final-answer fusion needs a split, source trace, and source samples")
            self.saved_answers = load_saved_answers(source_trace_path, source_samples_path, self.doc_ids)
        if mode == SHORT_ACTION_FUSION_MODE:
            if not source_trace_path or not self.doc_ids:
                raise ValueError("Short action fusion needs a split and saved full rollouts")
            self.full_rollouts = load_full_rollouts(source_trace_path, self.doc_ids)
        self.sample_count = int(sample_count)
        self.progress_log_interval = int(progress_log_interval)
        if self.progress_log_interval < 0:
            raise ValueError("progress_log_interval must be nonnegative")
        self.policy = json.loads(Path(policy_path).read_text()) if policy_path else {}
        if mode in {"text", "llada_verifier", "positions", "joint", "schedule_diagnostic"}:
            if not self.policy:
                raise ValueError("This experiment arm requires a calibrated policy_path")
        if mode in {"positions", "joint", "schedule_diagnostic"} and not self.policy.get("span_pass"):
            raise ValueError("Scheduling experiments require a passing span gate")
        if str(batch_size or 1) != "1":
            raise ValueError("TIE evaluation currently handles one question at a time")
        config = TIESamplerConfig(
            max_new_tokens=int(max_new_tokens),
            answer_tokens=int(answer_tokens),
            block_size=int(block_size),
            steps=int(steps),
            relay_interval=int(relay_interval) if mode == "relay" else 0,
        )
        members = []
        for name, checkpoint, placement in (
            ("llada", llada, llada_device or device),
            ("dream", dream, dream_device or device),
        ):
            selected_member = multi_model if mode == "multi" else mode
            if selected_member in {"llada", "dream"} and name != selected_member:
                continue
            args = ModelArguments(model_name_or_path=checkpoint, dtype=dtype)
            started = perf_counter()
            print(f"[{mode}] Loading {name} tokenizer from {checkpoint}", flush=True)
            tokenizer = get_tokenizer(args)
            print(f"[{mode}] Loading {name} model on {placement} ({dtype})", flush=True)
            model = get_model(args, device_map={"": placement}).eval()
            members.append(
                TIEMember(name, model, tokenizer, right_shift_logits=name == "dream")
            )
            print(f"[{mode}] {name} ready after {perf_counter() - started:.1f}s", flush=True)
        self.sampler = TIESampler(members, config)
        self.peer_blocks = PeerBlockProposals(self.sampler) if mode == PEER_BLOCK_MODE else None
        self.exact_fusion = ExactTextFusion(self.sampler) if mode == EXACT_TEXT_FUSION_MODE else None
        self.periodic_js = (
            PeriodicJSEnsemble(self.sampler, owner_index=tuple(member.name for member in members).index(js_owner))
            if mode in JS_ENSEMBLE_MODES else None
        )
        self.commit_support = (
            CommitSupport(self.sampler, replication=mode == REPLICATION_MODE)
            if mode in POSITION_STUDY_MODES else None
        )
        self.shared_audit = SharedCanvasAudit(self.sampler) if mode == "shared_audit" else None
        self.action_fusion = (
            ActionFusionDiagnostic(self.sampler, wide=mode == "action_fusion_wide")
            if mode in ACTION_FUSION_MODES else None
        )
        self.short_fusion = (
            ShortActionFusionDiagnostic(self.sampler) if mode in SHORT_ACTION_FUSION_MODES else None
        )
        self.experiment = None
        if mode in COMMIT_MODES:
            prefer = None
            if mode == "ctca":
                from .ctca_control import create_ctca_control
                print("[ctca] Preparing the vocabulary projection", flush=True)
                prefer = create_ctca_control(*members)
            self.experiment = CommitExperiment(self.sampler, mode, self.policy, prefer=prefer)
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.touch(exist_ok=False)
        self._traced_relay_directions = set()
        print(f"[{mode}] Ready. Waiting for lm-eval generation requests.", flush=True)

    @property
    def tokenizer_name(self):
        return "__".join(
            member.tokenizer.name_or_path.replace("/", "__")
            for member in self.sampler.members
        )

    def loglikelihood(self, requests):
        raise NotImplementedError("TIE supports generation tasks")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("TIE supports generation tasks")

    def generate_until(self, requests):
        responses = []
        total = len(requests)
        operation = "Scoring saved answers" if self.saved_answers is not None else "Starting generation"
        print(f"[{self.mode}] {operation} for {total} questions", flush=True)
        progress = tqdm(
            requests, desc=f"{self.mode} GSM8K", unit="question", disable=False
        )
        for completed, request in enumerate(progress, start=1):
            started = perf_counter()
            context, kwargs = request.args
            doc_id = self.doc_ids[request.doc_id] if self.doc_ids is not None else request.doc_id
            prompts = [
                member.tokenizer.apply_chat_template(
                    [{"role": "user", "content": context}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for member in self.sampler.members
            ]
            if self.saved_answers is not None:
                source_prompt, saved = self.saved_answers[doc_id]
                if context != source_prompt["arg_0"] or kwargs != source_prompt["arg_1"]:
                    raise ValueError(f"Evaluation prompt or stop settings changed for doc {doc_id}")
                scoring_texts = None
                if self.mode == "final_answer_fusion_extract":
                    scoring_texts = {
                        model: self.trim(candidate["response"], kwargs)
                        for model, candidate in saved.items()
                    }
                record = score_saved_answers(
                    self.sampler.members, prompts, self.sampler.config.answer_marker, saved,
                    scoring_texts=scoring_texts,
                )
                response = record["candidates"]["llada"]["response"]
                if isinstance(getattr(request, "doc", None), dict) and "answer" in request.doc:
                    for candidate in record["candidates"].values():
                        if candidate["correct"] != self.correct(
                            request.doc, self.trim(candidate["response"], kwargs)
                        ):
                            raise ValueError(f"Saved candidate correctness changed for doc {doc_id}")
            elif self.peer_blocks is not None:
                record = self.peer_blocks.sample(prompts, doc_id)
                response = record["baseline"]["response"]
                if isinstance(getattr(request, "doc", None), dict) and "answer" in request.doc:
                    candidates = [record["baseline"]]
                    for event in record["checkpoints"]:
                        if "peer" in event:
                            candidates.extend((event["peer"], event["owner"]))
                    for candidate in candidates:
                        candidate["correct"] = self.correct(
                            request.doc, self.trim(candidate["response"], kwargs)
                        )
            elif self.shared_audit is not None:
                record = self.shared_audit.sample(prompts)
                response = record["owners"]["llada"]["baseline"]["response"]
                if isinstance(getattr(request, "doc", None), dict) and "answer" in request.doc:
                    for owner in record["owners"].values():
                        owner["baseline"]["correct"] = self.correct(
                            request.doc, self.trim(owner["baseline"]["response"], kwargs)
                        )
                        for event in owner["checkpoints"]:
                            for alternative in event["alternatives"].values():
                                completion = alternative["completion"]
                                completion["correct"] = self.correct(
                                    request.doc, self.trim(completion["response"], kwargs)
                                )
            elif self.short_fusion is not None:
                reference = self.full_rollouts[doc_id] if self.full_rollouts is not None else None
                record = self.short_fusion.sample(
                    prompts, reference, lambda text: self.trim(text, kwargs)
                )
                response = record["baseline"]["response"]
                if isinstance(getattr(request, "doc", None), dict) and "answer" in request.doc:
                    correct = self.correct(request.doc, self.trim(response, kwargs))
                    if reference is not None and correct != reference["baseline"]["correct"]:
                        raise ValueError(f"Full-rollout baseline correctness differs for doc {doc_id}")
                    record["baseline"]["correct"] = correct
                    if reference is None:
                        for event in record["checkpoints"]:
                            event["full"]["keep"]["correct"] = correct
                            swap = event["full"]["swap"]
                            swap["correct"] = self.correct(request.doc, self.trim(swap["response"], kwargs))
            elif self.action_fusion is not None:
                record = self.action_fusion.sample(prompts)
                response = record["baseline"]["response"]
                if isinstance(getattr(request, "doc", None), dict) and "answer" in request.doc:
                    candidates = [record["baseline"],
                                  *(event["alternate"] for event in record["checkpoints"])]
                    for candidate in candidates:
                        candidate["correct"] = self.correct(
                            request.doc, self.trim(candidate["response"], kwargs)
                        )
            elif any((self.experiment, self.exact_fusion, self.commit_support, self.periodic_js)):
                if self.periodic_js is not None:
                    record = self.periodic_js.sample(prompts, confidence_threshold=self.routing_confidence_threshold)
                    arm = "selective_js" if self.mode == SELECTIVE_JS_MODE else "periodic_js"
                    selection = record["branches"][arm]
                elif self.commit_support is not None:
                    record = self.commit_support.sample(prompts, doc_id)
                    selection = record["branches"]["positions" if self.mode == REPLICATION_MODE else "support"]
                elif self.exact_fusion is not None:
                    record = self.exact_fusion.sample(prompts)
                    selection = record["branches"]["joint"]
                else:
                    record = self.experiment.sample(prompts, doc_id)
                    selection = record["branches"].get(self.mode, record["branches"]["keep"])
                response = selection["response"]
                if isinstance(getattr(request, "doc", None), dict) and "answer" in request.doc:
                    for branch in record["branches"].values():
                        branch["correct"] = self.correct(request.doc, self.trim(branch["response"], kwargs))
            elif self.mode == "multi":
                record, response = self.generate_multi(prompts, doc_id)
            else:
                output = self.sampler.sample(
                    prompts, capture_relay_canvases=self.trace_path is not None
                )
                record, response = asdict(output), output.selected.response
            response = self.trim(response, kwargs)
            responses.append(response)
            if self.progress_log_interval and (
                completed == 1 or completed == total
                or completed % self.progress_log_interval == 0
            ):
                progress.write(
                    f"[{self.mode}] {completed}/{total} questions complete "
                    f"(doc {doc_id}, {perf_counter() - started:.1f}s)"
                )
            if self.trace_path is None:
                continue
            record["doc_id"] = doc_id
            record["mode"] = self.mode
            record["response"] = response
            if isinstance(getattr(request, "doc", None), dict) and "answer" in request.doc:
                record["correct"] = self.correct(request.doc, response)
            for relay in record.get("relays", ()):
                leader = relay["leader"]
                if relay["applied"] and leader not in self._traced_relay_directions:
                    self._traced_relay_directions.add(leader)
                else:
                    del relay["before"], relay["after"]
            with self.trace_path.open("a", encoding="utf-8") as trace:
                trace.write(json.dumps(record, allow_nan=False) + "\n")
        if self.mode == SELECTIVE_JS_MODE:
            devices = {member.device for member in self.sampler.members if member.device.type == "cuda"}
            for device in sorted(devices, key=str):
                peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
                print(f"[{self.mode}] {device} {torch.cuda.get_device_name(device)} peak allocated memory {peak_gib:.2f} GiB", flush=True)
        return responses

    @staticmethod
    def correct(doc, response):
        return bool(batch_running_accuracy("gsm8k_cot", [doc], [response]))

    @staticmethod
    def trim(response, kwargs):
        stops = kwargs.get("until") or ()
        if isinstance(stops, str):
            stops = (stops,)
        for stop in stops:
            if stop:
                response = response.split(stop, 1)[0]
        return response

    @torch.inference_mode()
    def generate_multi(self, prompts, doc_id):
        states = []
        for index in range(self.sample_count):
            state = self.sampler.initialize(prompts)[0]
            generator = None if index == 0 else torch.Generator(
                device=state.member.device
            ).manual_seed(1234 + 1009 * doc_id + index)
            for block, count in self.sampler.schedule:
                state.step(block, count, generator=generator)
            states.append(state)
        selected = min(range(len(states)), key=lambda i: states[i].history.rank(final=True))
        completions = [asdict(state.completion(self.sampler.config.answer_marker)) for state in states]
        return {"selected_index": selected, "completions": completions}, completions[selected]["response"]


if __name__ == "__main__":
    print("Starting lm-eval.", flush=True)
    cli_evaluate()
