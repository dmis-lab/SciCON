import argparse
import base64
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

DEFAULT_CHOICES = ["A", "B", "C", "D"]
LETTER_POOL = [chr(ord("A") + i) for i in range(26)]
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"


def safe_display_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return path.name

def find_dataset_file(dataset: str, suffixes: Sequence[str], required_terms: Sequence[str]) -> Optional[Path]:
    if not DATA_ROOT.exists():
        return None
    candidates: List[Path] = []
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        text = str(path.relative_to(DATA_ROOT)).lower()
        if all(term in text for term in required_terms):
            candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def find_dataset_dir(required_terms: Sequence[str]) -> Optional[Path]:
    if not DATA_ROOT.exists():
        return None
    candidates: List[Path] = []
    for path in DATA_ROOT.rglob("*"):
        if not path.is_dir():
            continue
        text = str(path.relative_to(DATA_ROOT)).lower()
        if all(term in text for term in required_terms):
            candidates.append(path)
    return sorted(candidates)[0] if candidates else None


def auto_path_for_mac(split: str) -> Optional[Path]:
    return find_dataset_file("mac", [".jsonl"], ["mac", split])


def auto_mac_image_root() -> Optional[Path]:
    return find_dataset_dir(["images", "mac"]) or find_dataset_dir(["images"])


def auto_scifi_parquet() -> Optional[Path]:
    return find_dataset_file("scifi", [".parquet"], ["scifi"]) or find_dataset_file(
        "scifi", [".parquet"], []
    )


def auto_mmsci_json() -> Optional[Path]:
    return find_dataset_file("mmsci", [".json"], ["mmsci"]) or find_dataset_file(
        "mmsci", [".json"], []
    )


def auto_mmsci_image_root() -> Optional[Path]:
    return find_dataset_dir(["mmsci", "images"]) or find_dataset_dir(["images"])


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    if args.dataset == "mac":
        args.input_jsonl = args.input_jsonl or auto_path_for_mac("test")
        args.image_root = args.image_root or auto_mac_image_root()
    elif args.dataset == "scifi":
        args.input_parquet = args.input_parquet or auto_scifi_parquet()
    elif args.dataset == "mmsci":
        args.input_mmsci_json = args.input_mmsci_json or auto_mmsci_json()
        args.image_root = args.image_root or auto_mmsci_image_root()

    if args.output_jsonl is None:
        args.output_jsonl = RESULTS_ROOT / f"{args.dataset}_predictions.jsonl"


def validate_dataset_inputs(args: argparse.Namespace) -> None:
    if args.dataset == "mac" and args.input_jsonl is None:
        raise ValueError("No MAC input JSONL found. Place the file under data/ or pass --input-jsonl.")
    if args.dataset == "mac" and args.image_root is None:
        raise ValueError("No MAC image root found. Place the images under data/ or pass --image-root.")
    if args.dataset == "scifi" and args.input_parquet is None:
        raise ValueError("No SciFI parquet found. Place the file under data/ or pass --input-parquet.")
    if args.dataset == "mmsci" and args.input_mmsci_json is None:
        raise ValueError("No MMSCI JSON found. Place the file under data/ or pass --input-mmsci-json.")
    if args.dataset == "mmsci" and args.image_root is None:
        raise ValueError("No MMSCI image root found. Place the images under data/ or pass --image-root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Always-contrastive baseline over all candidates (sglang API backend)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mac",
        choices=["mac", "scifi", "mmsci"],
        help="Input dataset type.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Input JSONL for MAC_Bench.",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=None,
        help="Input parquet path for SciFI.",
    )
    parser.add_argument(
        "--input-mmsci-json",
        type=Path,
        default=None,
        help="Input JSON path for MMSCI.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Image root for datasets that reference image files.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://127.0.0.1:30000/v1",
        help="OpenAI-compatible sglang base URL.",
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="",
        help="Model name for API. Empty means auto-detect from /v1/models.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--contrastive-alpha", type=float, default=0.66)
    parser.add_argument("--fallback-max-tokens", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="top_logprobs for chat/completions (vLLM often limits this to <=20).",
    )
    return parser.parse_args()


def load_mac_rows(path: Path, max_samples: int) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples > 0 and idx >= max_samples:
                break
            raw = json.loads(line)
            options_map = {c: str(raw[f"option_{c}"]) for c in DEFAULT_CHOICES}
            rows.append(
                {
                    "id": raw["id"],
                    "question": raw["question"],
                    "answer": str(raw["answer"]).strip().upper(),
                    "candidate_choices": list(DEFAULT_CHOICES),
                    "options_map": options_map,
                    "cover_image": raw["cover_image"],
                    "image_bytes": None,
                    "journal": raw.get("journal", ""),
                    "meta": {"dataset": "mac"},
                }
            )
    return rows


def strip_choice_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-Z]\s*[\)\.\:]\s*", "", text).strip()


def extract_scifi_image_bytes(images_field: object) -> Optional[bytes]:
    if images_field is None:
        return None
    try:
        images = list(images_field)
    except TypeError:
        images = [images_field]
    if not images:
        return None
    first = images[0]
    if isinstance(first, dict):
        data = first.get("bytes")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    return None


def load_scifi_rows(parquet_input: Path, max_samples: int) -> List[Dict]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to load SciFI parquet files.") from exc

    parquet_files: List[Path]
    if parquet_input.is_dir():
        parquet_files = sorted(parquet_input.glob("*.parquet"))
    else:
        parquet_files = [parquet_input]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found from {parquet_input}")

    rows: List[Dict] = []
    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path)
        for _, record in frame.iterrows():
            options_raw = list(record["Options"])
            if len(options_raw) > len(LETTER_POOL):
                raise RuntimeError("Too many options for supported letter pool")
            candidate_choices = LETTER_POOL[: len(options_raw)]
            options_map = {
                candidate_choices[i]: strip_choice_prefix(str(options_raw[i]))
                for i in range(len(options_raw))
            }
            answer = str(record["Answer"]).strip().upper()[:1]
            image_bytes = extract_scifi_image_bytes(record.get("Images"))
            rows.append(
                {
                    "id": f"{parquet_path.stem}_{record['ID']}",
                    "question": str(record["Question"]),
                    "answer": answer,
                    "candidate_choices": candidate_choices,
                    "options_map": options_map,
                    "cover_image": "",
                    "image_bytes": image_bytes,
                    "journal": str(record.get("Category", "")),
                    "meta": {
                        "dataset": "scifi",
                        "parquet_file": parquet_path.name,
                        "raw_id": int(record["ID"]),
                        "category": str(record.get("Category", "")),
                    },
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                return rows
    return rows


def parse_mmsci_question(question: str) -> Tuple[str, List[str], Dict[str, str]]:
    text = str(question).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate_choices: List[str] = []
    options_map: Dict[str, str] = {}
    stem_lines: List[str] = []
    saw_option = False
    for line in lines:
        m = re.match(r"^([A-Z])[\.:]\s*(.+)$", line)
        if m:
            saw_option = True
            choice = m.group(1).strip().upper()
            option_text = m.group(2).strip()
            candidate_choices.append(choice)
            options_map[choice] = option_text
            continue
        if not saw_option:
            stem_lines.append(line)
    if candidate_choices:
        stem = " ".join(stem_lines).strip()
        return stem, candidate_choices, options_map
    return text, [], {}


def parse_mmsci_answer_choice(answer_text: str) -> str:
    m = re.match(r"^\s*([A-Z])[\.:]?\s*", str(answer_text))
    if not m:
        return ""
    return m.group(1).upper()


def load_mmsci_matching_rows(mmsci_json: Path, max_samples: int) -> List[Dict]:
    if not mmsci_json.exists():
        raise RuntimeError(f"MMSci json not found: {mmsci_json}")

    with mmsci_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw and isinstance(raw[0], list):
        records = [x for group in raw for x in group]
    else:
        records = list(raw)

    rows: List[Dict] = []
    for idx, record in enumerate(records):
        stem, candidate_choices, options_map = parse_mmsci_question(record.get("question", ""))
        if len(candidate_choices) < 2:
            continue
        answer = parse_mmsci_answer_choice(record.get("answer", ""))
        if answer not in candidate_choices:
            continue

        uid = str(record.get("uid", f"sample_{idx}"))
        image_name = str(record.get("image", "")).strip()
        if not image_name:
            continue
        rows.append(
            {
                "id": f"mmsci_{mmsci_json.stem}_{idx}_{uid}",
                "question": stem if stem else "Which option best matches the image?",
                "answer": answer,
                "candidate_choices": candidate_choices,
                "options_map": options_map,
                "cover_image": image_name,
                "image_bytes": None,
                "journal": str(record.get("subject", "")),
                "meta": {
                    "dataset": "mmsci",
                    "source_file": mmsci_json.name,
                    "uid": uid,
                    "category": str(record.get("category", "")),
                    "subject": str(record.get("subject", "")),
                },
            }
        )
        if max_samples > 0 and len(rows) >= max_samples:
            return rows
    return rows


def load_mmsci_conversations_rows(mmsci_json: Path, max_samples: int) -> List[Dict]:
    if not mmsci_json.exists():
        raise RuntimeError(f"MMSci json not found: {mmsci_json}")

    with mmsci_json.open("r", encoding="utf-8") as f:
        records = list(json.load(f))

    rows: List[Dict] = []
    for idx, record in enumerate(records):
        convs = list(record.get("conversations", []))
        if len(convs) < 2:
            continue
        human = str(convs[0].get("value", ""))
        answer_raw = str(convs[1].get("value", ""))

        lines = [ln.strip() for ln in human.splitlines() if ln.strip()]
        lines = [ln for ln in lines if ln != "<image>"]
        candidate_choices: List[str] = []
        options_map: Dict[str, str] = {}
        stem_lines: List[str] = []
        saw_option = False
        for line in lines:
            m = re.match(r"^([A-Z])[\.:]\s*(.+)$", line)
            if m:
                saw_option = True
                choice = m.group(1).strip().upper()
                option_text = m.group(2).strip()
                candidate_choices.append(choice)
                options_map[choice] = option_text
                continue
            if not saw_option:
                stem_lines.append(line)
        if len(candidate_choices) < 2:
            continue
        answer = parse_mmsci_answer_choice(answer_raw)
        if answer not in candidate_choices:
            continue
        image_name = str(record.get("image", "")).strip()
        if not image_name:
            continue
        uid = str(record.get("uid", f"sample_{idx}"))
        rows.append(
            {
                "id": f"mmsci_{mmsci_json.stem}_{idx}_{uid}",
                "question": " ".join(stem_lines).strip()
                or "Which option best matches the image?",
                "answer": answer,
                "candidate_choices": candidate_choices,
                "options_map": options_map,
                "cover_image": image_name,
                "image_bytes": None,
                "journal": str(record.get("subject", "")),
                "meta": {
                    "dataset": "mmsci",
                    "source_file": mmsci_json.name,
                    "uid": uid,
                    "category": str(record.get("category", "")),
                    "subject": str(record.get("subject", "")),
                },
            }
        )
        if max_samples > 0 and len(rows) >= max_samples:
            return rows
    return rows


def load_mmsci_rows(mmsci_json: Path, max_samples: int) -> List[Dict]:
    with mmsci_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not raw:
        return []
    first = raw[0][0] if isinstance(raw[0], list) and raw[0] else raw[0]
    if isinstance(first, dict) and "conversations" in first:
        return load_mmsci_conversations_rows(mmsci_json, max_samples)
    return load_mmsci_matching_rows(mmsci_json, max_samples)


def load_parquet_rows(dataset: str, parquet_input: Path, max_samples: int) -> List[Dict]:
    if dataset == "scifi":
        return load_scifi_rows(parquet_input, max_samples)
    raise ValueError(f"Parquet loading is unsupported for dataset={dataset}")


def load_rows(args: argparse.Namespace) -> List[Dict]:
    if args.dataset == "mac":
        return load_mac_rows(args.input_jsonl, args.max_samples)
    if args.dataset == "mmsci":
        return load_mmsci_rows(args.input_mmsci_json, args.max_samples)
    return load_parquet_rows(args.dataset, args.input_parquet, args.max_samples)


def resolve_image_path(image_root: Path, cover_image: str) -> Path:
    relative = cover_image
    if relative.startswith("MAC_Bench/"):
        relative = relative[len("MAC_Bench/") :]
    return image_root / relative


def build_prompt_full(row: Dict) -> str:
    choices = row["candidate_choices"]
    option_lines = "\n\n".join(
        f"{choice}. {row['options_map'][choice]}" for choice in choices
    )
    choice_text = ", ".join(choices)
    choice_slash = "/".join(choices)
    return (
        f"Question: {row['question']}\n\n"
        f"Choose exactly one option ({choice_slash}) that best matches the image.\n\n"
        f"{option_lines}\n\n"
        f"Answer with one letter only: {choice_text}."
    )


def encode_image_bytes_to_data_url(image_bytes: bytes) -> str:
    header = image_bytes[:16]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif header.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif header[:4] == b"RIFF" and b"WEBP" in header:
        mime = "image/webp"
    else:
        mime = "image/png"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def encode_image_to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def resolve_api_model(api_base: str, api_model: str, timeout_sec: int) -> str:
    if api_model:
        return api_model
    resp = requests.get(f"{api_base}/models", timeout=timeout_sec)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    if not models:
        raise RuntimeError("No models found from /v1/models")
    return models[0]["id"]


def normalize_token_to_choice(token: str, choice_set: Sequence[str]) -> str:
    tok = token.strip()
    tok = tok[:1].upper() if tok else ""
    return tok if tok in choice_set else ""


def extract_choice_from_text(text: str, choice_set: Sequence[str]) -> str:
    m = re.search(r"\b([A-Z])\b", text.upper())
    if not m:
        return ""
    token = m.group(1)
    return token if token in choice_set else ""


def softmax(logits: Sequence[float]) -> List[float]:
    max_l = max(logits)
    exps = [math.exp(x - max_l) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def confidence_metrics(probs: Sequence[float]) -> Dict[str, float]:
    eps = 1e-12
    entropy = -sum(p * math.log(p + eps) for p in probs)
    norm_entropy = entropy / math.log(len(probs))
    sorted_probs = sorted(probs, reverse=True)
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = top1 - top2
    confidence = 0.5 * ((1.0 - norm_entropy) + margin)
    return {
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "top1_prob": top1,
        "top2_prob": top2,
        "margin": margin,
        "confidence": confidence,
    }


def compute_macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    labels = sorted({x for x in y_true if x} | {x for x in y_pred if x})
    if not labels:
        return 0.0
    f1s: List[float] = []
    for c in labels:
        tp = sum((t == c and p == c) for t, p in zip(y_true, y_pred))
        fp = sum((t != c and p == c) for t, p in zip(y_true, y_pred))
        fn = sum((t == c and p != c) for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def predict_from_logits(
    candidate_choices: Sequence[str],
    init_probs: Sequence[float],
    init_logits_map: Dict[str, float],
    txt_logits_map: Dict[str, float],
    alpha: float,
) -> Tuple[str, bool, List[float], List[float]]:
    l_img = [init_logits_map[c] for c in candidate_choices]
    l_txt = [txt_logits_map[c] for c in candidate_choices]
    l_ctr = [l_img[i] - alpha * l_txt[i] for i in range(len(candidate_choices))]
    p_ctr = softmax(l_ctr)
    best_idx = max(range(len(candidate_choices)), key=lambda i: p_ctr[i])
    return candidate_choices[best_idx], True, l_ctr, p_ctr


def request_distribution(
    api_base: str,
    model_name: str,
    prompt: str,
    candidate_choices: Sequence[str],
    timeout_sec: int,
    image_path: Optional[Path] = None,
    image_bytes: Optional[bytes] = None,
    top_logprobs: int = 20,
) -> Tuple[List[float], Dict[str, float], str]:
    content: List[Dict] = [{"type": "text", "text": prompt}]
    if image_bytes is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": encode_image_bytes_to_data_url(image_bytes)}}
        )
    elif image_path is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(image_path)},
            }
        )

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(
        f"{api_base}/chat/completions",
        json=payload,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    body = resp.json()

    message = body["choices"][0]["message"]["content"] or ""
    content_logprobs = body["choices"][0]["logprobs"]["content"][0]
    top = content_logprobs.get("top_logprobs", [])

    choice_logprob: Dict[str, float] = {c: float("-inf") for c in candidate_choices}
    min_lp = float("inf")
    for item in top:
        tok = item.get("token", "")
        lp = float(item.get("logprob", -100.0))
        min_lp = min(min_lp, lp)
        c = normalize_token_to_choice(tok, candidate_choices)
        if c in choice_logprob:
            choice_logprob[c] = max(choice_logprob[c], lp)

    floor = -20.0 if min_lp == float("inf") else min_lp - 5.0
    logits = [
        choice_logprob[c] if choice_logprob[c] != float("-inf") else floor
        for c in candidate_choices
    ]
    probs = softmax(logits)
    return probs, {c: logits[i] for i, c in enumerate(candidate_choices)}, message


def request_fallback_choice(
    api_base: str,
    model_name: str,
    prompt: str,
    timeout_sec: int,
    max_tokens: int,
    candidate_choices: Sequence[str],
    image_path: Optional[Path] = None,
    image_bytes: Optional[bytes] = None,
) -> str:
    choice, _ = request_fallback_choice_with_text(
        api_base=api_base,
        model_name=model_name,
        prompt=prompt,
        timeout_sec=timeout_sec,
        max_tokens=max_tokens,
        candidate_choices=candidate_choices,
        image_path=image_path,
        image_bytes=image_bytes,
    )
    return choice


def request_fallback_choice_with_text(
    api_base: str,
    model_name: str,
    prompt: str,
    timeout_sec: int,
    max_tokens: int,
    candidate_choices: Sequence[str],
    image_path: Optional[Path] = None,
    image_bytes: Optional[bytes] = None,
) -> Tuple[str, str]:
    content: List[Dict] = [{"type": "text", "text": prompt}]
    if image_bytes is not None:
        content.append(
            {"type": "image_url", "image_url": {"url": encode_image_bytes_to_data_url(image_bytes)}}
        )
    elif image_path is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(image_path)},
            }
        )

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(
        f"{api_base}/chat/completions",
        json=payload,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"] or ""
    choice = extract_choice_from_text(text, candidate_choices)
    return choice, text


def main() -> None:
    args = parse_args()
    apply_dataset_defaults(args)
    validate_dataset_inputs(args)
    rows = load_rows(args)
    if not rows:
        if args.dataset == "mac":
            raise RuntimeError(f"No rows found in {args.input_jsonl}")
        if args.dataset == "mmsci":
            raise RuntimeError(f"No rows found in {args.input_mmsci_json}")
        raise RuntimeError(f"No rows found in {args.input_parquet}")

    model_name = resolve_api_model(args.api_base, args.api_model, args.timeout_sec)
    print(f"Using API model: {model_name}", flush=True)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    missing = 0
    contrastive_applied = 0
    y_true: List[str] = []
    y_pred: List[str] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for row in rows:
            candidate_choices = row["candidate_choices"]
            image_path: Optional[Path] = None
            image_bytes: Optional[bytes] = row.get("image_bytes")

            if image_bytes is None:
                if not row.get("cover_image"):
                    missing += 1
                    out_f.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "error": "image_missing",
                                "gold": row["answer"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                image_path = resolve_image_path(args.image_root, row["cover_image"])
                if not image_path.exists():
                    missing += 1
                    out_f.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "image_path": safe_display_path(image_path),
                                "error": "image_not_found",
                                "gold": row["answer"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

            full_prompt = build_prompt_full(row)
            init_probs, init_logits_map, init_message = request_distribution(
                api_base=args.api_base,
                model_name=model_name,
                prompt=full_prompt,
                candidate_choices=candidate_choices,
                image_path=image_path,
                image_bytes=image_bytes,
                timeout_sec=args.timeout_sec,
                top_logprobs=args.top_logprobs,
            )
            conf = confidence_metrics(init_probs)
            ranked = sorted(
                zip(candidate_choices, init_probs), key=lambda x: x[1], reverse=True
            )

            l_img = [init_logits_map[c] for c in candidate_choices]
            txt_probs, txt_logits_map, txt_message = request_distribution(
                api_base=args.api_base,
                model_name=model_name,
                prompt=full_prompt,
                candidate_choices=candidate_choices,
                image_path=None,
                image_bytes=None,
                timeout_sec=args.timeout_sec,
                top_logprobs=args.top_logprobs,
            )
            l_txt = [txt_logits_map[c] for c in candidate_choices]
            l_ctr = [
                l_img[i] - args.contrastive_alpha * l_txt[i]
                for i in range(len(candidate_choices))
            ]
            p_ctr = softmax(l_ctr)

            use_contrastive = True
            contrastive_applied += 1
            best_idx = max(range(len(candidate_choices)), key=lambda i: p_ctr[i])
            final_choice = candidate_choices[best_idx]

            if final_choice not in candidate_choices:
                fallback = request_fallback_choice(
                    api_base=args.api_base,
                    model_name=model_name,
                    prompt=full_prompt,
                    image_path=image_path,
                    image_bytes=image_bytes,
                    timeout_sec=args.timeout_sec,
                    max_tokens=args.fallback_max_tokens,
                    candidate_choices=candidate_choices,
                )
                final_choice = fallback if fallback in candidate_choices else ranked[0][0]

            gold = row["answer"].strip().upper()
            is_correct = final_choice == gold
            total += 1
            correct += int(is_correct)
            y_true.append(gold)
            y_pred.append(final_choice)

            record = {
                "id": row["id"],
                "journal": row.get("journal", ""),
                "dataset": row.get("meta", {}).get("dataset", args.dataset),
                "meta": row.get("meta", {}),
                "image_path": safe_display_path(image_path),
                "image_source": "bytes" if image_bytes is not None else "path",
                "gold": gold,
                "pred": final_choice,
                "correct": is_correct,
                "used_contrastive": use_contrastive,
                "initial_probs": {
                    candidate_choices[i]: init_probs[i] for i in range(len(candidate_choices))
                },
                "initial_logits": init_logits_map,
                "text_only_probs": (
                    {
                        candidate_choices[i]: txt_probs[i]
                        for i in range(len(candidate_choices))
                    }
                    if txt_probs is not None
                    else None
                ),
                "text_only_logits": txt_logits_map,
                "confidence": conf,
                "initial_raw": init_message,
                "text_only_raw": txt_message,
                "contrastive_detail": {
                    "contrastive_alpha": args.contrastive_alpha,
                    "normalized_entropy": conf["normalized_entropy"],
                    "candidate_order": candidate_choices,
                    "all_candidate_img_logits": {
                        candidate_choices[i]: l_img[i]
                        for i in range(len(candidate_choices))
                    },
                    "all_candidate_txt_logits": {
                        candidate_choices[i]: l_txt[i]
                        for i in range(len(candidate_choices))
                    },
                    "all_candidate_contrastive_logits": {
                        candidate_choices[i]: l_ctr[i]
                        for i in range(len(candidate_choices))
                    },
                    "all_candidate_contrastive_probs": {
                        candidate_choices[i]: p_ctr[i]
                        for i in range(len(candidate_choices))
                    },
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(
                f"[{total}] id={row['id']} gold={gold} pred={final_choice or 'N/A'} "
                f"correct={is_correct} top1={ranked[0][0]} "
                f"Hnorm={conf['normalized_entropy']:.3f} ctr={use_contrastive}",
                flush=True,
            )

    inferred = total
    attempted = total + missing
    accuracy = (correct / inferred) if inferred > 0 else 0.0
    macro_f1 = compute_macro_f1(y_true, y_pred)
    print("\n=== Summary ===")
    print(f"dataset: {args.dataset}")
    print(f"rows_requested: {len(rows)}")
    print(f"rows_attempted: {attempted}")
    print(f"rows_inferred: {inferred}")
    print(f"images_missing: {missing}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"macro_f1: {macro_f1:.4f}")
    print(f"contrastive_applied: {contrastive_applied}")
    print(
        f"contrastive_apply_rate: {(contrastive_applied / inferred) if inferred > 0 else 0.0:.4f}"
    )
    print(f"output: {safe_display_path(args.output_jsonl)}")


if __name__ == "__main__":
    main()
