"""Latency benchmark for the retrieval service.

Three modes, from innermost to outermost -- comparing them tells you where the
time goes (model vs. HTTP/serialization vs. network/TLS/Vertex front end):

  # 1) in-process: model encode + index search only, no HTTP
  python scripts/benchmark_latency.py inprocess --checkpoint outputs/fusion_model.pt \
      --index outputs/gallery_index.pt

  # 2) HTTP against a local container / uvicorn
  python scripts/benchmark_latency.py http --url http://localhost:8080

  # 3) a deployed Vertex AI endpoint (auth via `gcloud auth print-access-token`)
  python scripts/benchmark_latency.py vertex --project <id> --region us-central1 --endpoint_id <id>

Common flags: --n 300 --warmup 20 --concurrency 1 --top_k 5 --out benchmarks/
Report: p50/p90/p95/p99 of client-side latency, plus the server-reported
breakdown (encode_ms / search_ms / inference_ms) when available.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QUERIES = [
    "a dog running on the beach", "two men playing basketball", "a child eating ice cream",
    "a woman riding a bicycle in the city", "people sitting at a restaurant table",
    "a man climbing a rock wall", "a group of people at a concert", "a red car on the street",
    "a little girl in a pink dress", "a surfer riding a wave",
]


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values):
    if not values:
        return None
    return {"n": len(values), "mean": statistics.fmean(values),
            **{f"p{p}": percentile(values, p) for p in (50, 90, 95, 99)},
            "max": max(values)}


# ---------------- request functions: each returns (client_ms, server_timing_dict|None) -------
def make_inprocess(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch

    from src.serving.service import load_service

    if args.threads:
        torch.set_num_threads(args.threads)
    service = load_service(args.checkpoint, args.index, device=args.device)
    print(f"device={service.encoder.device} torch_threads={torch.get_num_threads()} "
          f"index_size={len(service.index)}")

    def call(query):
        start = time.perf_counter()
        out = service.search_text(query, args.top_k)
        return (time.perf_counter() - start) * 1000, out["timing_ms"]
    return call


def _http_caller(url, headers, body_fn, extract_timing):
    import httpx

    client = httpx.Client(timeout=30.0, headers=headers,
                          limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))
    errors = {"count": 0}
    lock = threading.Lock()

    def call(query):
        start = time.perf_counter()
        r = client.post(url, json=body_fn(query))
        elapsed = (time.perf_counter() - start) * 1000
        if r.status_code != 200:
            with lock:
                errors["count"] += 1
            return elapsed, None
        return elapsed, extract_timing(r.json())
    call.errors = errors
    return call


def make_http(args):
    return _http_caller(args.url.rstrip("/") + "/search", {},
                        lambda q: {"query": q, "top_k": args.top_k},
                        lambda j: j.get("timing_ms"))


def make_vertex(args):
    token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    url = (f"https://{args.region}-aiplatform.googleapis.com/v1/projects/{args.project}"
           f"/locations/{args.region}/endpoints/{args.endpoint_id}:predict")
    return _http_caller(url, {"Authorization": f"Bearer {token}"},
                        lambda q: {"instances": [{"text": q}], "parameters": {"top_k": args.top_k}},
                        lambda j: j["predictions"][0].get("timing_ms"))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    p_in = sub.add_parser("inprocess")
    p_in.add_argument("--checkpoint", default="outputs/flickr30k/fusion_model.pt")
    p_in.add_argument("--index", default="outputs/flickr30k/gallery_index.pt")
    p_in.add_argument("--device", default=None)
    p_in.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    p_http = sub.add_parser("http")
    p_http.add_argument("--url", default="http://localhost:8080")
    p_vx = sub.add_parser("vertex")
    p_vx.add_argument("--project", required=True)
    p_vx.add_argument("--region", default="us-central1")
    p_vx.add_argument("--endpoint_id", required=True)
    for p in (p_in, p_http, p_vx):
        p.add_argument("--n", type=int, default=300, help="measured requests")
        p.add_argument("--warmup", type=int, default=20, help="discarded requests")
        p.add_argument("--concurrency", type=int, default=1)
        p.add_argument("--top_k", type=int, default=5)
        p.add_argument("--out", default="benchmarks", help="directory for the JSON report")
    args = parser.parse_args()

    call = {"inprocess": make_inprocess, "http": make_http, "vertex": make_vertex}[args.mode](args)
    queries = [QUERIES[i % len(QUERIES)] for i in range(args.n)]

    for i in range(args.warmup):
        call(QUERIES[i % len(QUERIES)])

    wall_start = time.perf_counter()
    if args.concurrency == 1:
        samples = [call(q) for q in queries]
    else:
        with ThreadPoolExecutor(args.concurrency) as pool:
            samples = list(pool.map(call, queries))
    wall_s = time.perf_counter() - wall_start

    client_ms = [c for c, _ in samples]
    server = [t for _, t in samples if t]
    report = {
        "mode": args.mode, "n": args.n, "concurrency": args.concurrency, "top_k": args.top_k,
        "throughput_rps": args.n / wall_s,
        "errors": getattr(call, "errors", {"count": 0})["count"],
        "client_latency_ms": summarize(client_ms),
        "server_timing_ms": {k: summarize([t[k] for t in server if k in t])
                             for k in ("encode_ms", "search_ms", "inference_ms")} if server else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def row(name, s):
        if s:
            print(f"  {name:<22} p50={s['p50']:7.2f}  p90={s['p90']:7.2f}  p95={s['p95']:7.2f}  "
                  f"p99={s['p99']:7.2f}  mean={s['mean']:7.2f}  max={s['max']:7.2f}")

    print(f"\n[{args.mode}] n={args.n} concurrency={args.concurrency} "
          f"throughput={report['throughput_rps']:.1f} req/s errors={report['errors']}")
    row("client latency (ms)", report["client_latency_ms"])
    for k, s in (report["server_timing_ms"] or {}).items():
        row(f"server {k}", s)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"latency_{args.mode}_c{args.concurrency}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out_file}")


if __name__ == "__main__":
    main()
