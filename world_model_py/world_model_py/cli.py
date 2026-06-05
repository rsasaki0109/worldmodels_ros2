"""``world-model`` command-line entry point.

    world-model list                         # list registered adapters
    world-model info --adapter dummy         # print adapter metadata
    world-model bench --adapter dummy --out report.html --runs 100

Runs with no ROS install and no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys

from .bench import bench_to_file
from .registry import available_models, load_model


def _cmd_list(_args) -> int:
    for name in available_models():
        print(name)
    return 0


def _cmd_info(args) -> int:
    adapter = load_model(args.adapter, **_kwargs(args))
    print(json.dumps(adapter.info(), indent=2))
    return 0


def _cmd_bench(args) -> int:
    result = bench_to_file(
        args.adapter,
        args.out,
        runs=args.runs,
        horizon=args.horizon,
        **_kwargs(args),
    )
    print(
        f"[{result.adapter}] p50={result.latency_ms_p50:.2f}ms "
        f"p95={result.latency_ms_p95:.2f}ms "
        f"throughput={result.throughput_hz:.1f}Hz "
        f"vram={'n/a' if result.vram_peak_mb is None else round(result.vram_peak_mb, 1)}MB"
    )
    print(f"wrote {args.out}")
    return 0


def _kwargs(args) -> dict:
    """Adapter constructor kwargs shared across subcommands (e.g. --url)."""
    kw = {}
    if getattr(args, "url", None):
        kw["url"] = args.url
    return kw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="world-model", description="World Model ROS 2 runtime CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered adapters").set_defaults(func=_cmd_list)

    p_info = sub.add_parser("info", help="print adapter metadata")
    p_info.add_argument("--adapter", default="dummy")
    p_info.add_argument("--url", default=None, help="remote adapter url")
    p_info.set_defaults(func=_cmd_info)

    p_bench = sub.add_parser("bench", help="benchmark an adapter -> HTML report")
    p_bench.add_argument("--adapter", default="dummy")
    p_bench.add_argument("--out", default="report.html")
    p_bench.add_argument("--runs", type=int, default=50)
    p_bench.add_argument("--horizon", type=int, default=8)
    p_bench.add_argument("--url", default=None, help="remote adapter url")
    p_bench.set_defaults(func=_cmd_bench)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
