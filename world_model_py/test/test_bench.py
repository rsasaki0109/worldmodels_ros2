"""Benchmark + comparison dashboard. No ROS, no GPU."""
import os

from world_model_py.bench import run_bench, render_comparison_html
from world_model_py.cli import main as cli_main


def test_comparison_html_lists_all_adapters():
    results = [run_bench("dummy", runs=4, warmup=1), run_bench("dummy", runs=4, warmup=1)]
    html = render_comparison_html(results, title="t")
    assert "<table" in html
    assert html.count("dummy") >= 2
    assert "p95" in html


def test_bench_compare_cli_local_vs_remote(tmp_path):
    out = str(tmp_path / "compare.html")
    rc = cli_main(["bench-compare", "--adapters", "dummy,remote", "--runs", "5", "--out", out])
    assert rc == 0
    assert os.path.exists(out)
    body = open(out).read()
    assert "dummy" in body and "remote" in body
    # the remote row is flagged as remote
    assert "yes" in body


def test_bench_compare_cli_single_adapter(tmp_path):
    out = str(tmp_path / "one.html")
    rc = cli_main(["bench-compare", "--adapters", "dummy", "--runs", "3", "--out", out])
    assert rc == 0
    assert os.path.exists(out)
