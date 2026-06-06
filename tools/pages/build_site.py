"""Build the GitHub Pages site: a demo landing (GIFs) + the live benchmark.

    python3 tools/pages/build_site.py --repo-root . --out _site

Copies docs/*.gif, generates the adapter benchmark dashboard via the CLI
(GPU-free dummy vs remote-over-HTTP), and writes a self-contained index.html.
Used by .github/workflows/pages.yml.
"""
import argparse
import glob
import html
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/rsasaki0109/worldmodels_ros2"

CAPTIONS = {
    "hero.gif": "Runtime anomaly monitor: real LeRobot SO-101 footage → V-JEPA 2 → self-calibrated anomaly flag.",
    "imagination.gif": "Top-down: predicted future occupancy of an obstacle (green→red) + risk.",
    "jepa_compare.gif": "Same camera stream, two real World Models: I-JEPA vs V-JEPA 2 surprise.",
    "nav2_scoring.gif": "Nav2 candidate paths ranked by model-based risk; safest highlighted.",
    "ijepa_surprise.gif": "Real I-JEPA surprise spiking on scene changes (GPU).",
}
HERO = "hero.gif"
ORDER = ["jepa_compare.gif", "imagination.gif", "nav2_scoring.gif", "ijepa_surprise.gif"]


def _card(gif: str) -> str:
    cap = html.escape(CAPTIONS.get(gif, gif))
    return (f'<figure><img src="{gif}" alt="{cap}" loading="lazy">'
            f'<figcaption>{cap}</figcaption></figure>')


def build_index(gifs: list) -> str:
    hero = _card(HERO) if HERO in gifs else ""
    gallery = "\n".join(_card(g) for g in ORDER if g in gifs)
    extras = "\n".join(_card(g) for g in sorted(gifs)
                       if g != HERO and g not in ORDER)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>world_model_ros2 — the ROS 2 layer for World Models</title>
<style>
 :root{{color-scheme:dark}}
 body{{font-family:system-ui,sans-serif;margin:0;background:#15151a;color:#e9e9ee}}
 .wrap{{max-width:980px;margin:0 auto;padding:2rem 1.2rem 4rem}}
 h1{{font-size:1.8rem;margin:.2rem 0}}
 a{{color:#5ad1ff}}
 .tag{{color:#a6a6b3;font-size:1.05rem}}
 .btns{{margin:1.2rem 0 2rem}}
 .btn{{display:inline-block;background:#3b82f6;color:#fff;padding:.55rem 1rem;border-radius:8px;text-decoration:none;margin-right:.6rem;font-weight:600}}
 .btn.alt{{background:#2a2a33;color:#e9e9ee}}
 ul.hi{{line-height:1.7}}
 figure{{margin:0 0 1.6rem}}
 img{{width:100%;border:1px solid #2a2a33;border-radius:10px;display:block}}
 figcaption{{color:#a6a6b3;font-size:.92rem;margin-top:.4rem}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.4rem}}
 @media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}
 footer{{color:#6f6f7a;margin-top:2rem;font-size:.9rem}}
</style></head><body><div class="wrap">
<h1>world_model_ros2</h1>
<p class="tag">Run existing World Models from ROS 2 — runtime, adapters,
benchmark, visualization. <em>The ROS 2 layer for World Models, not another
foundation model.</em></p>
<div class="btns">
  <a class="btn" href="{REPO_URL}">View on GitHub</a>
  <a class="btn alt" href="bench.html">Live benchmark &rarr;</a>
</div>

<ul class="hi">
  <li><b>Two real model backends, one contract</b> — I-JEPA (image) &amp; V-JEPA&nbsp;2 (video).</li>
  <li><b>Local &harr; remote split</b> over a shared JSON wire (Cosmos/DreamZero-ready).</li>
  <li><b>Compiled Nav2 costmap layer</b> — predicted occupancy into the live costmap.</li>
  <li><b>rosbag2 &rarr; LeRobot</b> dataset export, GPU-free.</li>
  <li><b>RViz + Foxglove</b> imagination viewer.</li>
</ul>

{hero}
<div class="grid">
{gallery}
{extras}
</div>

<footer>Auto-generated from the repo on every push · all GIFs are real
pipeline / real model output · <a href="{REPO_URL}">source</a></footer>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="_site")
    ap.add_argument("--runs", type=int, default=300)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    gifs = []
    for path in sorted(glob.glob(os.path.join(repo, "docs", "*.gif"))):
        shutil.copy(path, out)
        gifs.append(os.path.basename(path))

    subprocess.run(
        [sys.executable, "-m", "world_model_py.cli", "bench-compare",
         "--adapters", "dummy,remote", "--runs", str(args.runs),
         "--out", os.path.join(out, "bench.html")],
        cwd=os.path.join(repo, "world_model_py"), check=True,
    )

    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(build_index(gifs))
    print(f"built site at {out} ({len(gifs)} gifs + bench.html + index.html)")


if __name__ == "__main__":
    main()
