import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 로그 폴더와 출력 파일 (경로 고정)
LOG_DIR = Path(r"Z:\ECT_FrameGenerator\Spoofing\framegenerator\x64\Release\results\Main")
OUT = Path("Fig4_test_cd_hist.png")

CD_TH = 12.0  # th = 0.90, D_max = 120


def load_cd(path):
    """프레임 로그에서 ChamferRaw(px)만 추출. RectError(-1)는 제외."""
    vals = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"ChamferRaw=([-\d.]+)", line)
            if m and float(m.group(1)) >= 0:
                vals.append(float(m.group(1)))
    return vals


real = load_cd(LOG_DIR / "real_merged_result.txt")
attack = load_cd(LOG_DIR / "attack_merged_result.txt")
print(f"bona-fide {len(real)} frames: {min(real):.2f}-{max(real):.2f} px")
print(f"attack    {len(attack)} frames: {min(attack):.2f}-{max(attack):.2f} px")

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                     "mathtext.fontset": "stix"})

fig, ax = plt.subplots(figsize=(3.5, 2.2))  # 단(column) 폭 기준, 인치

bins = range(0, 150, 2)  # 2px 간격
ax.hist(real, bins=bins, color="#2b6cb0", alpha=0.85, label=f"Bona-fide (n={len(real):,})")
ax.hist(attack, bins=bins, color="#c05621", alpha=0.75, label=f"Attack (n={len(attack):,})")

ax.axvline(CD_TH, color="black", linestyle="--", linewidth=0.8)
ax.text(CD_TH + 2, 1500, r"CD$_{th}$ = 12 px", fontsize=8)

ax.set_yscale("log")
ax.set_xlabel("Bidirectional Chamfer Distance (px)")
ax.set_ylabel("Number of frames")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)


fig.tight_layout()
fig.savefig(OUT, dpi=600)
print("saved:", OUT.resolve())