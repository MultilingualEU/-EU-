"""중간점검용 그래프 4개. Drive 결과 파일만 읽고, figures/ 폴더에 PNG로 저장함.

실행: python make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from b_common import ROOT  # Drive 기계학습 폴더 (EU_ROOT 환경변수로 변경 가능)

OUT = Path(__file__).resolve().parent / 'figures'
TARGETS = ['DE', 'FR', 'EL', 'FI', 'PL', 'BG', 'MT']

# 색과 잉크 (dataviz 기본 팔레트, light). 범주 색은 1~3번 슬롯만 순서대로 사용
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, BASELINE = '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': BASELINE, 'axes.linewidth': 1, 'grid.color': GRID, 'grid.linewidth': 1,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11,
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('저장:', OUT / name)


def rq2_dumbbell():
    """언어별 Standard → Diagnostic → Direct를 한 줄에 점으로 표시 (덤벨). 오른쪽에 Transfer Gap."""
    t = pd.read_csv(ROOT / 'results/rq2/rq2_table.csv')
    t['lang'] = t['lang'].str.upper()
    t = t.set_index('lang')
    en = t.loc['EN', 'standard_micro_f1'] * 100
    t = t.loc[TARGETS].sort_values('observed_gap_micro_f1')  # 손실 작은 언어가 아래
    s, g, d = (t[f'{c}_micro_f1'] * 100 for c in ['standard', 'diagnostic', 'direct'])
    y = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hlines(y, s, d, color=GRID, lw=6, zorder=1, capstyle='round')  # 잃은 구간
    ax.scatter(s, y, s=90, color=ORANGE, edgecolors=SURFACE, linewidths=2, zorder=3,
               label='Standard: 영어 모델 그대로')
    ax.scatter(g, y, s=60, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.8, zorder=3,
               label='Diagnostic: 기준값만 다시 맞춤')
    ax.scatter(d, y, s=90, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3,
               label='Direct: 그 언어로 직접 학습')
    ax.axvline(en, color=BASELINE, lw=1, zorder=0)
    ax.text(en, len(t) - 0.35, f'EN {en:.1f}', ha='center', va='bottom', fontsize=9, color=MUTED)
    for yi, gap in zip(y, t['observed_gap_micro_f1'] * 100):
        ax.text(77.2, yi, f'{gap:.1f}', ha='left', va='center', fontsize=11, color=INK)
    ax.text(77.2, len(t) - 0.35, 'Transfer Gap', ha='left', va='bottom', fontsize=9, color=MUTED)

    ax.set_yticks(y, t.index)
    ax.set_xlim(56, 80)
    ax.set_ylim(-0.6, len(t) - 0.1)
    ax.set_xticks([60, 65, 70, 75])
    ax.set_xlabel('test Micro-F1')
    ax.xaxis.grid(True, zorder=0)
    ax.tick_params(length=0)
    ax.tick_params(axis='y', labelsize=12, labelcolor=INK)
    ax.spines['left'].set_visible(False)
    ax.set_title('RQ2. 영어 모델을 그대로 쓰면 잃는 점수 (손실 큰 언어가 위)', loc='left', fontsize=13, pad=34)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.01, 1.0), ncol=3, frameon=False, fontsize=9.5,
              handletextpad=0.3, columnspacing=1.2, borderaxespad=0.3)
    save(fig, 'rq2_conditions.png')


def rq3_recovery():
    """언어별 회복률 곡선 (10 seeds 평균, 띠는 ±1 SD). 언어가 7개라 작은 그래프 7개로 나눔."""
    r = pd.read_csv(ROOT / 'rq3_fewshot_결과/summary/recovery.csv')
    r['lang'] = r['lang'].str.upper()
    order = r[r.N == 1000].sort_values('recovery_mean', ascending=False).lang.tolist()  # 많이 회복한 순
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, lang in zip(axes.flat, order):
        d = r[r.lang == lang].sort_values('N')
        ax.axhline(100, color=BASELINE, lw=1, zorder=1)
        ax.fill_between(d.N, d.recovery_mean - d.recovery_sd, d.recovery_mean + d.recovery_sd,
                        color=BLUE, alpha=0.10, lw=0, zorder=2)
        ax.plot(d.N, d.recovery_mean, color=BLUE, lw=2, marker='o', ms=5,
                mec=SURFACE, mew=1.5, solid_capstyle='round', zorder=3)
        last = d.iloc[-1]
        ax.text(last.N, last.recovery_mean - 14, f'{last.recovery_mean:.0f}%', ha='right', fontsize=10, color=INK)
        ax.set_title(lang, loc='left', fontsize=12, color=INK)
        ax.set_xscale('log')
        ax.set_xticks([10, 30, 100, 300, 1000], ['10', '30', '100', '300', '1000'])
        ax.minorticks_off()
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 50, 100])
        ax.yaxis.grid(True, zorder=0)
        ax.tick_params(length=0, labelsize=9)
        ax.spines['left'].set_visible(False)
    note = axes.flat[-1]  # 빈 칸은 읽는 법 설명
    note.axis('off')
    note.text(0.05, 0.75, '읽는 법', fontsize=11, color=INK, weight='bold', transform=note.transAxes)
    note.text(0.05, 0.62, '선: 10회 평균 회복률\n띠: ±1 SD (회차별 흔들림)\n위 회색 선: 100% (완전 회복)',
              fontsize=10, color=INK2, va='top', linespacing=1.7, transform=note.transAxes)
    axes[0, 3].xaxis.set_tick_params(labelbottom=True)
    fig.supxlabel('추가한 그 언어 문서 수 N (로그 눈금)', fontsize=11, color=INK2)
    fig.supylabel('회복률 (%)', fontsize=11, color=INK2)
    fig.suptitle('RQ3. 그 언어 문서를 N개 추가했을 때 Transfer Gap 회복률 (N=1,000 회복률 높은 순)',
                 x=0.01, ha='left', fontsize=13)
    fig.tight_layout()
    save(fig, 'rq3_recovery.png')


def rq4_scatter():
    """Transfer Gap과 Top-1 alignment, fertility의 관계. 점마다 언어 이름."""
    d = pd.read_csv(ROOT / 'results/rq4/rq4_language_level.csv', encoding='utf-8-sig')
    d['lang'] = d['lang'].str.upper()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, col, label in [(axes[0], 'top1_retrieval_acc', 'Top-1 alignment (영어판 짝 찾기 정확도)'),
                           (axes[1], 'corpus_fertility', 'Fertility (단어당 token 수)')]:
        x, y = d[col], d['transfer_gap'] * 100
        rho = x.rank().corr(y.rank())  # Spearman ρ
        ax.scatter(x, y, s=80, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
        # 붙어 있는 점(BG, DE, FR)은 글자가 겹치지 않게 위치를 따로 지정
        offsets = {'FR': (6, -14), 'BG': (-8, 6), 'DE': (6, 6)} if col == 'corpus_fertility' else {}
        for xi, yi, lang in zip(x, y, d['lang']):
            dx, dy = offsets.get(lang, (7, 5))
            ax.annotate(lang, (xi, yi), xytext=(dx, dy), textcoords='offset points', fontsize=10,
                        color=INK, ha='right' if dx < 0 else 'left')
        ax.set_xlabel(label)
        ax.set_title(f'Spearman ρ = {rho:+.2f}', loc='left', fontsize=11, color=INK2)
        ax.yaxis.grid(True)
        ax.tick_params(length=0)
        ax.spines['left'].set_visible(False)
        ax.margins(x=0.12, y=0.15)
    axes[0].set_ylabel('Transfer Gap (Micro-F1 점)')
    fig.suptitle('RQ4. 손실(Transfer Gap)과 함께 움직이는 특성 (7개 언어, 탐색적)', x=0.01, ha='left', fontsize=13)
    fig.tight_layout()
    save(fig, 'rq4_scatter.png')


def rq1_ranks():
    """TF-IDF 순위와 E5 Direct 순위 비교. MT만 강조."""
    d = pd.read_csv(ROOT / 'd_phase2&3_results/rq1_rank_comparison.csv', encoding='utf-8-sig')
    rho = d.tfidf_rank.corr(d.e5_direct_rank)  # 순위끼리의 Pearson = Spearman ρ
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for _, row in d.iterrows():
        hi = row.language == 'MT'
        color, lw, z = (BLUE, 2.5, 3) if hi else (BASELINE, 1.5, 2)
        ax.plot([0, 1], [row.tfidf_rank, row.e5_direct_rank], color=color, lw=lw, zorder=z,
                marker='o', ms=6, mec=SURFACE, mew=1.5, solid_capstyle='round')
        weight = 'bold' if hi else 'normal'
        ax.text(-0.06, row.tfidf_rank, f"{row.language}  {row.tfidf_test_micro_f1 * 100:.1f}",
                ha='right', va='center', fontsize=10, color=INK if hi else INK2, weight=weight)
        ax.text(1.06, row.e5_direct_rank, f"{row.e5_direct_test_micro_f1 * 100:.1f}  {row.language}",
                ha='left', va='center', fontsize=10, color=INK if hi else INK2, weight=weight)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(8.6, 0.4)
    ax.set_xticks([0, 1], ['TF-IDF 순위', 'E5 Direct 순위'])
    ax.set_yticks([])
    ax.tick_params(length=0, labelsize=11, labelcolor=INK2)
    for s in ['left', 'bottom']:
        ax.spines[s].set_visible(False)
    ax.set_title(f'RQ1. 직접 학습 성능의 언어 순위 (Spearman ρ = {rho:+.2f})', loc='left', fontsize=13)
    save(fig, 'rq1_ranks.png')


if __name__ == '__main__':
    rq2_dumbbell()
    rq3_recovery()
    rq4_scatter()
    rq1_ranks()
