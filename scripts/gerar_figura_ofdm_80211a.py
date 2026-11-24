#!/usr/bin/env python3
"""Figura 5.2 - alocação OFDM do protótipo (FFT 64, fs 8 MS/s).

Reprodutível: python gerar_figura_ofdm_80211a.py
Requer: numpy, matplotlib. Saídas: figura_ofdm_80211a.pdf/.png.
O painel espectral mostra um símbolo sintético, NÃO uma captura de RF.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FFT_SIZE = 64
FS = 8e6
PILOTS = {-21, -7, 7, 21}
ACTIVE = set(range(-26, 0)) | set(range(1, 27))
DATA = ACTIVE - PILOTS
BINS = np.arange(-32, 32)
NULLS = set(BINS) - ACTIVE
assert (len(DATA), len(PILOTS), len(NULLS)) == (48, 4, 12)

# Símbolo didático, com dados QPSK e pilotos BPSK; nenhuma captura experimental.
rng = np.random.default_rng(80211)
X = np.zeros(FFT_SIZE, dtype=np.complex128)  # ordem NATIVA da FFT: k mod N
qpsk = np.array([-1-1j, -1+1j, 1-1j, 1+1j])  # energia média 2 (convenção didática)
for k in sorted(DATA):
    X[k % FFT_SIZE] = qpsk[rng.integers(0, len(qpsk))]
for k in sorted(PILOTS):
    X[k % FFT_SIZE] = 1 if rng.integers(0, 2) else -1

x = np.fft.ifft(X)  # Não aplicar ifftshift: X já está na ordem da FFT.
# FFT com preenchimento de zeros apenas para interpolação visual do espectro.
N_FINE = 16384
spectrum = np.fft.fftshift(np.fft.fft(x, n=N_FINE))
freq_mhz = np.fft.fftshift(np.fft.fftfreq(N_FINE, d=1/FS)) / 1e6
amp_db = 20*np.log10(np.maximum(np.abs(spectrum), 1e-12) / np.max(np.abs(spectrum)))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42,
})
C_DATA = '#2563a6'
C_PILOT = '#d75b28'
C_NULL = '#d3d7dd'
C_SPEC = '#243a59'
fig = plt.figure(figsize=(10.4, 5.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 2.2], hspace=0.34,
                      top=0.865, bottom=0.14, left=0.095, right=0.975)
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

# Ocupação por subportadora com 64 células e limites visíveis.
colors = [C_PILOT if k in PILOTS else (C_DATA if k in DATA else C_NULL) for k in BINS]
ax1.bar(BINS, np.ones(FFT_SIZE), width=0.91, color=colors, linewidth=0.28,
        edgecolor='#f8f8f8', align='center')
ax1.set_xlim(-32.7, 31.7); ax1.set_ylim(-0.05, 1.55)
ax1.set_yticks([])
ax1.set_xticks([-32,-26,-21,-7,0,7,21,26,31])
ax1.set_xticklabels(['−32','−26','−21','−7','0','7','21','26','31'], fontsize=8.5)
ax1.tick_params(axis='x', length=0, pad=2)
for k in [-21,-7,7,21]:
    ax1.annotate('', (k,1.05), (k,1.36),
                 arrowprops=dict(arrowstyle='-|>', color=C_PILOT, linewidth=1))
ax1.text(0, -0.33, 'Índice k da subportadora', ha='center', va='center', fontsize=9.5)
ax1.spines['left'].set_visible(False); ax1.spines['bottom'].set_visible(False)
fig.legend(handles=[Patch(facecolor=C_DATA, label='Dados (48)'),
                    Patch(facecolor=C_PILOT, label='Pilotos (4)'),
                    Patch(facecolor=C_NULL, label='Nulas (12)')],
           frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.54, 0.982), fontsize=9.5)

# Espectro sintético em relação à portadora, amostragem de 8 MS/s.
ax2.plot(freq_mhz, amp_db, color=C_SPEC, linewidth=1.12)
ax2.set_xlim(-4,4); ax2.set_ylim(-52,3)
ax2.set_xticks(np.arange(-4,5,1))
ax2.set_yticks([-50,-40,-30,-20,-10,0])
ax2.grid(axis='y', linewidth=.45, alpha=.3)
ax2.set_xlabel('Frequência relativa à portadora (MHz)')
ax2.set_ylabel('Magnitude normalizada (dB)')
ax2.set_title('Espectro de um símbolo OFDM sintético', fontsize=10.5, loc='left', pad=8)
# Bandas ocupadas em k = ±26; marca de frequência baseada em delta-f = 125 kHz.
for k in sorted(PILOTS):
    ax2.axvline(k*FS/FFT_SIZE/1e6, color=C_PILOT, alpha=.24, lw=.9, ls='--')
ax2.text(0.98, 0.97, 'Δf = 125 kHz', transform=ax2.transAxes,
         ha='right', va='top', color='#444444', fontsize=9,
         bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))

out = Path(__file__).resolve().parents[1] / "results/publicacao/figuras"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out/'figura_ofdm_80211a.pdf', facecolor='white')
fig.savefig(out/'figura_ofdm_80211a.png', dpi=240, facecolor='white')
plt.close(fig)
print('Gerados:', out/'figura_ofdm_80211a.pdf', out/'figura_ofdm_80211a.png')
