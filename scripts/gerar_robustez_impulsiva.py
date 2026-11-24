#!/usr/bin/env python3
"""Gera Figura BER × parâmetros impulsivos usando as 53.680 células auditadas.

Uso:
    python scripts/gerar_robustez_impulsiva.py

- Modelo MG; SNR adicional 18 dB; Pilot-CPE; 25 rajadas físicas comuns
- Calcula BER CONDICIONAL agregada = soma(bits errados) / soma(bits associados)
- Referência p=0.01, r=50, Lb=3 usa mesmas 25 rajadas em baseline_final
- Os segmentos isolam a variação de um parâmetro por vez.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter

METHODS = {
    'LS': 'ls',
    'LMMSE': 'lmmse',
    'SVR': 'svr_skdirect_rbf',
    'QSVR': 'qsvr_skdirect_cx_shots',
}
# Cores e marcadores próximos aos utilizados na figura AWGN do TCC
STYLE = {
    'LS': ('#505b68', 'o'),
    'LMMSE': ('#23699a', 's'),
    'SVR': ('#198c79', '^'),
    'QSVR': ('#b55029', 'D'),
}
VARIANTS = {
    'p0': 'dense_p_markov_gaussian_0p001',
    'p1': 'dense_p_markov_gaussian_0p05',
    'r0': 'dense_r_markov_gaussian_10',
    'r1': 'dense_r_markov_gaussian_250',
    'l0': 'dense_duration_markov_1',
    'l1': 'dense_duration_markov_10',
}
GROUPS = [
    ('Probabilidade $p$', (0, 1, 2), ('p0', 'base', 'p1'), ('0,001', '0,01\n(ref.)', '0,05')),
    ('Razão de variâncias $r$', (4, 5, 6), ('r0', 'base', 'r1'), ('10', '50\n(ref.)', '250')),
    ('Duração média $L_b$', (8, 9, 10), ('l0', 'base', 'l1'), ('1', '3\n(ref.)', '10')),
]

def subset(rows, *, case_id=None, method=None, campaign=None, physical_keys=None):
    out=[]
    for row in rows:
        if row['noise_family']!='markov_gaussian' or row['added_snr_db']!='18.0' or row['equalizer']!='pilot_cpe': continue
        if row['noise_replica']!='0': continue
        if method and row['method']!=method: continue
        if case_id is not None and row['case_id']!=case_id: continue
        if campaign and row['campaign']!=campaign: continue
        if physical_keys is not None and (row['session_id'],row['burst']) not in physical_keys: continue
        out.append(row)
    return out

def weighted_ber(rows):
    assert len(rows)==25, f'Eram esperadas 25 rajadas; obtidas {len(rows)}'
    return sum(int(r['observed_bit_errors']) for r in rows)/sum(int(r['matched_bits']) for r in rows)*1000

def generate(cells_path, out_base):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    with cells_path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ref_rows=subset(rows, case_id=VARIANTS['p0'], method=METHODS['LS'], campaign='tcc_priority_impulse')
    physical_keys={(r['session_id'],r['burst']) for r in ref_rows}
    assert len(physical_keys)==25
    points={}
    for label,method in METHODS.items():
        points[(label,'base')]=weighted_ber(subset(rows,case_id='',campaign='baseline_final',method=method,physical_keys=physical_keys))
        for short,case_id in VARIANTS.items():
            sub=subset(rows,case_id=case_id,campaign='tcc_priority_impulse',method=method,physical_keys=physical_keys)
            points[(label,short)]=weighted_ber(sub)
    fig,ax=plt.subplots(figsize=(9.25,4.55))
    fig.subplots_adjust(left=0.09,right=0.984,top=0.855,bottom=0.265)
    ax.set_yscale('log')
    for label in METHODS:
        color,marker=STYLE[label]
        for title,xx,variants,ticks in GROUPS:
            ax.plot(xx,[points[(label,v)] for v in variants],color=color,marker=marker,ms=5.5,lw=1.8,
                    label=label if xx[0]==0 else None)
    for xc in [3,7]: ax.axvline(xc,linewidth=0.8,color='#ccd1d7')
    for x in [1,5,9]: ax.axvspan(x-0.10,x+0.10,color='#f3f5f8',zorder=0)
    xticks=[x for _,xs,_,_ in GROUPS for x in xs]
    xlabels=[lab for _,_,_,labs in GROUPS for lab in labs]
    ax.set_xticks(xticks, xlabels)
    ax.tick_params(axis='x',labelsize=9.4,pad=6)
    ax.set_xlim(-0.35,10.35)
    ax.set_ylim(0.72,6.0)
    ax.yaxis.set_major_locator(FixedLocator([0.8,1,1.5,2,3,4,5,6]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:g}'.replace('.',',')))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel('BER condicional ($10^{-3}$)',fontsize=10.5)
    ax.grid(axis='y',which='both',alpha=.32,linewidth=.6)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=4,frameon=False,handlelength=2.0,columnspacing=2.2,fontsize=10.3)
    for title,xx,_,_ in GROUPS:
        c=sum(xx)/len(xx)
        ax.text(c,-.31,title,ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=10.4)
    fig.savefig(out_base.with_suffix('.pdf'),bbox_inches='tight',pad_inches=0.10)
    fig.savefig(out_base.with_suffix('.png'),bbox_inches='tight',pad_inches=0.10,dpi=220)
    plt.close(fig)
    with out_base.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['method','parameter','value','BER_conditional_x1e3','n_bursts','model','SNR_add_dB','equalizer'])
        for label in METHODS:
            for name,_,variants,ticks in GROUPS:
                for v,t in zip(variants,ticks):
                    w.writerow([label,name,t.replace('\n(ref.)',''),f'{points[(label,v)]:.8f}',25,'MG',18,'Pilot-CPE'])
    for label in METHODS:
        print(label, ' | '.join(f'{v}={points[(label,v)]:.4f}' for v in ['p0','base','p1','r0','r1','l0','l1']))
    print('Saídas:',out_base.with_suffix('.pdf'),out_base.with_suffix('.png'),out_base.with_suffix('.csv'))

if __name__=='__main__':
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells', type=Path, default=root/'data/campanha_final/master_cells_complete.csv')
    ap.add_argument('--out', type=Path, default=root/'results/publicacao/figuras/fig_ber_parametros_impulsivos')
    args = ap.parse_args()
    generate(args.cells, args.out)
