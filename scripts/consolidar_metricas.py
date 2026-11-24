#!/usr/bin/env python3
"""Recalcula contagens e tabelas da campanha; não executa o rádio.

Uso: python3 scripts/consolidar_metricas.py
Dependências: numpy e pandas. Caminhos relativos ao projeto.
"""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D, T = ROOT/'results/publicacao/agregados', ROOT/'results/publicacao/tabelas_auxiliares'
D5 = ROOT/'data/config'
for p in (D,T): p.mkdir(exist_ok=True, parents=True)
METHODS = ['ls','lmmse','svr_skdirect_rbf','qsvr_skdirect_cx_shots']
LABEL = dict(zip(METHODS,['LS','LMMSE','SVR','QSVR']))
EQL = {'static':'Estático','pilot_cpe':'Pilot-CPE','simpledfe':'SimpleDFE'}
def num(v,n=2): return f'{v:.{n}f}'.replace('.',',')
def sci(v):
    if v==0:return '$0$'
    e=int(np.floor(np.log10(abs(v))))
    return '$'+num(v/10**e,2)+r'\times10^{'+str(e)+'}$'

def aggregate(df,keys):
    counts=['transmitted','matched','delivered','observed_bit_errors','matched_bits',
            'expected_bits','evm_error_energy','evm_reference_energy','evm_symbols']
    g=df.groupby(keys,sort=False,dropna=False)[counts].sum().reset_index()
    g['ber_conditional']=g.observed_bit_errors/g.matched_bits
    g['per']=1-g.delivered/g.transmitted
    g['erasure_rate']=1-g.matched/g.transmitted
    g['ber_lower']=g.observed_bit_errors/g.expected_bits
    g['ber_upper']=(g.observed_bit_errors+g.expected_bits-g.matched_bits)/g.expected_bits
    g['evm_rms']=np.sqrt(g.evm_error_energy/g.evm_reference_energy)
    return g

source = ROOT / 'data/campanha_final'
cells_path = source / 'master_cells_complete.csv'
cells = pd.read_csv(cells_path, low_memory=False)
published = pd.read_csv(source / 'master_aggregates_complete.csv')
consolidation = json.loads((source / 'consolidation_completion_audit.json').read_text())

primary = {'baseline_final': 12000, 'extra_snr_21_27_33': 3600, 'tcc_dense_snr': 32400}
supplementary = {
    'tcc_priority_stability': 1200,
    'tcc_robustness_impulse_pilot': 240,
    'tcc_priority_impulse': 2000,
    'tcc_markov_duration_expansion_b02_b10_b20': 240,
    'tcc_priority_interaction': 1600,
    'tcc_priority_sensitivity': 400,
}
assert cells.campaign.value_counts().to_dict() == primary | supplementary
assert len(cells) == consolidation['master_cells'] == consolidation['matched_rows'] == 53680
assert len(published) == consolidation['master_aggregates'] == 688
assert not consolidation['unmatched_rows'] and not consolidation['missing_sources']
sources = ['session_id', 'burst']
condition = ['noise_family', 'added_snr_db', 'probability', 'variance_ratio',
             'mean_bad_samples', 'noise_replica']
configuration = condition + ['equalizer', 'method', 'svr_c', 'svr_epsilon']
assert not cells.duplicated(sources + configuration).any()
assert cells.groupby(sources).physical_source_sha256.nunique().eq(1).all()
assert cells[sources].drop_duplicates().groupby('session_id').size().tolist() == [20]*5
for field in ['replay_sha256', 'corrected_sha256', 'trigger_sha256']:
    assert cells[field].str.fullmatch(r'[a-f0-9]{64}').all()
    assert cells.groupby(sources + condition, dropna=False)[field].nunique().eq(1).all()
assert cells.transmitted.eq(100).all() and cells.expected_bits.eq(40000).all()
assert cells.matched_bits.eq(cells.matched*400).all()
assert cells.evm_symbols.eq(cells.matched*200).all()
assert cells.evm_matched_packets.eq(cells.matched).all()
assert cells.fit_failures.eq(0).all()
assert cells.delivered.le(cells.matched).all() and cells.matched.le(cells.transmitted).all()
assert cells.observed_bit_errors.le(cells.matched_bits).all()
assert cells.evm_rms.isna().eq(cells.matched.eq(0)).all()
recomputed = aggregate(cells, configuration)
expected = published.merge(recomputed, on=configuration, validate='one_to_one',
                           suffixes=('_published', '_recalculated'))
assert len(expected) == len(recomputed) == len(published)
matched_groups = cells.groupby(configuration, dropna=False).size().rename('cell_count').reset_index()
coverage = published.merge(matched_groups, on=configuration, validate='one_to_one')
assert coverage.physical_bursts.eq(coverage.cell_count).all()
for metric in ['transmitted','matched','delivered','observed_bit_errors','matched_bits',
               'expected_bits','per','ber_conditional']:
    assert np.allclose(expected[metric+'_published'], expected[metric+'_recalculated'],
                       rtol=1e-10, atol=1e-13, equal_nan=True), metric
assert np.allclose(expected.pooled_evm_rms, expected.evm_rms)
with np.load(D5/'cx_4000.npz', allow_pickle=False) as cache:
    gram=cache['gram']
    carriers=cache['carriers']
    metadata=json.loads(str(cache['metadata']))
assert gram.shape==(52,52) and np.isfinite(gram).all()
assert np.array_equal(carriers,np.r_[np.arange(-26,0),np.arange(1,27)])
assert np.allclose(gram,gram.T,atol=1e-12) and np.allclose(gram.diagonal(),1,atol=1e-12)
minimum_eigenvalue=float(np.linalg.eigvalsh(gram).min())
assert minimum_eigenvalue>=-1e-10
assert hashlib.sha256(np.ascontiguousarray(gram).tobytes()).hexdigest()==metadata['gram_sha256']
assert metadata['shots']==4000 and metadata['pairs_measured']==1326
assert metadata['read_statevector_for_training'] is False and metadata['hardware_execution'] is False

raw = cells[cells.campaign.isin(primary)].copy()
raw['condition'] = [n if pd.isna(s) else f'{n}_{s:g}dB'
                    for n, s in zip(raw.noise_family, raw.added_snr_db)]
main = raw[raw.campaign.eq('baseline_final')]
assert len(raw) == 48000 and raw.groupby(['condition','equalizer','method']).size().eq(100).all()
summary = aggregate(raw,['condition','noise_family','added_snr_db','equalizer','method'])
summary = summary.rename(columns={'noise_family':'model','added_snr_db':'snr_add_db'})
summary.to_csv(D/'resumo_varredura.csv',index=False)
sessions=aggregate(raw,['session_id','condition','equalizer','method'])
sessions.to_csv(D/'resumo_sessoes.csv',index=False)

# Retain the exact reference cohort for every supplementary scenario.
comparisons = []
comparison_keys = ['campaign', 'case_id'] + configuration
paired_keys = sources + ['noise_family', 'added_snr_db', 'equalizer', 'method']
for key, block in cells[~cells.campaign.isin(primary)].groupby(comparison_keys, dropna=False):
    baseline = block[paired_keys].merge(raw, on=paired_keys, validate='one_to_one')
    if baseline.empty:
        continue  # SimpleDFE was not run at the standard setting at 21 dB.
    assert len(baseline) == len(block)
    left = aggregate(block, ['method']).iloc[0]
    right = aggregate(baseline, ['method']).iloc[0]
    comparisons.append(dict(zip(comparison_keys, key)) | {'physical_bursts':len(block)} |
                       {f'{m}_{suffix}': row[m]
                        for m in ('ber_conditional','per','erasure_rate','evm_rms')
                        for suffix,row in (('ensaio',left),('referencia',right))})
paired = pd.DataFrame(comparisons)
paired.to_csv(D/'comparacoes_pareadas.csv',index=False)
recomputed.merge(coverage[configuration + ['physical_bursts','physical_sessions']],
                 on=configuration,validate='one_to_one').to_csv(D/'resumo_completo.csv',index=False)
report = {'cells_by_campaign':primary | supplementary,'total_cells':len(cells),
          'aggregates_recomputed':len(expected),'physical_bursts':100,'physical_sessions':5,
          'cells_without_evm_symbols':int(cells.evm_symbols.eq(0).sum()),
          'all_fit_failures_zero':True,'identical_recorded_hashes_per_capture_condition':True,
          'gram_min_eigenvalue':minimum_eigenvalue,'gram_numeric_hash_verified':True,
          'cells_csv_sha256':hashlib.sha256(cells_path.read_bytes()).hexdigest()}
(D/'auditoria_recalculada.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def table(path,headers,rows,align=None):
    align=align or ('l'+'r'*(len(headers)-1))
    txt='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    txt+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
    txt+='\n\\bottomrule\n\\end{tabular}\n'
    (T/path).write_text(txt,encoding='utf-8')

rows=[]
for cond in ['natural','awgn_12dB']:
 for m in METHODS:
    r=summary[(summary.condition==cond)&(summary.equalizer=='pilot_cpe')&(summary.method==m)].iloc[0]
    rows.append(['Natural' if cond=='natural' else 'AWGN 12 dB',LABEL[m],f'{int(r.matched):,}'.replace(',','.'),sci(r.ber_conditional),sci(r.ber_lower),sci(r.ber_upper)])
table('limites_ber.tex',['Condição','Método','Associados','BER cond.','Limite inf.','Limite sup.'],rows,'llrrrr')

# Tempos: a mediana é entre rajadas, após normalizar cada tempo pelo número
# real de chamadas (inclui aquecimento). Evita dividir pelo número 100.
timing=main[(main.condition=='natural')&(main.equalizer=='pilot_cpe')].copy()
timing['fit_ms_call']=1000*timing.fit_seconds/timing.fit_calls
records=[];rows=[]
for w in [2,6]:
 for m in METHODS:
    g=timing[(timing.parallel_workers==w)&(timing.method==m)]
    q=g.receiver_seconds.quantile([.25,.5,.75]); f=g.fit_ms_call.quantile([.25,.5,.75])
    rows.append([str(w),LABEL[m],str(len(g)),f'{num(q.loc[.5],3)} [{num(q.loc[.25],3)}; {num(q.loc[.75],3)}]',f'{num(f.loc[.5],3)} [{num(f.loc[.25],3)}; {num(f.loc[.75],3)}]'])
    records.append({'workers':w,'method':m,'n':len(g),'receiver_med_s':q.loc[.5],'fit_med_ms_per_call':f.loc[.5]})
table('custo_observado.tex',['Workers','Método','$n$','Receptor (s/rajada)','Estimador (ms/chamada)'],rows,'llrrr')
pd.DataFrame(records).to_csv(D/'custo_recalculado.csv',index=False)
rows=[]
for m in ['svr_skdirect_rbf','qsvr_skdirect_cx_shots']:
    g=timing[(timing.parallel_workers==6)&(timing.method==m)]
    rows.append([LABEL[m],num(1000*(g.training_seconds/g.fit_calls).median(),3),num(1000*(g.prediction_seconds/g.fit_calls).median(),4),num(1000*g.kernel_seconds.median(),3),num(1000*g.initialization_seconds.median(),3)])
table('custo_componentes.tex',['Método','Ajuste + previsão','Reconstrução de $H$','Kernel/cache','Inicialização'],rows)

# Cobertura dos ensaios de interação, com ambos os equalizadores.
rows = []
interaction = recomputed[(recomputed.added_snr_db==21) &
                         recomputed.variance_ratio.isin([25,1000])]
for k, group in interaction.groupby(['noise_family','probability','variance_ratio','equalizer']):
    by_method = group.set_index('method')
    model, prob, ratio, eq = k
    rows.append([{'bernoulli_gaussian':'BG','markov_gaussian':'MG'}[model],
                 num(prob,3), str(int(ratio)), EQL[eq]]+
                 [sci(by_method.loc[m,'ber_conditional'])+' / '+num(100*by_method.loc[m,'per'],2)
                  for m in METHODS])
table('interacao.tex',['Ruído','$p$','$r$','Equalizador']+[LABEL[m] for m in METHODS],
      rows,'lrrlrrrr')

sensitivity = paired[paired.campaign.eq('tcc_priority_sensitivity')]
reference = sensitivity[sensitivity.case_id.eq('sens_dense_C3')].copy()
reference['svr_c'], reference['svr_epsilon'] = 9.615, 0.02
for metric in ['ber_conditional', 'per']:
    reference[metric+'_ensaio'] = reference[metric+'_referencia']
sensitivity = pd.concat([reference,sensitivity],ignore_index=True)
rows = []
for (model,c_value,eps), group in sensitivity.groupby(['noise_family','svr_c','svr_epsilon'],sort=False):
    by_method = group.set_index('method')
    rows.append(['Natural' if model=='natural' else 'MG 18 dB',num(c_value,3),num(eps,2)]+
                [sci(by_method.loc[m,'ber_conditional_ensaio'])+' / '+
                 num(100*by_method.loc[m,'per_ensaio'],2) for m in METHODS[2:]])
rows.sort(key=lambda row: (row[0]!='Natural',float(row[1].replace(',','.')),row[2]))
table('sensibilidade.tex',['Condição','$C$',r'$\epsilon$','SVR','QSVR'],rows,'lrrrr')
print(json.dumps(report,ensure_ascii=False,indent=2))
