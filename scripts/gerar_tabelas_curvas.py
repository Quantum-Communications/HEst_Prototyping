"""Reproduz as figuras, tabelas e verificações da monografia, sem executar o receptor."""
from pathlib import Path

import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

root = Path(__file__).resolve().parents[1]
figdir, tabdir, auditdir = (root / p for p in ("results/publicacao/figuras", "results/publicacao/tabelas", "results/publicacao/dados_figuras"))
for directory in (figdir, tabdir, auditdir):
    directory.mkdir(exist_ok=True, parents=True)

blue, rust, teal, plum = "#24557a", "#ae491f", "#187664", "#6c4e78"
plt.rcParams.update({"font.size": 9.2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.axisbelow": True, "axes.labelsize": 9.3,
                     "legend.frameon": False, "lines.linewidth": 1.6})
def save(fig, name):
    fig.savefig(figdir / (name + ".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
def comma(x, pos=None):
    return f"{x:g}".replace(".", ",")
def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")
def axis(a):
    a.yaxis.set_major_formatter(FuncFormatter(comma))
    a.grid(axis="y", alpha=.2)
def title(a, t):
    a.set_title(t, loc="left", fontsize=9.8, fontweight="bold", pad=8)

# A projeção de uma portadora retangular em frequência deslocada é uma sinc.
f = np.linspace(-1.7, 1.7, 600)
fig, (a,b) = plt.subplots(1, 2, figsize=(6.5, 2.55), layout="constrained", gridspec_kw={"width_ratios":[1.36,1]})
for k, color in zip((-1,0,1),(rust,blue,teal)):
    a.plot(f, abs(np.sinc(f-k)), color=color, label=f"k = {k}")
    a.scatter([k],[1], s=16, color=color)
a.set(xlabel=r"Frequência normalizada $f T_u$",ylabel="Módulo normalizado",ylim=(-.045,1.15),xticks=[-1,0,1])
a.legend(loc="upper center",ncol=3,bbox_to_anchor=(.48,-.29),fontsize=8.2,handlelength=1.1,columnspacing=.8)
for x in [-1,0,1]: a.axvline(x, lw=.7,color="#8e9398",ls=":",zorder=0)
title(a,"(a) Espectros sobrepostos")
offsets = np.array([-1,0,1])
b.bar(offsets-.17, abs(np.sinc(offsets)),width=.31,color=blue,label="Grade alinhada")
b.bar(offsets+.17, abs(np.sinc(offsets-.25)),width=.31,color=rust,label="Desvio de ¼ de Δf")
b.set(xlabel="Índice da portadora transmitida",xticks=offsets,ylabel="Módulo da projeção",ylim=(0,1.15))
b.legend(loc="upper center",bbox_to_anchor=(.49,-.29),fontsize=8.2)
title(b,"(b) Saída do receptor em k = 0")
for a0 in (a,b):axis(a0)
save(fig,"ortogonalidade")

# Valores escolhidos permitem acompanhar amostras individuais em dois percursos.
N, cp, delay = 12,3,2
x=np.array([.82,.31,-.5,-.85,.06,.50,.74,-.25,-.65,.65,-.48,.21])
n=np.arange(-cp,N); tx=x[n%N]; u=np.arange(N)
fig,(a,b)=plt.subplots(2,1,figsize=(6.5,3.7),sharex=True,layout="constrained")
for ax in (a,b):
    ax.axvspan(-3.5,-.5,color="#e7edf1"); ax.axvline(-.5,color="#6d7882",lw=.9)
    ax.axhline(0,color="#85909b",lw=.6);axis(ax);ax.set_xlim(-3.6,11.6)
a.plot(n,tx,"o-",color=blue,ms=4.1,lw=1.0)
for t, color, marker in zip(range(9,12),(rust,teal,plum),("s","D","^")):
    a.plot([t-N,t],[x[t],x[t]],ls="",marker=marker,color=color,ms=6.5)
a.set(ylabel=r"TX · $x^{CP}[n]$",ylim=(-1.05,1.13));title(a,"(a) O bloco transmitido começa com as três últimas amostras do símbolo")
b.plot(u,x, "o-",color=blue,ms=3.3,lw=1,label="Direto")
echo=.55*x[(u-delay)%N]
b.plot(u,echo,"s--",color=rust,ms=3.2,lw=1.1,label="Eco com atraso 2 e ganho 0,55")
b.plot(u,x+echo,"D-",color=teal,ms=3.6,lw=1.55,label="Soma na janela da FFT")
b.set(ylabel="RX · amplitude",xlabel="Índice da amostra n",xticks=np.arange(-3,12),ylim=(-1.32,1.4))
b.text(-2, .46,"Prefixo\ndescartado",ha="center",va="center",fontsize=8)
b.legend(loc="upper center",bbox_to_anchor=(.50,-.31),ncol=3,fontsize=8.2,columnspacing=1.5)
title(b,"(b) Na janela útil, as duas contribuições pertencem ao mesmo símbolo")
save(fig,"prefixo")

cache=np.load(root/"data/config/cx_4000.npz",allow_pickle=False)
k=cache["carriers"]; Gq=cache["gram"]; i=np.where(k==6)[0][0]
ell=np.arange(16); p=.7**ell; p/=p.sum()
F=np.exp(-2j*np.pi*k[:,None]*ell/64)
R=(F*p)@F.conj().T
W=np.linalg.solve(R+.03*np.eye(len(k)),R).T.conj()
# A inversa espectral e a multiplicação à direita devem coincidir.
evals,U=np.linalg.eigh(R)
assert np.allclose(W,(U*(np.maximum(evals,0)/(np.maximum(evals,0)+.03)))@U.conj().T)
fig,axes=plt.subplots(2,1,figsize=(6.3,3.5),sharex=True,layout="constrained")
for ax, y, col, name, yl in zip(axes,[abs(R[i]),abs(W[i])],[blue,rust],
        ["(a) Correlação prevista pelo perfil de potência","(b) Contribuição de cada observação à estimativa em k = 6"],
        [r"$|R[6,k]|$",r"$|W[6,k]|$"]):
    for region in (k<0,k>0):ax.plot(k[region],y[region],"o-",ms=2.7,lw=1,color=col)
    ax.axvline(6,color="#74808b",lw=.9,ls="--");ax.axvspan(-.4,.4,color="#dce1e4")
    ax.set(ylabel=yl,ylim=(0,max(y)*1.14));title(ax,name);axis(ax)
axes[1].set(xlabel="Índice k da subportadora observada",xticks=[-26,-18,-10,0,6,14,22,26])
save(fig,"lmmse")

harm=np.array([1,2,4,8,16]);phase=2*np.pi*k[:,None]*harm/64
feat=np.concatenate((np.sin(phase),np.cos(phase)),axis=1)
Kr=np.exp(-.125*np.sum((feat[:,None]-feat[None,:])**2,axis=-1))
fig,ax=plt.subplots(figsize=(6.4,2.65),layout="constrained")
for mat,c,m,l in [(Kr,blue,"o","Kernel RBF"),(Gq,rust,"s","Kernel de fidelidade")]:
    for reg in (k<0,k>0):
        ax.plot(k[reg],mat[i,reg],marker=m,color=c,ms=3.0,lw=1.15,label=l if reg[0] else None)
ax.axvline(6,color="#747e86",ls=":",lw=1)
ax.axvspan(-.4,.4,color="#dce1e4")
ax.set(ylim=(-.045,1.10),ylabel=r"Similaridade $K[6,k]$",xlabel="Índice k da segunda subportadora",xticks=[-26,-18,-10,0,6,14,22,26])
ax.legend(ncol=2,loc="upper left",fontsize=8.5);axis(ax)
save(fig,"similaridades")
pd.DataFrame({"k":k,"R_mod_k6":abs(R[i]),"W_mod_k6":abs(W[i]),"K_RBF_k6":Kr[i],"K_Q_k6":Gq[i]}).to_csv(auditdir/"linhas_matrizes.csv",index=False)

methods=["ls","lmmse","svr_skdirect_rbf","qsvr_skdirect_cx_shots"]
names=dict(zip(methods,["LS","LMMSE","SVR","QSVR"]))
eqs={"static":"Estático","pilot_cpe":"Pilot-CPE","simpledfe":"SimpleDFE"}
noise={"awgn":"AWGN","bernoulli_gaussian":"BG","markov_gaussian":"MG"}
pri=["baseline_final","extra_snr_21_27_33","tcc_dense_snr"]
source = root / "data/campanha_final"
cells_path = source / "master_cells_complete.csv"
df = pd.read_csv(cells_path, low_memory=False)
published = pd.read_csv(source / "master_aggregates_complete.csv")
primary=df[df.campaign.isin(pri)].copy()
summ=pd.read_csv(root/"results/publicacao/agregados/resumo_varredura.csv")
rawkeys=["noise_family","added_snr_db","equalizer","method"]
primary["noise_family"]=primary.noise_family.fillna("natural")
counts=["transmitted","matched","delivered","observed_bit_errors","matched_bits","expected_bits"]
def aggregate(rows, keys):
    s=rows.groupby(keys,dropna=False)[counts].sum().reset_index()
    s["ber"]=s.observed_bit_errors/s.matched_bits
    s["per"]=1-s.delivered/s.transmitted
    return s
v=aggregate(primary,rawkeys)
assert len(v)==len(summ)==480
for item in v.itertuples():
    grp=summ[(summ.model==item.noise_family)&(summ.equalizer==item.equalizer)&(summ.method==item.method)]
    grp=grp[grp.snr_add_db.isna()] if pd.isna(item.added_snr_db) else grp[grp.snr_add_db==item.added_snr_db]
    assert len(grp)==1 and np.isclose(item.ber,grp.iloc[0].ber_conditional) and item.matched==grp.iloc[0].matched
def table(name,columns,headers,rows,breaks=()):
    strs=[r"\begin{tabular}{"+columns+"}",r"\toprule"," & ".join(headers)+r" \\",r"\midrule"]
    for pos,row in enumerate(rows):
        if pos in breaks:strs.append(r"\addlinespace[3pt]")
        strs.append(" & ".join(map(str,row))+r" \\")
    strs.extend([r"\bottomrule",r"\end{tabular}"])
    (tabdir/(name+".tex")).write_text("\n".join(strs)+"\n")
def met(r):return [fmt(1000*r.ber,3),fmt(100*r.per),f"{int(r.matched):,}".replace(",",".")]
labels=[r"BER ($10^{-3}$)",r"PER (\%)",r"$N_{\rm assoc}$"]
natural=v[v.noise_family=="natural"].set_index(["equalizer","method"])
table("natural","llrrr",["Equalizador","Método"]+labels,
      [[eqs[e] if j==0 else "",names[m]]+met(natural.loc[e,m]) for e in eqs for j,m in enumerate(methods)],breaks=[4,8])
q=methods[-1]
snrs=v[(v.equalizer=="pilot_cpe")&(v.method==q)].set_index(["noise_family","added_snr_db"])
table("modelos","rlrrr",["SNR adicional","Modelo"]+labels,
      [[str(n)+r"~dB" if j==0 else "",noise[model]]+met(snrs.loc[model,n]) for n in (12,24) for j,model in enumerate(noise)],breaks=[3])
fig,ax=plt.subplots(figsize=(6.55,3.25),layout="constrained")
for m,col,mark in zip(methods,["#4e5661",blue,teal,rust],["o","s","^","D"]):
    pts=v[(v.equalizer=="pilot_cpe")&(v.noise_family=="awgn")&(v.method==m)].sort_values("added_snr_db")
    ax.plot(pts.added_snr_db,pts.ber,color=col,marker=mark,ms=3.8,lw=1.4,label=names[m])
ax.set(yscale="log",xlabel="SNR adicional (dB)",ylabel="BER condicional",xticks=[12,15,18,21,24,27,30,33],xlim=(11.5,33.5),ylim=(.0003,.036))
ax.legend(ncol=4,loc="upper right");ax.grid(axis="y",which="both",alpha=.18)
save(fig,"pilot_awgn")
# As contagens são razões agrupadas por sessão. A comparação entre rajadas é pareada por hash.
w=primary[(primary.noise_family=="awgn")&(primary.added_snr_db==31.5)&(primary.equalizer=="simpledfe")].copy()
assert len(w)==400 and w.physical_source_sha256.nunique()==100
sess=aggregate(w,["session_id","method"])
sess.to_csv(auditdir/"simpledfe_31p5_sessoes.csv",index=False)
allm=aggregate(w,["method"]).set_index("method")
stats=[]
for m in methods:
    s=sess[sess.method==m]
    stats.append([names[m]]+met(allm.loc[m])+[fmt(1000*s.ber.min(),3)+r" a "+fmt(1000*s.ber.max(),3)])
table("dfe","lrrrr",["Método"]+labels+[r"BER por sessão ($10^{-3}$)"],stats)
w["ber"]=w.observed_bit_errors/w.matched_bits
pair=w.pivot(index=["session_id","burst","physical_source_sha256"],columns="method",values="ber")
pair["diferenca_QSVR_menos_LS"]=pair[q]-pair["ls"]
pair.to_csv(auditdir/"simpledfe_31p5_rajadas.csv")
profiles={}
for e in eqs:
    sub=v[v.equalizer==e].copy()
    sub["snr"]=sub.added_snr_db.fillna(-1)
    ranks={}
    for metric in ["ber","per"]:
        wide=sub.pivot(index=["noise_family","snr"],columns="method",values=metric)
        ranks[metric]={"ordens":{},
            "vencedores":{names[m]:int(n) for m,n in wide.idxmin(axis=1).value_counts().items()},
            "empates_no_minimo":int((wide.eq(wide.min(axis=1),axis=0).sum(axis=1)>1).sum()),
            "Q_menor_LS":int((wide[q]<wide.ls).sum()),"Q_menor_SVR":int((wide[q]<wide[methods[2]]).sum())}
        order=wide.apply(lambda row:" < ".join(names[m] for m in row.sort_values().index),axis=1)
        ranks[metric]["ordens"]=order.value_counts().to_dict()
    profiles[e]=ranks
d=pair["diferenca_QSVR_menos_LS"].dropna()
pr=primary[(primary.noise_family=="natural")&(primary.equalizer=="pilot_cpe")].copy()
pr["ms"]=1000*pr.fit_seconds/pr.fit_calls
cost=pr.groupby(["method","parallel_workers"])[["ms","receiver_seconds"]].median()
table("custo","lrrrr",["Método",r"\multicolumn{2}{c}{Estimador (ms/chamada)}",r"\multicolumn{2}{c}{Receptor (s/rajada)}"],
      [["","2 processos","6 processos","2 processos","6 processos"]]+[[names[m],fmt(cost.loc[m,2].ms,3),fmt(cost.loc[m,6].ms,3),fmt(cost.loc[m,2].receiver_seconds,3),fmt(cost.loc[m,6].receiver_seconds,3)] for m in methods])
campaigns=df.groupby("campaign").agg(celulas=("method","size"),rajadas=("physical_source_sha256","nunique")).reset_index()
campaigns.to_csv(auditdir/"campanhas.csv",index=False)
aud={"cells_csv_sha256":hashlib.sha256(cells_path.read_bytes()).hexdigest(),"celulas":len(df),"agregados":len(published),"condicoes_principais":len(v)//4,"ordens":profiles,
     "pares_rajadas_validos":len(d),"Q_menor":int((d<0).sum()),"Q_igual":int((d==0).sum()),"Q_maior":int((d>0).sum()),
     "diferenca_p25_mediana_p75":d.quantile([.25,.5,.75]).tolist(),"diferenca_agrupada":allm.loc[q].ber-allm.loc["ls"].ber}
(auditdir/"checagens_v8.json").write_text(json.dumps(aud,indent=2,ensure_ascii=False)+"\n")

# Compara os níveis na mesma lista de fontes físicas; uma linha por fonte e método.
imp=df[(df.campaign=="tcc_priority_impulse")&(df.noise_family=="markov_gaussian")&
       (df.added_snr_db==18)&(df.method==q)&(df.equalizer=="pilot_cpe")]
source_sets=imp.groupby("case_id").physical_source_sha256.agg(frozenset)
assert len(set(source_sets))==1 and len(source_sets.iloc[0])==25
base_mg=primary[(primary.noise_family=="markov_gaussian")&(primary.added_snr_db==18)&
                (primary.method==q)&(primary.equalizer=="pilot_cpe")&
                (primary.physical_source_sha256.isin(source_sets.iloc[0]))]
assert len(base_mg)==25
samples=pd.concat([base_mg.assign(case_id="referencia"),imp])
ip=aggregate(samples,["case_id","probability","variance_ratio","mean_bad_samples"]).set_index("case_id")
ip.to_csv(auditdir/"impulsivos_25_rajadas.csv")
settings=["referencia","dense_p_markov_gaussian_0p001","dense_p_markov_gaussian_0p05",
          "dense_r_markov_gaussian_10","dense_r_markov_gaussian_250",
          "dense_duration_markov_1","dense_duration_markov_10"]
table("impulsivo","lrrrrr",["Parâmetro",r"$p$",r"$r$",r"$L_b$","BER ($10^{-3}$)",r"PER (\%)"],
      [[label,fmt(ip.loc[key].probability,3),int(ip.loc[key].variance_ratio),int(ip.loc[key].mean_bad_samples),
        fmt(1000*ip.loc[key].ber,3),fmt(100*ip.loc[key].per)] for key,label in zip(settings,["Referência","Probabilidade","","Intensidade","","Duração",""])])

evidence=summ[(summ.equalizer=="pilot_cpe")&((summ.model=="natural")|((summ.model=="awgn")&(summ.snr_add_db==12)))].copy()
evmrows=[]
for model,desc in [("natural","Natural"),("awgn","AWGN 12 dB")]:
    rr=evidence[evidence.model==model].set_index("method")
    for pos,m in enumerate(methods):
        row=rr.loc[m]
        evmrows.append([desc if pos==0 else "",names[m],f"{int(row.matched):,}".replace(",","."),
            f"{int(row.evm_symbols):,}".replace(",","."),fmt(100*row.evm_rms),
            fmt(1000*row.ber_conditional,3),fmt(100*row.per)])
table("evm","llrrrrr",["Condição","Método",r"$N_{\rm assoc}$","Símbolos",r"EVM (\%)",r"BER ($10^{-3}$)",r"PER (\%)"],evmrows,breaks=[4])
# Igualdade dos hashes e dos 688 agregados é verificada por consolidar_metricas.py.
def ranking_group(rows, keys):
    wide=aggregate(rows,keys+["method"]).pivot(index=keys,columns="method",values="ber")
    return wide.idxmin(axis=1).value_counts().to_dict()
inter=df[df.campaign=="tcc_priority_interaction"]
assert ranking_group(inter[inter.equalizer=="simpledfe"],["case_id"])[q]==3
for metric in ["ber","per"]:
    ranks=aggregate(inter[inter.equalizer=="pilot_cpe"],["case_id","method"]).pivot(index="case_id",columns="method",values=metric)
    assert len(ranks)==8 and (ranks["lmmse"]<ranks[q]).all() and (ranks[q]<ranks[methods[2]]).all() and (ranks[methods[2]]<ranks["ls"]).all()
    comp=v.pivot(index=["noise_family","added_snr_db","method"],columns="equalizer",values=metric).dropna()
    if metric=="ber":assert (comp.pilot_cpe<comp.simpledfe).all()
    else:
        for m in methods:
            tt=comp.xs(m,level="method")
            assert len(tt)==37 and ((tt.pilot_cpe<tt.simpledfe) if m=="lmmse" else (tt.pilot_cpe>tt.simpledfe)).all()
assert [profiles["simpledfe"]["ber"]["vencedores"][m] for m in ["LS","SVR","QSVR"]]==[18,10,9]
assert (len(d),int((d<0).sum()),int((d==0).sum()),int((d>0).sum()))==(100,27,58,15)
