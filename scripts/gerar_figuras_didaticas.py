#!/usr/bin/env python3
"""Regenera as cinco figuras herdadas com fonte numérica ou analítica."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, Circle

from matplotlib.colors import to_rgb
from sklearn.svm import SVR
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'results/publicacao/figuras';D=ROOT/'data/config'
F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
 'axes.titlesize':11,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
 'savefig.bbox':'tight','lines.linewidth':1.6})
BLUE='#0072B2';PURPLE='#8338A3';ORANGE='#D55E00';GRAY='#525D6B';GREEN='#00876C'
def save(fig,n):
    for ext in ('pdf',):
        target=F/(n+'.'+ext)
        temp=target.with_name(target.name+'.tmp')
        fig.savefig(temp,format=ext,bbox_inches='tight',pad_inches=.08,dpi=180)
        temp.replace(target)
    plt.close(fig)
def diagram(width=10,height=5,figsize=(7.2,3.8)):
    fig,ax=plt.subplots(figsize=figsize);ax.set_xlim(0,width);ax.set_ylim(0,height);ax.axis('off')
    return fig,ax
def box(ax,x,y,w,h,text,color=BLUE,fs=9,fill=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.06',
                 facecolor=tuple(.94+.06*c for c in to_rgb(color)) if fill else 'white',edgecolor=color,lw=1.2,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color='#18232C',zorder=3)

# Constelações com a mesma convenção matemática do texto.
fig,axs=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
for ax in axs:
    ax.axhline(0,color='#BCC3C9',lw=.8);ax.axvline(0,color='#BCC3C9',lw=.8)
    ax.set_xlim(-1.45,1.45);ax.set_ylim(-1.4,1.4);ax.set_aspect('equal')
    ax.set_xlabel('Componente I');ax.set_ylabel('Componente Q');ax.grid(alpha=.15)
axs[0].scatter([-1,1],[0,0],color=BLUE,s=65)
for x,s in [(-1,'0'),(1,'1')]:axs[0].annotate(s,(x,0),xytext=(0,11),textcoords='offset points',ha='center')
axs[0].set_title('BPSK: um bit por símbolo')
for b0,b1 in [(0,0),(1,0),(0,1),(1,1)]:
    z=((2*b0-1)+1j*(2*b1-1))/np.sqrt(2)
    axs[1].scatter(z.real,z.imag,color=PURPLE,s=65)
    axs[1].annotate(f'{b0}{b1}',(z.real,z.imag),xytext=(0,10),textcoords='offset points',ha='center')
axs[1].set_title('QPSK: dois bits por símbolo')
save(fig,'constelacoes')

# Faixa epsilon: regressor real didático.
rng=np.random.default_rng(17);xx=np.linspace(-2.6,2.6,30)
yy=.5*np.sin(xx)+.13*rng.normal(size=len(xx));mdl=SVR(C=2,epsilon=.12,gamma=.6).fit(xx[:,None],yy)
xxd=np.linspace(-2.8,2.8,500);pred=mdl.predict(xxd[:,None]);fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
ax.fill_between(xxd,pred-.12,pred+.12,color=BLUE,alpha=.13,label=r'Tolerância $\pm\epsilon$')
ax.plot(xxd,pred,color=BLUE,label='Função ajustada');ax.scatter(xx,yy,color=GRAY,s=18,label='Observações')
ax.scatter(xx[mdl.support_],yy[mdl.support_],facecolor='none',edgecolor=ORANGE,s=65,label='Vetores de suporte')
ax.set_xlabel('Entrada');ax.set_ylabel('Alvo real');ax.legend(ncol=2,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02));ax.grid(alpha=.16)
save(fig,'tubo_svr')

# Circuito exato antes de transpilar: cada CX ocupa uma coluna temporal.
fig,ax=diagram(width=13,height=6.1,figsize=(7.35,4.0))
ys=np.arange(5,0,-1)
cols=[1.35,2.75,4.25,5.65,6.45,7.25,8.05,9.6,11.4]
for j,y in enumerate(ys):
    ax.plot([.6,12.65],[y,y],color=GRAY,lw=.8,zorder=0)
    ax.text(.1,y,rf'$q_{j}$',va='center',fontsize=10)
    for x,w,label in [(cols[0],.6,'H'),(cols[1],1.05,rf'$P(\phi_{j})$'),(cols[2],1.25,rf'$R_y(a_{j})$'),(cols[-2],1.25,rf'$R_x(b_{j})$'),(cols[-1],1.25,rf'$R_z(c_{j})$')]:box(ax,x-w/2,y-.25,w,.5,label,PURPLE,fs=8.5)
for j,x in enumerate(cols[3:7]):
    ax.plot([x,x],[ys[j],ys[j+1]],color=PURPLE,lw=1.3)
    ax.add_patch(Circle((x,ys[j]),.07,color=PURPLE));ax.add_patch(Circle((x,ys[j+1]),.14,edgecolor=PURPLE,facecolor='white',lw=1.3))
    ax.plot([x-.14,x+.14],[ys[j+1],ys[j+1]],color=PURPLE,lw=1.2)
    ax.plot([x,x],[ys[j+1]-.14,ys[j+1]+.14],color=PURPLE,lw=1.2)
ax.text(6.6,5.75,'Preparação U(k): leitura da esquerda para a direita',ha='center',fontsize=10)
ax.text(6.5,.22,r'$a_j=0{,}50\sin\phi_j\quad b_j=0{,}35\cos\phi_j\quad c_j=0{,}25\phi_j$',ha='center',fontsize=10)
save(fig,'circuito_cx')

# Ilustração geométrica de canal + CPE; ZF ideal apenas para separar efeitos.
rng=np.random.default_rng(812);book=np.array([1+1j,-1+1j,-1-1j,1-1j])/np.sqrt(2)
idx=rng.integers(0,4,480);sym=book[idx];carrier=np.tile(np.arange(48),10)
hh=.9*(1+.22*np.cos(2*np.pi*carrier/48))*np.exp(1j*(.45+.3*np.sin(2*np.pi*carrier/48)))
theta=.32;vv=.07*(rng.normal(size=480)+1j*rng.normal(size=480));received=sym*hh*np.exp(1j*theta)+vv
stages=[sym,received,received/hh,received/hh*np.exp(-1j*theta)]
titles=['(a) QPSK transmitida','(b) Canal + fase comum + ruído','(c) ZF: compensa H[k]','(d) ZF + correção de fase comum']
fig,axs=plt.subplots(2,2,figsize=(7.1,5.4),layout='constrained')
for ax,z,title in zip(axs.ravel(),stages,titles):
    ax.scatter(z.real,z.imag,c=idx,cmap='tab10',s=5,alpha=.6)
    ax.scatter(book.real,book.imag,marker='+',c='black',s=55)
    ax.set_title(title,fontsize=10);ax.set_aspect('equal');ax.set_xlim(-1.7,1.7);ax.set_ylim(-1.7,1.7)
    ax.set_xlabel('I');ax.set_ylabel('Q');ax.grid(alpha=.15)
save(fig,'equalizacao_complexa')

# Antenas: leitura dos dados originais do NanoVNASaver.
antenna=[]
for i in range(1,5):
    stem='antena_3_3' if i==3 else f'antena_{i}'
    vs=np.loadtxt(D/'antenas'/(stem+'_vswr.dat'),comments='#')
    zi=np.loadtxt(D/'antenas'/(stem+'_impedancia.dat'),comments='#')
    antenna.append((i,vs,zi))
colors=[BLUE,GREEN,PURPLE,ORANGE]
fig,axs=plt.subplots(2,1,figsize=(7.1,4.6),sharex=True,layout='constrained')
for (i,vs,zi),c in zip(antenna,colors):
    f=vs[:,0]/1e6 if vs[:,0].max()>1e5 else vs[:,0]
    gamma=(vs[:,1]-1)/(vs[:,1]+1)
    axs[0].plot(f,vs[:,1],c=c,label=f'Antena {i}');axs[1].plot(f,20*np.log10(np.maximum(gamma,1e-10)),c=c)
for ax in axs:ax.axvline(907.98,c=GRAY,ls=':',lw=1);ax.grid(alpha=.18)
axs[0].legend(ncol=4,frameon=False);axs[0].set_ylabel('VSWR');axs[1].set_ylabel(r'$20\log_{10}|S_{11}|$ (dB)');axs[1].set_xlabel('Frequência (MHz)')
save(fig,'antenas_frequencia')
