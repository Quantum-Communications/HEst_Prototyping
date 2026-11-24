from pathlib import Path
import sys
from collections import Counter
import numpy as np
from sklearn.svm import SVR
from ofdm_benchmark.config import PHY, Noise
from ofdm_benchmark.estimators import Estimator, METHODS
from ofdm_benchmark.phy import ACTIVE, ACTIVE_K, sync_words
from ofdm_benchmark.estimators.quick_sklearn import load_custom_gram
from ofdm_benchmark.noise import generate
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from tcc_campaign import read_plan, units, methods, config_for
from ofdm_benchmark.config import load

def test_plan():
    rows=read_plan();assert len(rows)==53680
    assert len({(r['session_id'],r['burst']) for r in rows})==100
    assert set(r['method'] for r in rows)==set(METHODS)
    assert sum(len(v) for v in units(rows).values())==53680
    cfg=load(ROOT/'configs/tcc.example.json')
    for key, records in units(rows).items():
        new=config_for(cfg,key,records,None)
        assert len(new.conditions)==1 and set(new.equalizers)=={r['equalizer'] for r in records}
        assert new.seed==cfg.seed+int(key[8])*1000003
        assert len({(r['method'],r['equalizer']) for r in records})==len(records)

def test_estimators():
    rng=np.random.default_rng(194); phy=PHY(); first,second=sync_words()
    x=second[ACTIVE].real
    k=ACTIVE_K; angles=2*np.pi*k[:,None]*np.array([1,2,4,8,16])/64
    z=np.stack((np.sin(angles),np.cos(angles)),axis=-1).reshape(52,-1)
    kr=np.exp(-.125*np.sum((z[:,None]-z[None,:])**2,axis=-1))
    kq,_=load_custom_gram(ROOT/'data/config/cx_4000.npz','cx')
    for _ in range(12):
        raw=np.stack([first,second,np.zeros(64)]).astype('c8');raw[1,ACTIVE]+=rng.normal(0,.08,52)+1j*rng.normal(0,.08,52)
        y=raw[1,ACTIVE].astype(complex);s=max(np.sqrt(np.mean(abs(y)**2)),1e-12)
        for m in methods():
            model=Estimator(m,phy); h,offset=model.estimate(raw);assert offset==0 and h.shape==(64,) and np.isfinite(h).all()
            if m.name=='ls': assert np.allclose(h[ACTIVE],y/x,atol=2e-6)
            elif m.name=='lmmse':
                F=np.exp(-2j*np.pi*k[:,None]*np.arange(16)/64);Fc=np.exp(-2j*np.pi*(np.arange(64)-32)[:,None]*np.arange(16)/64);p=.7**np.arange(16);p/=p.sum();cov=(F*p)@F.conj().T;expected=(Fc*p)@F.conj().T@np.linalg.solve(cov+.03*np.eye(52),y/x);assert np.allclose(h,expected,atol=2e-6)
            else:
                kernel=kr if m.name=='svr_skdirect_rbf' else kq;g=x[:,None]*kernel*x[None,:];a=SVR(kernel='precomputed',C=9.615,epsilon=.02).fit(g,(y/s).real).predict(g);b=SVR(kernel='precomputed',C=9.615,epsilon=.02).fit(g,(y/s).imag).predict(g);assert np.allclose(h[ACTIVE],s*x*(a+1j*b),atol=2e-6);assert np.count_nonzero(np.delete(h,ACTIVE))==0

def test_noise():
    for name in ('awgn','bernoulli_gaussian','markov_gaussian'):
        n=Noise(name,18);v,_=generate(40960,n,123,.01);v2,_=generate(40960,n,123,.01);assert np.array_equal(v,v2);assert np.isclose(np.mean(abs(v)**2),.01*10**(-1.8),rtol=2e-6)
