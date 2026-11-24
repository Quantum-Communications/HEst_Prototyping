# Autoria e atribuições

Caio Neves Silva, CEFET/RJ. Este repositório reúne o código do protótipo QML – Channel Estimation Prototyping utilizado no TCC e os registros consolidados da campanha. O código conserva a licença GPL-3.0-or-later declarada pelo projeto original. O texto da licença está em `LICENSE`.

O sincronizador em `frontend.py` deriva da topologia de `ofdm_sync_sc_cfb_impl.cc` do GNU Radio, Copyright 2013 Free Software Foundation, Inc., GPL-3.0-or-later. A correlação para CFO inteiro em `phy.py` segue a ordem de acumulação de `ofdm_chanest_vcvc_impl.cc`. O transmissor e o receptor usam blocos e convenções dos exemplos OFDM do GNU Radio. A adaptação acrescenta a porta de energia e permite comparar vários estimadores sobre os mesmos eventos de sincronização.

GNU Radio, gr-osmosdr, libhackrf, scikit-learn, NumPy, SciPy, pandas, Matplotlib e Qiskit são dependências externas, distribuídas por seus respectivos projetos. Os dados e as curvas de antena acompanham o código como material experimental do TCC. Ao reutilizá-los, identifique o trabalho, sua autoria e a versão consultada.
