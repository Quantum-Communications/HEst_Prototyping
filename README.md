# Estimação de canal OFDM em dois HackRF One

Código e dados do TCC de Caio Neves Silva em Engenharia de Computação, CEFET/RJ. O transmissor envia quadros OFDM pelo ar. Após a aquisição, a mesma rajada, a mesma realização de ruído digital e os mesmos eventos de sincronização alimentam os receptores LS, LMMSE, SVR e QSVR. O QSVR usa uma matriz de fidelidades de circuitos simulados com medidas de contagem finita.

| Objetivo | Entrada | Comando principal |
|---|---|---|
| Recalcular resultados da monografia | CSVs, matriz CX e medidas de antena incluídos | `python scripts/reproduzir.py` |
| Reprocessar o sinal recebido na campanha histórica | Capturas I/Q externas, referências TX e configurações originais das sessões | `python scripts/tcc_campaign.py replay … --run` |
| Medir o desempenho de um enlace presente | Dois HackRF One, configuração física local | `python scripts/tcc_campaign.py link-test … --acquire` ou `run-session … --acquire` |

## Instalação no Fedora

O experimento utilizou Fedora 42, GNU Radio 3.10.11.0, interface Osmocom, dois HackRF One r9 com firmwares v2.0.0 e v2.0.1. O notebook Acer Nitro AN515-57 tinha Core i7-11800H (8 núcleos, 16 threads), 16 GiB de RAM e RTX 3050 Laptop. O processamento descrito utiliza CPU. Use o Python do sistema que consegue importar GNU Radio e PMT. Confirme a versão instalada, pois os repositórios da distribuição podem oferecer outras versões. Um notebook já configurado pode aproveitar os pacotes nativos existentes.

```bash
sudo dnf install python3 python3-pip gnuradio gr-osmosdr hackrf
# Dentro da pasta qml-ofdm-tcc extraída
unset PYTHONPATH
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[plots,dev]'
python -c 'import numpy, scipy, sklearn, pandas, matplotlib; print(numpy.__version__, scipy.__version__, sklearn.__version__)'
python -c 'from gnuradio import gr; import pmt, osmosdr; print(gr.version()); assert gr.version() == "3.10.11.0"'
python -c 'import ofdm_benchmark; print(ofdm_benchmark.__file__)'
```

O último caminho deve pertencer a este checkout. Num ambiente exclusivamente numérico, instale o projeto em um venv com Python 3.11 ou superior. Os testes de estimadores, o planejamento e a geração de resultados funcionam sem GNU Radio, Qiskit ou rádio. GNU Radio e PMT são necessários para executar o receptor. Uma nova aquisição requer também gr-osmosdr, libhackrf e os dois dispositivos. O arquivo `pyproject.toml` especifica as dependências Python. NumPy manipula I/Q e matrizes, scikit-learn fornece o LIBSVM e pandas e Matplotlib reconstroem os resultados. O simulador Qiskit só é necessário para reconstruir o Gram.

## Testes sem transmissão

```bash
python -m pytest -q tests
python scripts/tcc_campaign.py check
python scripts/tcc_campaign.py replay
python scripts/reproduzir.py
```

O teste `check` ajusta os quatro modelos sobre um preâmbulo público sintético. O planejamento deve listar **53.680 células em 5.920 grupos de execução**, distribuídas por nove campanhas. A geração de resultados confere **688 agregados** das cinco sessões e 100 rajadas físicas. A entrada principal é `data/campanha_final/master_cells_complete.csv`. Uma célula corresponde a uma combinação experimental aplicada a uma rajada. As condições de uma mesma rajada reutilizam suas amostras.

| Em `scripts/` | Entrada experimental | Saída em `results/publicacao/` |
|---|---|---|
| `consolidar_metricas.py` | Células, agregados publicados, auditoria de consolidação e Gram | Agregados, curvas por sessão, comparações pareadas, custo e tabelas auxiliares |
| `gerar_tabelas_curvas.py` | CSVs, resumo da varredura calculado na etapa anterior e Gram | Tabelas natural, modelos, DFE, impulsivo, EVM e custo. Figuras de BER com AWGN, ortogonalidade, prefixo, LMMSE e similaridades |
| `gerar_robustez_impulsiva.py` | Células das 25 rajadas comuns com ruído Markov-Gaussiano | Gráfico de sensibilidade e CSV dos pontos |
| `gerar_figuras_didaticas.py` | Geometria e exemplos sintéticos, medidas de antena em `data/config/antenas/` | Constelações, tubo SVR, circuito CX, equalização e resposta das antenas |
| `gerar_figura_ofdm_80211a.py` | Parâmetros de OFDM do exemplo didático | Disposição espectral em PDF e PNG |

O comando `reproduzir.py` chama essas cinco etapas em ordem. Os scripts localizam os dados a partir do próprio checkout. Fotografias do enlace, logotipo e capturas de tela dos fluxogramas pertencem aos arquivos editoriais da monografia.

## Uma rajada de 100 pacotes no enlace físico

Use `hackrf_info` para identificar qual equipamento está ligado à antena TX e qual à RX. Preencha os identificadores localmente. A configuração solicitada pelo comando abaixo fica fora do controle de versão. Os diretórios de saída de aquisição precisam ser novos.

```bash
hackrf_info
cp -n configs/tcc.example.json configs/local01.json
python - <<'PY'
import json
from pathlib import Path
p = Path("configs/local01.json")
cfg = json.loads(p.read_text())
with open("/dev/tty") as terminal:
    def read(label):
        print(label, end=" ", flush=True)
        return terminal.readline().strip()
    cfg["radio"]["tx_serial"] = read("Serial de 32 dígitos hexadecimais do TX:")
    cfg["radio"]["rx_serial"] = read("Serial de 32 dígitos hexadecimais do RX:")
    cfg["session_id"] = read("Identificação da nova sessão:")
    cfg["channel_description"] = read("Posição das antenas e condição do enlace:")
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
PY
python scripts/tcc_campaign.py check --config configs/local01.json
python scripts/tcc_campaign.py link-test --config configs/local01.json \
  --output "captures/teste-$(date +%Y%m%d-%H%M%S)" --acquire
```

`link-test` fixa uma rajada, 100 pacotes de medição, condição natural e Pilot-CPE. Primeiro gera a forma de onda e registra a aquisição. Após fechar os rádios, processa a captura pelos quatro métodos. O terminal informa TX, associados, PER e BER condicional. `measurements.json` registra também entregues, erros de bits e EVM. São esperadas quatro linhas com 100 transmitidos. No teste anterior do autor, os quatro métodos associaram 100/100 com PER e BER condicional zero. O resultado de uma nova medição depende do enlace.

## Campanhas, captura e replay

```bash
# Planejar a calibração opcional dos ganhos, sem rádio
python -m nlos_v1.rf_calibration_v2 --config configs/tcc.example.json --output results/calibracao
# Capturar todas as rajadas previstas na configuração local
python scripts/tcc_campaign.py capture --config configs/local01.json --output captures/minha-sessao --acquire
# Em outra sessão independente, adquirir e avaliar os métodos e condições do JSON
python scripts/tcc_campaign.py run-session --config configs/local01.json --output captures/minha-avaliacao --acquire
```

Para executar a calibração física, use a configuração local preenchida, um diretório novo e acrescente `--acquire` ao comando de calibração. A busca avalia candidatos com LS/Pilot-CPE, confirmação e holdout independentes. A calibração é opcional. O JSON de exemplo traz os ganhos finais. O período de aquecimento gera pacotes adicionais que ficam fora do conjunto de medição.

Para reprocessar I/Q históricos, o diretório indicado por `--root` deve conter as cinco pastas `rf_final_01` a `rf_final_05`. Dentro de cada sessão são necessários `evaluation.json`, `acquisition/captures/burst_NNN/capture.c64`, `capture.json` e `tx/truth.json` nas respectivas rajadas. O exemplo local de configuração serve como ponto de partida para novas aquisições. O replay histórico usa a configuração da própria sessão, inclusive sua semente. A matriz CX distribuída é selecionada pelo executor.

```bash
# Primeiro, quatro células da mesma rajada. O replay não abre os dispositivos.
python scripts/tcc_campaign.py replay --root captures --campaign baseline_final \
  --session rf_final_01 --burst 1 --noise natural --equalizer pilot_cpe \
  --output results/replay-quatro --run
# Campanha histórica completa, com espaço em disco para os registros dos receptores
python scripts/tcc_campaign.py replay --root captures --output results/replay-completo --run
```

O executor compara o SHA-256 do I/Q com o registro da célula e confronta os digests de I/Q e da referência TX com o protocolo histórico das 100 rajadas em `fontes_iq.csv`. Verifica também os bytes dos dois arquivos. Compara os hashes da entrada perturbada, das amostras corrigidas e dos disparos, as contagens, a BER, a PER e a EVM. As quatro células sem símbolos para EVM têm resultado indefinido. Cronômetros, versões das bibliotecas e carga do host devem ser interpretados separadamente. A troca de `--workers` altera o regime de custo.

Para executar a mesma malha em **novas capturas**, use uma raiz independente com os cinco nomes de sessão do plano, 20 rajadas por sessão e a identificação correspondente em cada JSON. Acrescente `--new-captures` ao replay. O arquivo `cells.csv` identificará as medidas como novas. Guarde as configurações, as referências TX e os hashes junto com as amostras. Para uma campanha de desenho próprio, `run-session` usa diretamente as condições definidas no JSON e dispensa os nomes históricos.

## Cadeia física e estimadores

O transmissor em `transmit.py` monta o payload de 50 bytes com o identificador OFDM, contador e dados pseudoaleatórios. O GNU Radio gera os números de sequência no cabeçalho. FFT64, CP16, 48 tons de dados e quatro pilotos em −21, −7, +7 e +21 formam o quadro com cabeçalho BPSK e payload QPSK. O segundo preâmbulo conhece o sinal BPSK nas 52 portadoras ativas. O transmissor usa a IFFT sem normalização e a escala digital 0,0167. A figura didática usa QPSK de energia unitária. A tabela de símbolos efetiva do protótipo tem componentes ±√2 e energia quatro, convenção que também entra na EVM. A referência física é 907,98 MHz, 8 MS/s, banda 6 MHz, TX RF/IF 14/47 e RX RF/IF/BB 14/32/20.

Em `acquisition.py`, a janela com TX desligado fornece DC e potência de repouso. O receptor estima uma referência de potência das regiões ativas e registra saturação e diagnósticos USB. Em `benchmark.py` remove-se a DC antes de adicionar AWGN, Bernoulli-Gaussiano ou Markov-Gaussiano à aquisição digital. A SNR adicional é relativa à potência de atividade. O ruído de fundo e as imperfeições do enlace já estão presentes. As sementes combinam configuração, réplica, rajada e família.

Em `frontend.py`, o sincronizador de Schmidl–Cox correlaciona as metades do preâmbulo com a convenção de sinal dos tons ímpares. Usa energia das duas metades, porta de potência e detector de platô. O limiar é 0,65 e a porta de potência é 5 × 10⁻⁵. A correção fracionária precede o demultiplexador. `phy.integer_cfo` testa os deslocamentos −2, 0 e +2 após FFT. Em `native.py`, o estimador entrega o canal e o deslocamento por tags. O cabeçalho decodificado governa a liberação do payload.

| Método | Operação | Local |
|---|---|---|
| LS | Calcula Y/X em cada tom ativo do segundo preâmbulo e zera os tons inativos | `phy.ls_taps` |
| LMMSE | Aplica a covariância cruzada dos 64 tons à estimativa nos 52 ativos. Base própria, suporte de 16 atrasos, PDP proporcional a 0,7ˡ e carregamento 0,03 | `estimators.Estimator`, ramo LMMSE |
| SVR | Dez características seno/cosseno dos atrasos 1, 2, 4, 8 e 16. RBF com γ=0,125. Ajustes independentes de I e Q por ε-SVR precomputado | `estimators/fidelity.py` e `estimators/quick_sklearn.py` |
| QSVR | Mesmo ajuste com o Gram CX de cinco qubits. Atrasos 0,5, 1, 2, 4 e 8. Foram usados 4.000 shots em cada um dos 1.326 pares distintos | `physical_circuit`, `load_custom_gram` e `data/config/cx_4000.npz` |

Nos dois SVRs, o alvo é o segundo preâmbulo recebido Y dividido por seu RMS s. Os sinais conhecidos X=±1 entram em G = diag(X) K diag(X). O solver do scikit-learn fornece os coeficientes duais e o intercepto, com C=9,615 e ε=0,02 na configuração principal. A saída ativa é s X ŷ. Nenhuma amostra recebida entra nos ângulos do circuito. As componentes de custo `training_seconds` dos registros históricos incluem as chamadas a `fit` e `predict` do scikit-learn. `prediction_seconds` mede o trecho final de montagem de H.

O receptor estático conserva o canal estimado. Pilot-CPE acrescenta a correção causal da fase comum pelos pilotos. SimpleDFE usa decisões e pilotos com α=0,1. O LS nativo do GNU Radio permanece como controle de paridade. Em `metrics.py`, a primeira carga completa por número de cabeçalho entra na associação. PER = (transmitidos − entregues sem erro) / transmitidos. BER condicional = erros observados / bits associados. A EVM agregada é a raiz da razão entre a soma das energias de erro e a soma das energias de referência. A verdade TX só entra na pontuação após o receptor.

## Gram quântico

A matriz congelada 52 × 52 é suficiente para todos os replays. Os 1.326 pares do triângulo superior mediram P(00000) após o circuito composto. A geração espelha os pares, projeta a matriz no cone positivo semidefinido quando necessário e renormaliza a diagonal. A semente histórica é 2901, com lotes de 24 pares. O custo do Gram é externo aos tempos de recepção.

```bash
python -m pip install -e '.[quantum]'
python scripts/generate_gram.py --output results/gram-cx-regenerado.npz
python - <<'PY'
import numpy as np
from ofdm_benchmark.estimators.quick_sklearn import load_custom_gram
a, ma = load_custom_gram("data/config/cx_4000.npz", "cx")
b, mb = load_custom_gram("results/gram-cx-regenerado.npz", "cx")
print("Digest histórico:", ma["gram_sha256"], "digest local:", mb["gram_sha256"])
print("Máxima diferença absoluta:", np.max(np.abs(a-b)))
PY
```

As contagens são do BasicSimulator. A versão do SDK e a álgebra numérica podem afetar a regeneração. Para reproduzir o TCC, use o arquivo histórico cujo digest numérico começa por `5bb09544`. A versão local fica separada.

## Referências

A monografia contextualiza a estimação OFDM em van de Beek et al. (1995) e Molisch, a regressão ε-SVR em Smola e Schölkopf e o kernel em Schuld e Petruccione. A notação em frequência usa índices centrados. As constelações e o sentido da FFT seguem o GNU Radio. A licença do código e a atribuição dos componentes encontram-se em `LICENSE` e `NOTICE.md`.
