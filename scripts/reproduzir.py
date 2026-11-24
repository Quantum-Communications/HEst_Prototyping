"""Reconstrói as tabelas e figuras a partir das entradas públicas do TCC."""
from pathlib import Path
import subprocess
import sys


def main():
    scripts = Path(__file__).resolve().parent
    for name in ("consolidar_metricas", "gerar_tabelas_curvas", "gerar_robustez_impulsiva",
                 "gerar_figuras_didaticas", "gerar_figura_ofdm_80211a"):
        subprocess.run([sys.executable, str(scripts / (name + ".py"))], check=True)


if __name__ == "__main__":
    main()
