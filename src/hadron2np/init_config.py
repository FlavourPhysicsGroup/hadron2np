"""config 用于配置 fdm 程序本身, 并不直接参与物理过程"""
from pathlib import Path
import yaml

config = {}

# particle_name : PDGID
with (Path(__file__).parent / 'data/pdg_particle.yaml').open() as file:
    info_pdgid = yaml.safe_load(file)
config['pdgid'] = info_pdgid

# fdm configuration
with (Path(__file__).parent / 'data/config.yaml').open() as file:
    info_config = yaml.safe_load(file)
config.update(info_config)

# 所有已实现的粒子衰变过程
with (Path(__file__).parent / 'data/hadron_matrix_metadata.yaml').open() as file:
    info_hadron_matrix = yaml.safe_load(file)
config['hadron matrix element'] = info_hadron_matrix
