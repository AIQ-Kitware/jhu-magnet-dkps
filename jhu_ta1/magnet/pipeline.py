"""
The DKPS instance-prediction DAG: one fit per seed.

    dkps_auc[seed]      one job per replicate, and one cell per job

No gather. Each seed is an independent replicate, and the card states its
relation against each -- which is what the seed sweep already meant.
"""
import kwdagger

from jhu_ta1.magnet.dkps_auc import DkpsAucConfig


class DkpsAuc(kwdagger.ProcessNode):
    name = 'dkps_auc'
    executable = 'python -m jhu_ta1.magnet.dkps_auc'
    params = DkpsAucConfig

    def load_result(self, node_dpath):
        pass


def dkps_pipeline():
    return kwdagger.Pipeline([DkpsAuc()])
