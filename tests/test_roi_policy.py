import unittest
from types import SimpleNamespace

import numpy as np

from envs.maskable_actions import decode_maskable_action
from exps.roi.policy import RoiPolicy
from exps.roi.wait_lookup import SplitLookup, StationLookup


def _constant_wait_lookup(station_id: int, wait: float) -> StationLookup:
    table = np.full((6, 2, 1), float(wait), dtype=np.float64)
    sample_count = np.ones((6, 2, 1), dtype=np.int64)
    return StationLookup(
        station_id=station_id,
        capacity=1,
        q_max=5,
        delta_grid=(0.0,),
        table=table,
        sample_count=sample_count,
    )


class RoiPolicyTests(unittest.TestCase):
    def test_policy_splits_when_downstream_station_has_lower_wait_pressure(self) -> None:
        lookup = SplitLookup(
            split_name="unit",
            q_max=5,
            delta_grid=(0.0,),
            stations=(
                _constant_wait_lookup(0, 10.0),
                _constant_wait_lookup(1, 0.0),
            ),
            config={},
        )
        env = SimpleNamespace(
            pending_vehicle=SimpleNamespace(route=[0, 1], duration=10.0),
            n_bins=5,
            num_stations=2,
            min_first_charge=2.0,
            min_second_charge=2.0,
            clock=0.0,
            _sim=SimpleNamespace(
                get_state=lambda query_time: {
                    "stations": {
                        0: {"charger_status": [1.0], "queue_waiting_time": [1, 2, 3]},
                        1: {"charger_status": [0.0], "queue_waiting_time": []},
                    },
                }
            ),
            _orchestrator=SimpleNamespace(
                _build_travel_time_matrix=lambda: [[0.0, 1.0], [1.0, 0.0]],
            ),
        )

        action = RoiPolicy(lookup).select_action(env)
        second_choice, frac_bin = decode_maskable_action(
            action_int=action,
            n_bins=env.n_bins,
            num_stations=env.num_stations,
        )

        self.assertEqual(second_choice, 1)
        self.assertLess(frac_bin, env.n_bins - 1)


if __name__ == "__main__":
    unittest.main()
