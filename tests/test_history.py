import unittest

from simulator.history import ChargingHistoryLog
from simulator.models import ChargingHistoryRecord


class ChargingHistoryLogTests(unittest.TestCase):
    def test_records_are_appended_in_submission_order(self) -> None:
        history_log = ChargingHistoryLog()

        history_log.append(
            ChargingHistoryRecord(
                vehicle_id=1,
                station_id=0,
                charger_id=0,
                arrival_time=0.0,
                start_time=0.0,
                end_time=4.0,
                wait_time=0.0,
            )
        )
        history_log.append(
            ChargingHistoryRecord(
                vehicle_id=1,
                station_id=2,
                charger_id=0,
                arrival_time=10.0,
                start_time=10.0,
                end_time=16.0,
                wait_time=0.0,
            )
        )

        self.assertEqual(
            [(item.vehicle_id, item.station_id, item.arrival_time) for item in history_log.records()],
            [(1, 0, 0.0), (1, 2, 10.0)],
        )
