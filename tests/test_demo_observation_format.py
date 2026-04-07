import unittest

from simulator.demo_observation_format import run_observation_format_demo


class ObservationFormatDemoTests(unittest.TestCase):
    def test_demo_returns_observation_and_decision_io_sections(self) -> None:
        payload = run_observation_format_demo()

        self.assertIn("observation_input", payload)
        self.assertIn("observation_output", payload)
        self.assertIn("decision_input", payload)
        self.assertIn("decision_output", payload)

        self.assertIn("current_ev", payload["observation_input"])
        self.assertIn("now", payload["observation_input"])
        self.assertIn("sim_state", payload["observation_output"])
        self.assertIn("future_demand", payload["observation_output"])
        self.assertIn("travel_time_matrix", payload["observation_output"])
        station_payload = payload["observation_output"]["sim_state"]["stations"][0]
        self.assertIn("queue_waiting_time", station_payload)
        self.assertIn("queue_demand", station_payload)
        self.assertNotIn("queue", station_payload)

        self.assertIn("decision", payload["decision_input"])
        self.assertIn("first_request", payload["decision_output"])
        self.assertIn("first_assignment", payload["decision_output"])
