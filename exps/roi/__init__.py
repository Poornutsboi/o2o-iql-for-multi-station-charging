"""ROI benchmark from Dastpak et al. (2024).

Reoptimization with Occupancy Indicator (ROI) for the Electric Vehicle
Shortest Path Problem with charging-station occupancy information.
The route of every vehicle is fixed in this project, so the deterministic
EVSPP collapses to enumerating split-charging actions and scoring each one
against an offline M/M/C/inf wait-time lookup.
"""
