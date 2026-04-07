# Online Maskable Charging Env Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `envs` package that exposes an online, maskable-action charging environment backed by `SplitChargingOrchestrator`.

**Architecture:** Keep future arrivals hidden from the agent by advancing through an internal arrival stream one vehicle at a time. The env only surfaces the current decision vehicle, decodes discrete `(r, lambda)` actions into planner decisions, and automatically submits second-leg arrivals to the simulator when their noisy real arrival time occurs.

**Tech Stack:** Python, Gymnasium, NumPy, simulator/orchestrator module, unittest

---

## Chunk 1: Lock Behavior With Tests

### Task 1: Confirm the online env contract

**Files:**
- Modify: `tests/test_charging_env.py`

- [ ] **Step 1: Add tests for default minimum-duration masking**
- [ ] **Step 2: Add a regression test for automatic second-leg submission**
- [ ] **Step 3: Run `python -m unittest tests.test_charging_env -v` and confirm failure before implementation**

## Chunk 2: Create the `envs` Package

### Task 2: Implement action encoding and decoding

**Files:**
- Create: `envs/__init__.py`
- Create: `envs/maskable_actions.py`

- [ ] **Step 1: Define the discrete action layout for `(r, lambda_bin)`**
- [ ] **Step 2: Implement valid-action enumeration with downstream-station and minimum-duration constraints**
- [ ] **Step 3: Export helpers used by tests and the env**

### Task 3: Implement the online charging env

**Files:**
- Create: `envs/charging_env.py`

- [ ] **Step 1: Add the online arrival stream and second-leg event handling**
- [ ] **Step 2: Build observations via `SplitChargingOrchestrator.build_observation()`**
- [ ] **Step 3: Decode discrete actions into `ChargingDecision` values**
- [ ] **Step 4: Compute reward from per-step delta of total `queue_time`**
- [ ] **Step 5: Keep terminal observations and masks stable for RL consumers**

## Chunk 3: Verify

### Task 4: Prove the env works

**Files:**
- Test: `tests/test_charging_env.py`
- Test: `tests/test_orchestrator_observation.py`
- Test: `tests/test_simulator_history_integration.py`

- [ ] **Step 1: Run the env tests**
- [ ] **Step 2: Run focused simulator/orchestrator regression tests**
- [ ] **Step 3: Re-run until all targeted checks are green**
