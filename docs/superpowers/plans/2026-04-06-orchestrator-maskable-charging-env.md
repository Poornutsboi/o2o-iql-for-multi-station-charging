# Orchestrator Maskable Charging Env Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `envs/charging_env.py` so the environment only accepts maskable discrete actions, returns orchestrator-built observations, and computes reward from the per-step delta of system `queue_time`.

**Architecture:** Keep the existing event-driven episode loop and episode-bank support, but replace the observation/reward/action plumbing with orchestrator-backed behavior. The environment itself becomes the maskable-discrete API surface, while action decoding and valid-action masking continue to rely on `envs.maskable_actions`.

**Tech Stack:** Python, Gymnasium, NumPy, simulator/orchestrator module, unittest/pytest-compatible test execution

---

## Chunk 1: Add Regression Tests

### Task 1: Observation and reward behavior

**Files:**
- Create: `tests/test_charging_env.py`
- Modify: `envs/charging_env.py`

- [ ] **Step 1: Write failing tests for the new environment contract**
- [ ] **Step 2: Run the targeted test file and confirm failures are caused by the old env behavior**
- [ ] **Step 3: Cover orchestrator observation keys, maskable discrete action space, action masks, and delta-queue-time reward**

## Chunk 2: Rewrite Environment Core

### Task 2: Replace the old Dict-action env behavior

**Files:**
- Modify: `envs/charging_env.py`

- [ ] **Step 1: Rebuild `MultiStationChargingEnv` around `SplitChargingOrchestrator`**
- [ ] **Step 2: Make the env action space directly `spaces.Discrete(8 * n_bins)`**
- [ ] **Step 3: Decode actions with `envs.maskable_actions`, preserve no-split and minimum-duration validation, and keep invalid-action penalties**
- [ ] **Step 4: Return `orchestrator.build_observation(...)` observations and terminal observations with the same top-level schema**
- [ ] **Step 5: Compute reward as `-reward_scale * delta(sum(queue_time))`**

## Chunk 3: Clean Up Factory and Variants

### Task 3: Keep the public entry points coherent

**Files:**
- Modify: `envs/charging_env.py`

- [ ] **Step 1: Remove or retire wrapper-only code paths that conflict with the new maskable-only API**
- [ ] **Step 2: Keep `EpisodeBankChargingEnv`, `get_state/set_state`, and `make_env(...)` working with the new env contract**
- [ ] **Step 3: Ensure `make_env(...)` defaults and validation reflect the maskable-only surface**

## Chunk 4: Verify

### Task 4: Prove the rewrite works

**Files:**
- Test: `tests/test_charging_env.py`
- Test: any directly impacted existing simulator/orchestrator tests

- [ ] **Step 1: Run the new env tests**
- [ ] **Step 2: Run a focused orchestrator/simulator regression subset**
- [ ] **Step 3: Review failures, fix if needed, and re-run until green**
