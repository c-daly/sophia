# LOGOS State Architecture - Design Decisions

**Date:** 2026-01-01
**Status:** Draft - Core concepts resolved

## Core Insight

**There is one graph: the HCG.**

- CWM (Causal World Model) is **content within the HCG**, not a separate system
- CWM-A/G/E are node categories within the HCG
- CWMState is a **transport/event envelope** for HCG change notifications (name is historical)
- Provenance lives on HCG nodes, not the envelope

## Two Systems

| System | What It Stores |
|--------|----------------|
| Talos | World state (physical reality) |
| HCG | Everything Sophia knows (CWM nodes, plans, processes, types) |

## Memory Tiers (No Redis)

| Tier | Storage | Lifetime |
|------|---------|----------|
| Ephemeral | In-memory | Session (not yet in HCG) |
| Mid-term | HCG + `expires_at` | Days-weeks |
| Long-term | HCG (no expiry) | Indefinite |

Mid-term vs long-term = presence/absence of `expires_at` field.

## CWM Node Categories (within HCG)

| Category | Content |
|----------|---------|
| CWM-G | Grounded beliefs, predictions |
| CWM-A | Abstract knowledge, goals, plans |
| CWM-E | Affective states, confidence |

Cross-category edges allowed.

## Key Files

- `logos/docs/plans/2026-01-01-state-architecture-design.md` - Full design doc