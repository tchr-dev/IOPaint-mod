# Epic 3.4 — Models Cache Table Plan

## Goal
Add a lightweight `models_cache` table to persist model metadata (id, provider, capabilities, updated timestamps) so the UI can load model lists faster and support offline mode.

## One-Step Plan
Implement `models_cache` schema + storage accessors, then wire the API to read/write cached models (with TTL) and expose a refresh endpoint for the frontend.
