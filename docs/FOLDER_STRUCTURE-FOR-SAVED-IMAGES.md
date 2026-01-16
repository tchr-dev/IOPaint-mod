Here’s a clean, production-ready recommendation that works well for IOpaint,
async OpenAI jobs, and later auditing/debugging.

⸻

Short answer (recommended default)

👉 Use session-centric structure with date bucketing + thumbnails

~/.iopaint/data/openai/ └── 2026/ └── 01/ └── 16/ └── session_<session_id>/ └──
job_<job_id>/ ├── input.png ├── mask.png ├── output.png ├── output_thumb.webp
├── meta.json └── logs.txt

This gives you: •	human-navigable storage •	fast cleanup by date •	strong
traceability per session & job •	UI-friendly thumbnails

⸻

Why not pure date-only or pure session-only

❌ Pure session-only

openai/<session_id>/<job_id>/

Problems •	Hard to garbage-collect old data •	Millions of entries in one
directory •	Poor ops visibility (“what happened today?”)

⸻

❌ Pure date-only

openai/2026/01/16/<job_id>/

Problems •	You lose UI session grouping •	Harder to correlate user actions
•	Sessions spanning multiple days fragment badly

⸻

Recommended hybrid structure (explained)

1️⃣ Date buckets (filesystem hygiene) •	Prevents huge directories •	Enables TTL
cleanup (cron / background GC) •	Ops-friendly

YYYY/MM/DD/

⸻

2️⃣ Session folder (UI & undo/redo semantics)

session_<uuid>/

Use the same session_id you already track in IOpaint (canvas lifecycle).

Benefits: •	Supports undo / redo •	Supports “restore session” •	Supports
multi-job batching

⸻

3️⃣ Job folder (immutability)

job_<job_id>/

Each OpenAI job = immutable snapshot.

This is critical for: •	async retries •	auditability •	debugging safety /
moderation issues

⸻

File layout per job (strongly recommended)

job_<job_id>/ ├── input.png # Original image ├── mask.png # Binary or alpha mask
├── output.png # Final result (full res) ├── output_thumb.webp # UI preview
(256–512px) ├── meta.json └── logs.txt # optional

⸻

meta.json (do this — you’ll thank yourself)

{ "session_id": "b5b0...", "job_id": "job_abc123", "backend": "openai", "model":
"gpt-image-1", "state": "succeeded", "created_at": "2026-01-16T09:41:22Z",
"prompt": "remove person from background", "seed": null, "size": "1024x1024",
"cost_estimate": { "tokens": 812, "usd": 0.032 }, "retry_count": 0, "error":
null }

This enables: •	cost dashboards •	per-job replay •	offline debugging •	safety
audits

⸻

Thumbnails — YES, store them

Why thumbnails matter •	UI loads instantly •	No GPU decode needed •	Avoids
loading 4–8MB PNGs in grids

Recommendation •	Format: WEBP •	Size: 256px or 384px •	Name: output_thumb.webp

Optional: •	input_thumb.webp •	mask_thumb.webp (debug mode only)

⸻

Optional but powerful additions

🔹 Deduplication by content hash

input_<sha256>.png

→ reuse identical inputs across jobs

⸻

🔹 Garbage collection strategy •	Keep thumbnails forever •	Delete full images
after N days •	Keep meta.json always

⸻

🔹 Atomic writes

Always:

write → fsync → rename

Especially important for async jobs.

⸻

Final recommendation summary

✅ Use hybrid structure •	Date → Session → Job

✅ Store thumbnails •	WEBP, UI-first

✅ Always include meta.json •	Job is useless without metadata

✅ Immutable job folders •	Never overwrite outputs

Just tell me which one you want next.
