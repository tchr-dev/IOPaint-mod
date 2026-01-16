Below is a practical, backend-oriented status model you can safely implement for
OpenAI-compatible job processing (including ProxyAPI / async image or response
jobs).

⸻

1. Common job lifecycle states

Most OpenAI-compatible backends expose a job / task / response object with a
state (or status) field. Conceptually, states fall into three groups:

🔄 Non-terminal (in-progress)

These should NOT end processing.

State	Meaning	Action queued	Job accepted, not yet started	Poll pending	Alias of
queued (some backends)	Poll running	Model is executing	Poll processing	Same as
running	Poll in_progress	Same as running	Poll

✅ Rule: keep polling with backoff.

⸻

2. Terminal states (no retry)

These mean the job will never change.

✅ Successful

State	Meaning succeeded	Completed successfully completed	Alias used by some APIs

➡️ Consume result, persist output.

⸻

❌ Hard failure (do NOT retry automatically)

State	Meaning	Notes failed	Model/runtime failure	Inspect error payload
cancelled	Explicit user or system cancellation	Treat as final
blocked_budget	Budget / quota exhausted	Requires user action blocked	Policy /
moderation block	Permanent unless input changes rejected	Validation or safety
rejection	Input must change expired	Job TTL exceeded	Usually permanent

➡️ Mark terminal, surface error.

⸻

3. Retryable terminal-looking states ⚠️

Some states look terminal but can be retried safely under controlled logic.

State	Retry?	When timeout	✅ Yes	Network or worker timeout rate_limited	✅
Yes	Respect Retry-After overloaded	✅ Yes	Backend saturation
service_unavailable	✅ Yes	5xx equivalent internal_error	⚠️ Maybe	Retry once or
twice aborted	⚠️ Depends	If system-initiated

✅ Best practice: max retry count + exponential backoff.

⸻

4. Recommended classification logic (production-safe)

TERMINAL_SUCCESS = [ "succeeded", "completed", ]

TERMINAL_FAILURE_NO_RETRY = [ "failed", "cancelled", "blocked",
"blocked_budget", "rejected", "expired", ]

RETRYABLE = [ "timeout", "rate_limited", "overloaded", "service_unavailable",
"internal_error", ]

IN_PROGRESS = [ "queued", "pending", "running", "processing", "in_progress", ]

⸻

5. Important nuance for OpenAI / Responses API

In Responses / Images APIs, a job can be logically complete even if: •	Output
array is empty •	Tool call failed but response object exists •	Partial outputs
exist with an error field

➡️ Always inspect: •	response.status •	response.error •	response.output[]
•	tool-specific results (e.g. image_generation_call)

This behavior is documented in the OpenAI Images & Vision guide ￼.

⸻

6. What you should log for observability

For each job, persist: •	job_id •	state •	error.code •	error.type
•	error.message •	retry_count •	model •	billing_reason (budget / quota)

This is critical for blocked_budget vs failed diagnostics.

⸻

7. TL;DR (safe defaults)

Treat as terminal immediately

succeeded failed cancelled blocked blocked_budget rejected expired

Retry with backoff

timeout rate_limited overloaded service_unavailable internal_error (limited)

Everything else → poll
