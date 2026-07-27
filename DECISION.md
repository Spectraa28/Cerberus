# Cerberus — Design Decisions & Known Gaps

## Known gaps

- **context_seeded spawn mode is untested against Anthropic/OpenAI.** Gemini
  tolerates seeded history that references a tool outside the sub-agent's
  declared scope (e.g. a shell-only sub-agent seeded with a `search_files`
  tool_result). Anthropic and OpenAI pair tool_use/tool_result by id within
  a stricter contract and may reject this. Needs a history filter that
  strips out-of-scope tool turns before seeding, OR needs to be tested
  against both providers to confirm whether it's actually a problem.

- **shell_exec's command allowlist is a guardrail, not a sandbox.** It
  catches simple binary-name checks (`grep`, `curl`, etc. via `;`, `&&`,
  `||`, `|` splitting) but does not catch command substitution
  (`$(...)`, backticks) or indirect invocation (e.g. `python -c
  "os.system(...)"`). Sufficient for a cooperative model reaching for the
  wrong tool; not sufficient against adversarial input. Real sandboxing
  (containers/seccomp/no-network namespace) needed if Cerberus ever runs
  untrusted input through a sub-agent.

## Decisions

- Prefix-scoping alone (`ToolRegistry.scoped()`) is not a real permission
  boundary when shell access is in scope, since shell can reimplement
  other tools. Default posture: shell access is granted deliberately per
  sub-agent, not bundled in casually.