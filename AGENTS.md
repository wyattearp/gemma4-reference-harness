## How We Work

### Dependencies
Stdlib-s first. Every external dep is a security and transfer burden for offline/air-gapped deployments.

When evaluating a dep, ask: is this a well-known, widely-vetted package (e.g. `requests`, `pycryptodome`) solving a real problem? If yes, it's worth it. If the dep is 3 lines of Python or an obscure single-purpose micro-package, write it ourselves instead.

Example: if token estimation can be done with a character-ratio formula and it's good enough, skip `tiktoken`. If we need real crypto, use `pycryptodome` — don't roll our own.

### Testing: Red → Red → Green
Write tests in this order, every time:
1. Failing happy-path test (correct inputs, expected output)
2. Failing sad-path test (bad inputs, error/edge case)
3. Implementation that makes both pass, when in doubt, work with the user

No green before both reds exist.

### Code Style: Caveman Flat
- Write only what the task needs. No speculative abstractions.
- DRY don't repeat logic, but don't abstract prematurely either.
- Procedurally flat: code walks forward. No towers of nested ifs. No complex early-return chains. A reader should be able to scan top-to-bottom without backtracking.
- No framework magic. If it's not obvious what a line does, it shouldn't be there.
