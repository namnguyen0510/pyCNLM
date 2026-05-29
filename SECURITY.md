# Security Policy

We take the security of `pycnlm` seriously and appreciate any responsibly
disclosed vulnerability reports.

## Supported versions

`pycnlm` is pre-1.0; only the **latest released minor version** receives
security patches. Once the project hits `1.0.0` we will document a longer
support window here.

| Version  | Supported          |
|----------|--------------------|
| `0.1.x`  | :white_check_mark: |
| `< 0.1`  | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Instead, use one of the following private channels:

1. **GitHub private vulnerability reporting** — preferred.
   Open the repository's *Security* tab and click *Report a vulnerability*.
   GitHub will deliver the report privately to the maintainers.

2. **Email** — fall-back if you cannot use GitHub's private reporting.
   Send a description of the issue to the address listed in the
   repository's `CITATION.cff` or on the maintainers' GitHub profiles.

When reporting, please include:

- A clear description of the issue and its impact.
- Steps to reproduce, with a minimal example if possible.
- The `pycnlm` version (`pycnlm version`) and Python version.
- Any suggested mitigation.

## What to expect

- **Acknowledgement** within 5 business days.
- **Status update** within 14 days, including whether we accept the report,
  the severity assessment, and an expected timeline.
- **Coordinated disclosure**: we will agree on a public-disclosure date
  with the reporter and aim to release a fix beforehand.

We do not currently run a paid bug-bounty programme, but we will credit
reporters in the release notes unless they request anonymity.

## Scope

In scope:

- The `pycnlm` package itself (anything under `pycnlm/` except `pycnlm/core/LangevinCNLM/third_party/`).
- The CLI (`pycnlm/cli.py`).
- The CI / release workflows (`.github/workflows/`).

Out of scope:

- The vendored `third_party/` reference implementations — please file
  security reports for those upstream with the respective projects.
- Issues that require a malicious local user already holding code-execution
  privileges on the target machine.
