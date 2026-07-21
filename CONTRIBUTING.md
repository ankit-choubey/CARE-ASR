# Contributing Guidelines

Thank you for your interest in contributing to **CARE-ASR**! This document provides guidelines and workflows to maintain code quality, reproducibility, and robust collaboration across our team.

---

## 1. Branch Strategy

We follow a modified **GitFlow** branching methodology tailored for fast-paced ML/ASR pipelines:

- `main`: Production-ready, fully verified, and tested releases.
- `develop`: Primary integration branch. All feature branches merge here after passing CI/CD checks.
- `feature/<task_id>-<short_description>`: Individual task work branches mapped directly to execution plan tasks (e.g., `feature/T2-semantic-faiss`, `feature/T3-entropy-gate`).
- `fix/<issue_id>-<short_description>`: Bug fixes for reported issues (e.g., `fix/S3-logit-shape`).
- `docs/<short_description>`: Documentation additions and architectural updates.

---

## 2. Commit Naming

Commits must follow the **Conventional Commits** specification:

`<type>(<scope>): <short description>`

### Allowed Types:
- `feat`: A new feature or pipeline capability (e.g., `feat(retrieval): add RRF fusion engine`)
- `fix`: A bug fix or patch (e.g., `fix(confidence): handle zero-length token logit vectors`)
- `docs`: Documentation updates (e.g., `docs(contract): update FusionCandidate data types`)
- `style`: Formatting, missing semi-colons, white-spaces (e.g., `style(core): format with black and ruff`)
- `refactor`: Code refactoring without behavioral changes (e.g., `refactor(transcriber): optimize Whisper batching`)
- `test`: Adding or modifying test suites (e.g., `test(safety): add Levenshtein edit distance boundary test`)
- `chore`: Maintenance, dependencies, tool configurations (e.g., `chore(deps): update PyTorch dependency pin`)

---

## 3. Pull Request (PR) Rules

1. **All PRs require an issue or task reference**: Link the corresponding execution task (e.g., `Resolves #T5` or `Fixes #42`).
2. **CI/CD Checks Must Pass**: All automated tests (`pytest`), linter (`ruff`), and code formatter (`black`) checks must pass cleanly.
3. **No Direct Pushes**: Direct pushing to `main` or `develop` is strictly disabled.
4. **Clean Git History**: Rebase feature branches onto `develop` prior to submitting a PR to maintain a linear history.
5. **PR Template**: Complete all sections in `.github/PULL_REQUEST_TEMPLATE.md`.

---

## 4. Merge Policy

- **Required Approvals**: At least **1 mandatory review** from a code owner (see `.github/CODEOWNERS`).
- **Squash and Merge**: Preferred merge strategy for feature branches to keep `develop` and `main` histories clean.
- **Interface Modifications**: Any PR changing shared dataclasses in `docs/interface_contract.md` requires explicit sign-off from **Ankit Choubey** (Integration Lead).

---

## 5. Coding Conventions & Python Style Guide

We adhere strictly to modern Python standards:

- **Python Version**: Target Python **3.10+**.
- **Type Hints**: All function signatures must be fully type-annotated (`def compute_entropy(logits: torch.Tensor, q: float = 0.5) -> EntropyScore:`).
- **Docstrings**: Use **Google-style docstrings** for all public modules, classes, and methods.
- **Formatting Tool**: Use **Black** with default line length of **88 characters**.
- **Linting Tool**: Use **Ruff** for style enforcement and import sorting (`isort` rules enabled).
- **Immutability**: Prefer frozen `@dataclass(frozen=True)` or Pydantic models for shared data objects.

---

## 6. Review Process

1. **Submission**: Open a PR targeting `develop`.
2. **Automated Validation**: GitHub Actions will execute `.github/workflows/tests.yml`.
3. **Code Review**: Reviewers evaluate:
   - Interface contract compliance.
   - Test coverage for edge cases (e.g., OOV tokens, empty audio clips).
   - Numerical stability (e.g., log-sum-exp, division-by-zero guards).
4. **Approval & Merge**: Once approved and checks pass, the PR is merged and the working branch is deleted.
