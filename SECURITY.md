# Security Policy

## Supported Versions

Only the latest version of CARE-ASR is actively supported for security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously, especially given CARE-ASR's application domain in medical speech processing.

If you discover a security vulnerability or potential data leak risk (e.g., unintended retention of PHI/PII audio or transcript data), please report it to our team privately:

- **Primary Contact**: Ankit Choubey (ankit@example.com)
- **Response Time**: You will receive an initial response acknowledging your report within 48 hours.

Please **do not** open a public GitHub issue for security vulnerabilities.

### What to Include in Your Report
1. Description of the vulnerability and potential impact.
2. Step-by-step instructions or proof-of-concept script to reproduce the issue.
3. Any suggested remediations or patches.

## Handling PHI and Privacy
CARE-ASR operates as an on-premise/local pipeline. By default:
- No audio or transcript data is transmitted to third-party cloud APIs unless explicitly configured.
- Local LLM inference (e.g., via Ollama/vLLM) is enforced for HIPAA-compliant deployments.
