name: Bug report
description: Create a report to help us improve CARE-ASR
title: "[BUG] "
labels: ["bug"]
assignees: []

body:
  - type: markdown
    attributes:
      value: Thanks for taking the time to fill out this bug report!
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us what you expected to happen.
      placeholder: Describe the bug...
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: How to reproduce
      description: Step-by-step instructions or minimal python snippet to reproduce the issue.
      placeholder: |
        1. Run `python -m care_asr.demo --audio clip.wav`
        2. See error...
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Copy and paste any error tracebacks or console logs.
      render: shell
