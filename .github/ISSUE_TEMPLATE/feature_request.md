name: Feature request
description: Suggest an idea or enhancement for CARE-ASR
title: "[FEAT] "
labels: ["enhancement"]
assignees: []

body:
  - type: textarea
    id: feature-description
    attributes:
      label: Is your feature request related to a problem? Please describe.
      placeholder: A clear and concise description of what the problem is.
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      placeholder: A clear and concise description of what you want to happen.
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      placeholder: Any alternative solutions or features considered.
