name: Task
description: Track an execution task from the CARE-ASR execution plan
title: "[TASK] "
labels: ["task"]
assignees: []

body:
  - type: input
    id: task-id
    attributes:
      label: Execution Task ID
      placeholder: e.g., T2, T5, S3
    validations:
      required: true
  - type: input
    id: owner
    attributes:
      label: Task Owner
      placeholder: e.g., Ankit, Mahi, Aarth, Divya
    validations:
      required: true
  - type: textarea
    id: task-details
    attributes:
      label: Task Scope & Expected Deliverables
      placeholder: Describe the deliverables for this execution step...
    validations:
      required: true
