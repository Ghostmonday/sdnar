# SentimentDNA Linear Project Import

## Quick Start Options

### Option 1: Use Linear's In-App Import

1. Go to Linear Settings → Import & Export
2. Import the CSV files in this folder

### Option 2: Use the Linear Import CLI

```bash
npm i -g @linear/import
linear-import
# Select "Linear CSV" and point to the CSV files
```

### Option 3: Connect Linear MCP to Claude (Recommended)

Follow instructions at: https://mcp.linear.app

## File Structure

```
SentimentDNA-Linear/
├── epics.csv                 # Top-level Epics
├── milestones.csv            # All milestones by phase
├── tasks_phase1.csv          # Phase 1: Product Validation
├── tasks_phase2.csv          # Phase 2: Monetization Architecture
├── tasks_event.csv           # Genesis Key Drop Event
├── tasks_subscription.csv    # Recurring Revenue Engine
├── tasks_protection.csv      # Founder Protection
├── tasks_ops.csv             # Ops Backbone
├── automation_runbooks.csv   # Automation jobs & runbooks
├── risks.csv                 # Risk register
├── labels.json               # Label definitions
├── workflow_states.json      # Custom workflow states
└── README.md                 # This file
```

## Linear Configuration

### Recommended Labels

- `validation` - Evidence and testing phase
- `evidence` - Requires proof/artifacts
- `gate` - Hard stop checkpoint
- `automation` - Automated job
- `security` - Security-related
- `payment` - Payment/billing
- `event` - Genesis Key Drop event
- `subscription` - Recurring revenue
- `risk` - Tracked hazard
- `compliance` - Legal/regulatory

### Workflow States

1. Backlog
2. Ready
3. In Progress
4. Review
5. Validation
6. Gate Passed
7. Scheduled
8. Live
9. Monitor
10. Completed

### Zero-Chance Gates (Critical)

These states require:

- **Validation**: Evidence artifacts attached + Tests passed
- **Gate Passed**: All linked risks mitigated
- **Live**: Runbook present + Rollback defined
- **Completed**: SLOs met + Metrics captured

---

Created: 2025-12-10
Project: SentimentDNA Launch
