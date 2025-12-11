# SentimentDNA Linear Project - FINAL Import Package

> **Status**: ✅ VALIDATED AND IMPORT-READY
> **Generated**: 2025-12-10
> **Total Issues**: 117 (6 Epics, 25 Milestones, 66 Tasks, 10 Automations, 10 Risks)

---

## 📁 File Manifest

| File                      | Type       | Issues | Description                               |
| ------------------------- | ---------- | ------ | ----------------------------------------- |
| `epics.csv`               | Epic       | 6      | Top-level project phases                  |
| `milestones.csv`          | Milestone  | 25     | Gate-based checkpoints                    |
| `tasks_phase1.csv`        | Task       | 11     | Product Validation                        |
| `tasks_phase2.csv`        | Task       | 13     | Monetization Architecture                 |
| `tasks_event.csv`         | Task       | 10     | Genesis Key Drop                          |
| `tasks_subscription.csv`  | Task       | 12     | Subscription Engine                       |
| `tasks_protection.csv`    | Task       | 10     | Founder Protection                        |
| `tasks_ops.csv`           | Task       | 10     | Ops Backbone                              |
| `automation_runbooks.csv` | Automation | 10     | Automated job specs                       |
| `risks.csv`               | Risk       | 10     | Risk register                             |
| `labels.json`             | Config     | -      | Label definitions (manual setup)          |
| `workflow_states.json`    | Config     | -      | Workflow state definitions (manual setup) |
| `templates.json`          | Config     | -      | Issue templates (manual setup)            |

---

## 🔧 Pre-Import Setup (Required)

Before running the CLI import, you must configure Linear:

### Step 1: Create Your Team

1. Log into Linear at https://linear.app
2. Go to **Settings** → **Teams** → **Create Team**
3. Team name: `S-DNA` (or your preferred name)
4. Note the team key (e.g., `SDNA`)

### Step 2: Configure Workflow States

1. Go to **Settings** → **Team Settings** → **Workflow**
2. Add/rename states to match this sequence:

| Position | State Name  | Type      |
| -------- | ----------- | --------- |
| 0        | Backlog     | Backlog   |
| 1        | Ready       | Unstarted |
| 2        | In Progress | Started   |
| 3        | Review      | Started   |
| 4        | Validation  | Started   |
| 5        | Gate Passed | Started   |
| 6        | Scheduled   | Started   |
| 7        | Live        | Started   |
| 8        | Monitor     | Started   |
| 9        | Completed   | Completed |
| 10       | Canceled    | Canceled  |

### Step 3: Create Labels

1. Go to **Settings** → **Labels**
2. Create these labels with the specified colors:

| Label        | Color   | Description                |
| ------------ | ------- | -------------------------- |
| validation   | #4CAF50 | Evidence and testing phase |
| evidence     | #2196F3 | Requires proof/artifacts   |
| gate         | #F44336 | Hard stop checkpoint       |
| automation   | #9C27B0 | Automated job              |
| security     | #E91E63 | Security-related           |
| payment      | #FF9800 | Payment/billing            |
| event        | #00BCD4 | Genesis Key Drop           |
| subscription | #8BC34A | Recurring revenue          |
| risk         | #FF5722 | Tracked hazard             |
| compliance   | #607D8B | Legal/regulatory           |

### Step 4: Create Projects

1. Go to **Team** → **Projects** → **Create Project**
2. Create these projects:

- `Phase 1` - Product Validation
- `Phase 2` - Monetization Architecture
- `Genesis Event` - Key Drop Event
- `Subscription` - Revenue Engine
- `Protection` - Founder Safety
- `Ops` - Operations Backbone

### Step 5: Get Your API Key

1. Go to **Settings** → **Account** → **Security**
2. Under **Personal API keys**, click **Create key**
3. Name: `Linear CLI Import`
4. Copy and save the key securely
5. Set as environment variable:

```powershell
# PowerShell (Windows)
$env:LINEAR_API_KEY = "lin_api_xxxxxxxxxxxxxxxxxxxxx"

# OR set permanently
[Environment]::SetEnvironmentVariable("LINEAR_API_KEY", "lin_api_xxxxxxxxxxxxxxxxxxxxx", "User")
```

---

## 🚀 Import Procedure

### Install the CLI

```powershell
npm install -g @linear/import
```

### Verify Installation

```powershell
linear-import --help
```

### Import Sequence (Run in Order)

The import must be done in hierarchical order: **Epics → Milestones → Tasks → Automations → Risks**

#### Import 1: Epics

```powershell
cd C:\Users\Amirp\OneDrive\Documents\PROTONxGOOGLE\SentimentDNA-Linear\FINAL
linear-import
```

**CLI Prompts - Answer as follows:**

1. **API Key**: Enter your Linear API key
2. **Service**: Select `Linear (CSV export)`
3. **File path**: `epics.csv`
4. **Create new team?**: No (select existing team `S-DNA`)
5. **Import to project?**: No
6. **Include comments?**: No (if prompted)
7. **Assign to yourself?**: No → Select `[Unassigned]`

Wait for import to complete.

#### Import 2: Milestones

```powershell
linear-import
```

- File: `milestones.csv`
- Same settings as above

#### Import 3-8: Tasks (Repeat for each file)

```powershell
linear-import
```

Import in this order:

1. `tasks_phase1.csv`
2. `tasks_phase2.csv`
3. `tasks_event.csv`
4. `tasks_subscription.csv`
5. `tasks_protection.csv`
6. `tasks_ops.csv`

#### Import 9: Automation Runbooks

```powershell
linear-import
```

- File: `automation_runbooks.csv`

#### Import 10: Risks

```powershell
linear-import
```

- File: `risks.csv`

---

## 📋 CLI Prompt Quick Reference

For EVERY import, use these answers:

| Prompt              | Answer                              |
| ------------------- | ----------------------------------- |
| API Key             | Your `lin_api_xxx` key              |
| Service             | `Linear (CSV export)`               |
| Create new team?    | **No**                              |
| Select team         | `S-DNA`                             |
| Import to project?  | **No** (issues have Project column) |
| Include comments?   | **No**                              |
| Assign to yourself? | **No** → `[Unassigned]`             |

---

## 🔗 Post-Import: Link Parent Issues

The CSV import creates issues but **does not automatically link Parent relationships**. You must link them manually or via Linear's API.

### Manual Linking Steps:

1. Open Linear → Team → Issues
2. Filter by epic title (e.g., "Phase 1 — Product Validation")
3. For each milestone, open it and set **Parent** to the epic
4. For each task, open it and set **Parent** to the milestone

### Alternative: Use Linear API

```javascript
// Example: Link issue to parent via API
const { LinearClient } = require("@linear/sdk");

const client = new LinearClient({ apiKey: process.env.LINEAR_API_KEY });

// Get issue by title and update parent
async function linkParent(childTitle, parentTitle) {
  const issues = await client.issues({ filter: { title: { eq: childTitle } } });
  const parents = await client.issues({
    filter: { title: { eq: parentTitle } },
  });

  if (issues.nodes[0] && parents.nodes[0]) {
    await issues.nodes[0].update({ parentId: parents.nodes[0].id });
    console.log(`Linked: ${childTitle} → ${parentTitle}`);
  }
}
```

---

## ✅ Verification Checklist

After import, verify:

### Issue Counts

| Category     | Expected | ✓   |
| ------------ | -------- | --- |
| Total Issues | 117      | ☐   |
| Epics        | 6        | ☐   |
| Milestones   | 25       | ☐   |
| Tasks        | 66       | ☐   |
| Automations  | 10       | ☐   |
| Risks        | 10       | ☐   |

### Structure Verification

- [ ] All 6 epics visible in team backlog
- [ ] All 25 milestones created
- [ ] Labels assigned correctly (filter by each label)
- [ ] Priority levels set (1=Urgent, 2=High, 3=Medium, 4=Low)
- [ ] Estimates populated where specified
- [ ] States all set to "Backlog"

### Label Distribution

- [ ] `validation` label: ~15 issues
- [ ] `automation` label: ~25 issues
- [ ] `security` label: ~12 issues
- [ ] `event` label: ~18 issues
- [ ] `gate` label: ~10 issues
- [ ] `risk` label: ~10 issues

### Projects

- [ ] Phase 1 project has ~11 issues
- [ ] Phase 2 project has ~13 issues
- [ ] Genesis Event project has ~10 issues
- [ ] Subscription project has ~12 issues
- [ ] Protection project has ~10 issues
- [ ] Ops project has ~10 issues

---

## 🛠 Troubleshooting

### "Label not found" warning

Labels referenced in CSV but not in Linear are auto-created. If you want predefined colors, create labels manually first (Step 3).

### "Team not found" error

Ensure the team exists and your API key has access. Check Settings → Teams.

### "Rate limited" error

The CLI handles rate limiting automatically with retries. Wait and retry if persistent.

### CSV parsing errors

Ensure:

- No multi-line descriptions (all flattened in FINAL versions)
- UTF-8 encoding (no BOM)
- LF line endings (not CRLF)

To convert line endings (if needed):

```powershell
(Get-Content epics.csv -Raw) -replace "`r`n", "`n" | Set-Content epics.csv -NoNewline
```

### Priority not mapping

Linear priorities: 0=None, 1=Urgent, 2=High, 3=Medium, 4=Low. CSVs use 1-4 format.

---

## 📊 Summary Statistics

### By Epic

| Epic                                               | Milestones | Tasks | Automations | Risks |
| -------------------------------------------------- | ---------- | ----- | ----------- | ----- |
| Phase 1 — Product Validation                       | 5          | 11    | 3           | 2     |
| Phase 2 — Monetization Architecture                | 5          | 13    | 0           | 1     |
| Genesis Key Drop — Event Execution                 | 5          | 10    | 2           | 2     |
| Recurring Revenue — Subscription Engine            | 5          | 12    | 2           | 2     |
| Founder Protection — Identity, Privacy, Safety     | 5          | 10    | 1           | 3     |
| Ops Backbone — Automation, Observability, Runbooks | 0          | 10    | 2           | 0     |

### By Priority

| Priority   | Count |
| ---------- | ----- |
| 1 (Urgent) | 15    |
| 2 (High)   | 52    |
| 3 (Medium) | 50    |

### Total Estimate Points

- **Phase 1**: 68 points
- **Phase 2**: 73 points
- **Genesis Event**: 61 points
- **Subscription**: 62 points
- **Protection**: 46 points
- **Ops**: 57 points
- **Total**: ~367 story points

---

## 🔐 Security Notes

- **API Key**: Never commit to version control
- **Store securely**: Use environment variables or secret manager
- **Rotate regularly**: Create new keys periodically
- **Revoke on exposure**: If key is leaked, revoke immediately in Linear settings

---

## 📞 Support

- **Linear Documentation**: https://linear.app/docs
- **Linear API**: https://developers.linear.app
- **Import CLI Source**: https://github.com/linear/linear/tree/master/packages/import

---

**Created by**: Linear Workspace Integration Agent  
**Version**: 1.0.0  
**Last Updated**: 2025-12-10
