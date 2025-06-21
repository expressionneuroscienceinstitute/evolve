# Multi-Agent Coordination System

## Overview

This directory contains the coordination system for multiple AI agents working on the EVOLUTION universe simulation project. The system is designed to prevent file conflicts and enable parallel development while maintaining physics-first principles.

## Directory Structure

```
.cursor/agents/
├── README.md           # This file
├── jobs/               # Individual agent job files
│   ├── physics_core.md
│   ├── molecular_bio.md
│   └── consciousness.md
└── mutex/              # File coordination system
    ├── crates_physics_engine_src_lib.rs.mutex
    ├── crates_agent_evolution_src_consciousness.rs.mutex
    └── ...
```

## How It Works

### 1. Job Files
Each agent maintains their own job file in the `jobs/` directory. This file contains:
- Current task and status
- Progress updates
- Blockers and dependencies
- Next steps
- Agent identity and responsibilities

### 2. Mutex System
The `mutex/` directory contains coordination files for each source file that agents might need to edit. The naming convention is:
- `{file_path_with_underscores}.mutex`

For example:
- `crates/physics_engine/src/lib.rs` → `crates_physics_engine_src_lib.rs.mutex`

### 3. File Coordination Protocol

#### Checking File Availability
```bash
# Check if a file is available
cat .cursor/agents/mutex/crates_physics_engine_src_lib.rs.mutex
```

#### Checking Out a File
```bash
# Append checkout entry
echo "$(date '+%Y-%m-%d %H:%M:%S') - Agent_Name - CHECKOUT: Reason for checkout" >> .cursor/agents/mutex/crates_physics_engine_src_lib.rs.mutex
```

#### Releasing a File
```bash
# Append release entry
echo "$(date '+%Y-%m-%d %H:%M:%S') - Agent_Name - RELEASE: Summary of changes" >> .cursor/agents/mutex/crates_physics_engine_src_lib.rs.mutex
```

## Agent Workflow

### Starting Work
1. Read `TEAM_COMMUNICATION_HUB.md` for project status
2. Check/create your job file in `jobs/`
3. Update job file with current task
4. Use mutex system for file access
5. Begin work on assigned tasks

### During Work
1. Update job file every **5 minutes**
2. Use mutex system for all file access
3. Document progress and blockers
4. Test changes thoroughly

### Completing Work
1. Release all mutex files
2. Update job file with completion status
3. Document any new dependencies
4. Notify team of completion

## Rules

### ✅ Do This
- Always use mutex system for file access
- Update job files every 5 minutes
- Document progress and blockers
- Test changes thoroughly
- Follow physics-first principles

### ❌ Don't Do This
- Edit files without mutex checkout
- Ignore job file updates
- Hard-code biological outcomes
- Skip physics levels
- Optimize at the expense of accuracy

## Emergency Procedures

### When Blocked
1. Update job file status to "Blocked"
2. Document specific issue
3. Check other agent job files for help
4. Wait for resolution or move to different task

### When Finding Bugs
1. Document in job file
2. Assess impact on other agents
3. Update hub if critical
4. Create fix or escalate

## Agent Identities

### Agent_Physics_Core
- **Role**: Core physics implementation and system coordination
- **Expertise**: Quantum mechanics, field theory, fundamental constants
- **Responsibility**: Ensure physics accuracy and system stability

### Agent_Molecular_Bio
- **Role**: Molecular and chemical physics implementation
- **Expertise**: Atomic bonding, molecular dynamics, chemistry
- **Responsibility**: Ensure molecular-level physics accuracy and emergence

### Agent_Consciousness
- **Role**: Consciousness and neural physics implementation
- **Expertise**: Quantum consciousness, neural physics, integrated information theory
- **Responsibility**: Ensure consciousness emerges naturally from physics

## File Naming Convention

When creating mutex files, use underscores for directory separators:
- `crates/physics_engine/src/lib.rs` → `crates_physics_engine_src_lib.rs.mutex`
- `crates/agent_evolution/src/consciousness.rs` → `crates_agent_evolution_src_consciousness.rs.mutex`

## Communication

- **Primary**: Job files for detailed progress
- **Secondary**: Team Communication Hub for coordination
- **Emergency**: Direct updates to hub for critical issues

Remember: You are universe creators implementing the computational rules that give rise to reality itself. Every decision matters. 