# **Agent Team Guidelines: Physics-First Universe Simulation**

## **Mission Statement**

You are building a **universe simulation from absolute scratch** - where the only hard-coded elements are the fundamental laws of physics themselves. Everything else must emerge naturally from these laws through computational rules operating at the most fundamental level.

**Core Philosophy**: Think like a universe creator, not a programmer. You are implementing the computational rules that give rise to reality itself.

---

## **Fundamental Rules (Never Break These)**

### **1. Physics-First Principle**
- **ONLY** hard-code fundamental constants (speed of light, Planck constant, etc.)
- **NEVER** hard-code biological outcomes, chemical reactions, or emergent phenomena
- **EVERYTHING** must emerge from particle interactions and fundamental forces
- If you find yourself writing `if cancer_detected` or `if cell_divides`, you're doing it wrong

### **2. Emergence-Only Design**
- Start with the simplest possible computational rules
- Let complexity arise through iteration and interaction
- If something doesn't emerge naturally, the rules are wrong, not the simulation
- **No shortcuts, no cheat codes, no predetermined outcomes**

### **3. Bottom-Up Architecture**
- Build from quarks → hadrons → atoms → molecules → cells → organisms
- Each level must emerge from the level below
- Never skip levels or assume higher-level behavior exists

---

## **Implementation Guidelines**

### **When You Get Stuck (3-Step Process)**

#### **Step 1: Physics Check**
Ask yourself: "What are the fundamental forces and particles involved?"
- If you can't answer this, you're thinking too high-level
- Go back to the most basic physics principles
- Reference: [Wolfram Physics Project](https://www.wolframphysics.org/technical-introduction/introduction/index.html)

#### **Step 2: Emergence Check**
Ask yourself: "Can this behavior emerge from simpler rules?"
- If yes, implement the simpler rules and let it emerge
- If no, you're probably hard-coding something that should emerge
- Look for patterns in the existing codebase that might already create this behavior

#### **Step 3: Validation Check**
Ask yourself: "Does this violate any fundamental physics laws?"
- If yes, fix the physics implementation
- If no, test with minimal parameters and observe emergence

### **Communication Protocol**

#### **Daily Standup Format**
```
1. What I'm working on: [Specific physics rule or emergent behavior]
2. Current challenge: [What's blocking me]
3. Physics level: [Quark/Hadron/Atom/Molecule/Cell/Organism]
4. Emergence status: [Is it emerging naturally or am I forcing it?]
5. Next step: [Specific action to take]
```

#### **When Blocked (Escalation Process)**
1. **Try for 30 minutes** - Apply the 3-step process above
2. **Ask the team** - Post in team channel with specific physics question
3. **Review existing code** - Check if similar patterns already exist
4. **Document the challenge** - Add to TODO.md with physics-first context
5. **Move to different task** - Don't get stuck on one problem

---

## **Code Quality Standards**

### **Physics Validation**
- Every function must have a clear physics justification
- Constants must be scientifically accurate
- Units must be consistent throughout
- Performance optimizations must not sacrifice physics accuracy

### **Emergence Validation**
- Test with minimal initial conditions
- Verify behavior emerges without hard-coded triggers
- Document the emergence pathway
- Ensure reproducibility across different random seeds

### **Documentation Requirements**
- Every module needs physics justification
- Document the computational rules being implemented
- Explain how higher-level phenomena emerge
- Include references to relevant physics papers

---

## **Common Pitfalls to Avoid**

### **❌ Don't Do This**
- Hard-code biological outcomes ("if cell_type == cancer")
- Assume chemical reactions exist ("if molecule_a + molecule_b")
- Create artificial rules ("if energy > threshold")
- Skip physics levels ("quark → cell" without intermediate steps)

### **✅ Do This Instead**
- Let outcomes emerge from particle interactions
- Build chemical reactions from atomic physics
- Let energy thresholds emerge from fundamental forces
- Implement each physics level completely before moving up

---

## **Success Metrics**

### **Physics Accuracy**
- All fundamental constants match real-world values
- Conservation laws are preserved
- Relativistic effects are properly implemented
- Quantum mechanics operates correctly

### **Emergence Quality**
- Complex behaviors arise from simple rules
- No hard-coded biological or chemical outcomes
- Reproducible emergence across different conditions
- Scalable from small to large systems

### **Performance**
- Simulation runs at acceptable speed
- Memory usage scales reasonably
- Multicore utilization is efficient
- No physics shortcuts for performance

---

## **Team Collaboration Rules**

### **Code Review Focus**
1. **Physics accuracy** - Is this physically correct?
2. **Emergence quality** - Does this emerge naturally?
3. **Performance impact** - Does this maintain efficiency?
4. **Documentation** - Is the physics rationale clear?

### **Conflict Resolution**
- **Physics disagreements**: Reference scientific literature
- **Implementation disagreements**: Prototype both approaches
- **Performance vs. accuracy**: Accuracy wins, optimize later
- **Scope disagreements**: Start simpler, let complexity emerge

### **Knowledge Sharing**
- Share interesting emergence patterns you discover
- Document unexpected physics behaviors
- Create examples of successful emergence
- Maintain a library of physics references

---

## **Emergency Procedures**

### **When Everything Seems Wrong**
1. **Stop coding** - Take a 10-minute break
2. **Review fundamentals** - What are we actually trying to simulate?
3. **Simplify** - Remove complexity, start with basic physics
4. **Test minimal case** - Does the simplest possible scenario work?
5. **Build up slowly** - Add complexity one step at a time

### **When Performance is Unacceptable**
1. **Profile first** - Identify the actual bottleneck
2. **Physics check** - Are we doing unnecessary calculations?
3. **Algorithm review** - Can we use a better physics algorithm?
4. **Optimize carefully** - Don't sacrifice physics accuracy
5. **Consider parallelization** - Use multicore for physics calculations

---

## **Remember: You Are Universe Creators**

You're not just writing code - you're implementing the computational rules that give rise to reality itself. Every decision you make affects the fundamental nature of the universe you're creating.

**Think big, implement small, let emergence do the heavy lifting.**

---

## **Contact and Support**

- **Physics questions**: Reference RESEARCH_PAPERS.md and scientific literature
- **Implementation challenges**: Use the 3-step process above
- **Team coordination**: Follow the communication protocol
- **Documentation**: Keep everything updated in real-time

**The goal is not to simulate life - it's to create the conditions where life can emerge naturally from physics.**

Good luck, universe creators! 🌌 