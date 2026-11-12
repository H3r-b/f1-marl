# Research Methodology

## Initial research question

Does population-based self-play improve the robustness of competitive behaviour in a multi-agent formula-style racing environment compared with training against fixed opponents?

## Planned baselines

- Scripted waypoint-following / rule-based opponent (when implemented)
- Independent PPO agents
- MAPPO-style centralized training with decentralized execution
- Population-based self-play variant

## Planned metrics

- Mean and median lap time
- Race completion rate
- Collision rate per race and per lap
- Track-limit violations
- Overtaking attempts and success rate
- Win rate against a held-out opponent pool
- Performance variance across random seeds

## Experimental controls

- Keep evaluation tracks and opponent pools fixed across comparisons.
- Report multiple random seeds.
- Separate training opponents from held-out evaluation opponents.
- Record configuration, code revision, and environment version.
- Include ablations for reward terms and observation features.

## Caution

These are planned experiments, not results. Claims of improvement require measured evidence and uncertainty estimates.
