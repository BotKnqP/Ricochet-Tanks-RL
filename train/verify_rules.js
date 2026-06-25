"use strict";
// Verify the rule changes: (a) every round terminates in a death (no timeout, no hang) thanks to the
// fully-closing poison ring; (b) tanks never overlap (solid hitboxes). Runs the exported agent vs laika.
const path = require("path");
const fs = require("fs");
const Core = require(path.join(process.cwd(), "game_core.js"));
const { RicochetCore, policyForward } = Core;
global.window = global.window || {};
require(path.join(process.cwd(), "model_weights_v3champ.js"));   // committed policy (was a stale runs/auto_live JSON)
const POLICY = global.window.RICOCHET_POLICIES.v3champ;
const SEEDS = [1307, 300000, 300001, 300002, 900000, 100000, 207919, 42];
const CAP = 4000;

const res = {};
let maxSteps = 0, sumSteps = 0, overlaps = 0, minGapWorst = 1e9;
for (const seed of SEEDS) {
  const c = new RicochetCore({ seed, arenaMode: "survival", scenario: "moba1v1duel", spawnPowerups: true, shellDecay: true });
  c.reset(seed);
  let n = 0, done = false, result = "CAP", minGap = 1e9;
  while (n < CAP) {
    const a = policyForward(c.observe(0), POLICY);
    const o = c.step(a, "laika", { dt: 1 / 30, repeat: 2 });
    n++;
    const s = c.getPublicState();
    const [t0, t1] = s.tanks;
    if (t0.alive && t1.alive) {
      const d = Math.hypot(t0.x - t1.x, t0.y - t1.y) - (t0.radius + t1.radius);
      if (d < minGap) minGap = d;
      if (d < -2) overlaps++;   // overlap deeper than 2px = collision failed
    }
    if (o.done) { result = o.info.result; done = true; break; }
  }
  res[result] = (res[result] || 0) + 1;
  maxSteps = Math.max(maxSteps, n); sumSteps += n; minGapWorst = Math.min(minGapWorst, minGap);
  console.log(`seed ${String(seed).padStart(6)}: result=${String(result).padEnd(7)} steps=${String(n).padStart(4)} minGap=${minGap.toFixed(1)}px`);
}
console.log("-".repeat(60));
console.log("results:", JSON.stringify(res));
console.log(`avgSteps=${Math.round(sumSteps / SEEDS.length)} maxSteps=${maxSteps} (CAP=${CAP})  worstGap=${minGapWorst.toFixed(1)}px  deepOverlaps=${overlaps}`);
console.log(res.timeout || res.CAP ? "FAIL: timeout/hang present" : "PASS: all rounds resolved by death, no timeout/hang");
