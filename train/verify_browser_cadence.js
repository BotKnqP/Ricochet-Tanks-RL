"use strict";
// Reproduce the browser's stepping in node to isolate why the in-browser win rate looked low.
// OLD = advance(1/30) deciding BOTH tanks every physics update (the old watch path).
// NEW = step(action, opp, {dt:1/30, repeat:2}) — one decision per RL step (training/eval cadence).
// Runs the EXPORTED agent policy (policyForward) vs a script over diverse seeds.
const path = require("path");
const fs = require("fs");
const Core = require(path.join(process.cwd(), "game_core.js"));
const { RicochetCore, policyForward } = Core;

const polWrap = fs.readFileSync(path.join(process.cwd(), "runs/auto_live/live_policy.json"), "utf8");
const POLICY = JSON.parse(polWrap);
const OPP = process.argv[2] || "laika";
const N = parseInt(process.argv[3] || "40", 10);
const CAP = 2500;

function mkCore(seed) {
  return new RicochetCore({ seed, arenaMode: "survival", scenario: "moba1v1duel",
    spawnPowerups: true, shellDecay: true });
}
function runOld(seed) {  // decide every update
  const c = mkCore(seed); c.reset(seed);
  let n = 0, res = null;
  while (n < CAP) {
    const a = policyForward(c.observe(0), POLICY);
    const o = c.advance(1 / 30, a, OPP);
    n++; if (o.done) { res = o.result; break; }
  }
  return res;
}
function runNew(seed) {  // decide per RL step, action held repeat=2
  const c = mkCore(seed); c.reset(seed);
  let n = 0, res = null;
  while (n < CAP) {
    const a = policyForward(c.observe(0), POLICY);
    const o = c.step(a, OPP, { dt: 1 / 30, repeat: 2 });
    n++; if (o.done) { res = o.info.result; break; }
  }
  return res;
}
function tally(fn) {
  let w = 0, l = 0, d = 0;
  for (let i = 0; i < N; i++) {
    const seed = (100000 + i * 7919) >>> 0;
    const r = fn(seed);
    if (r === "win") w++; else if (r === "loss") l++; else d++;
  }
  return { win: (w / N).toFixed(2), loss: (l / N).toFixed(2), other: (d / N).toFixed(2) };
}
console.log(`agent vs ${OPP}, ${N} diverse seeds:`);
console.log("  OLD (advance, decide every update):", JSON.stringify(tally(runOld)));
console.log("  NEW (step, repeat=2 — training cadence):", JSON.stringify(tally(runNew)));
