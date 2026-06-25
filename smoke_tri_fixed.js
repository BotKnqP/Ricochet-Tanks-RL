"use strict";
// Smoke for spawnMode "tri_fixed": 3 fixed left points x 3 mirror right points = 9 discrete openings.
// Checks: 3 distinct blue lanes + 3 red lanes, all 9 combos reachable, min-sep ok, obs finite, and tanks
// are NOT spawned inside a wall (proxy: a tank told to drive forward actually moves). Run: node smoke_tri_fixed.js
const { RicochetCore } = require("./game_core.js");
const blueY = {}, redY = {}, combos = {};
let badSep = 0, badObs = 0, stuck = 0, n = 0;
for (let s = 0; s < 360; s++) {
  const seed = (1000 + s * 7) >>> 0;
  const c = new RicochetCore({ seed, arenaMode: "survival", scenario: "moba1v1duel", spawnPowerups: true,
    shellDecay: true, ruleset: "survival_v1", spawnMode: "tri_fixed", maxSteps: 3000 });
  c.reset(seed);
  const v0 = c.getPublicState();
  const b = v0.tanks[0], r = v0.tanks[1];
  const by = Math.round(b.y), ry = Math.round(r.y);
  blueY[by] = (blueY[by] || 0) + 1; redY[ry] = (redY[ry] || 0) + 1;
  combos[`${by}|${ry}`] = (combos[`${by}|${ry}`] || 0) + 1;
  if (Math.hypot(b.x - r.x, b.y - r.y) < 120) badSep++;
  const o0 = c.observe(0), o1 = c.observe(1);
  if ([...o0, ...o1].some((x) => !Number.isFinite(x))) badObs++;
  // "not in a wall" proxy: drive BOTH tanks straight forward for 12 steps; a tank stuck in a wall barely moves.
  const sb = { x: b.x, y: b.y };
  for (let k = 0; k < 12; k++) c.step({ throttle: 1, turn: 0, fire: false }, "stationary", { dt: 1 / 30, repeat: 2 });
  const v1 = c.getPublicState();
  if (Math.hypot(v1.tanks[0].x - sb.x, v1.tanks[0].y - sb.y) < 4) stuck++;
  n++;
}
const okLanes = Object.keys(blueY).length === 3 && Object.keys(redY).length === 3;
const okCombos = Object.keys(combos).length === 9;
console.log("blue lanes:", Object.keys(blueY).map(Number).sort((a, b) => a - b), "counts", blueY);
console.log("red  lanes:", Object.keys(redY).map(Number).sort((a, b) => a - b), "counts", redY);
console.log("distinct 9 combos:", Object.keys(combos).length, "| badSep:", badSep, "| badObs:", badObs, "| stuck-in-wall:", stuck, "| n:", n);
const pass = okLanes && okCombos && badSep === 0 && badObs === 0 && stuck === 0;
console.log(pass ? "TRI_FIXED SMOKE OK" : "TRI_FIXED SMOKE FAIL");
process.exit(pass ? 0 : 1);
