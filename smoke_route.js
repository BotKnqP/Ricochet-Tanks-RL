"use strict";
// Lesson 1b (nav_route_to_center) headless smoke tests. Run: node smoke_route.js
const assert = require("assert");
const { RicochetCore, constants, OBS_SIZE } = require("./game_core.js");

const SEED = 1307;
const CW = constants.CELL_W, CH = constants.CELL_H;
const cc = (x, y) => ({ x: x * CW + CW / 2, y: y * CH + CH / 2 });   // grid cell -> centre px
function makeRoute(extra) {
  const c = new RicochetCore(Object.assign(
    { seed: SEED, arenaMode: "survival", scenario: "nav_route_to_center", opponent: "none", maxSteps: 1800 },
    extra || {}));
  c.reset(SEED);
  return c;
}
function park(c, x, y) { c.state.tanks[0].x = x; c.state.tanks[0].y = y; }
const R = {};

// (1,2) obs size + range
{
  const obs = makeRoute().buildObservation(0);
  assert.strictEqual(OBS_SIZE, 105, "OBS_SIZE 105");
  assert.strictEqual(obs.length, 105, "route reset obs length 105");
  assert.ok(obs.every((v) => Number.isFinite(v) && v >= -1.000001 && v <= 1.000001), "route obs finite & in range");
}

// (3) blue spawns outside centerClear
{
  const ps = makeRoute().getPublicState();
  const b = ps.tanks[0], cl = ps.centerClear;
  const inside = b.x > cl.x && b.x < cl.x + cl.w && b.y > cl.y && b.y < cl.y + cl.h;
  assert.ok(!inside, "blue spawn outside centerClear");
  R.blueSpawn = { x: Math.round(b.x), y: Math.round(b.y) };
}

// (4) red dead + opponent obs zeroed
{
  const c = makeRoute();
  assert.ok(c.buildObservation(0).slice(15, 25).every((v) => v === 0), "opponent 10 obs zero");
  assert.strictEqual(c.getPublicState().tanks[1].alive, false, "red dead in route");
}

// (5) poison disabled but the safe-zone/centre-direction block is still informative
{
  const pb = makeRoute().buildObservation(0).slice(OBS_SIZE - 11);
  assert.strictEqual(pb[0], 0, "poisonActive flag 0 (poison off)");
  assert.strictEqual(pb[10], 0, "timeToPoisonStart 0 (no poison)");
  assert.ok(pb[4] > 0, "distance-to-centre present");
  assert.ok(pb[2] !== 0 || pb[3] !== 0, "centre direction provided");
  R.centerDir = { dx: Math.round(pb[2] * 100) / 100, dy: Math.round(pb[3] * 100) / 100, dist: Math.round(pb[4] * 100) / 100 };
}

// (6) spawnPowerups=false -> no powerups appear
{
  const c = makeRoute();
  for (let i = 0; i < 120; i++) c.step(0, "none");
  assert.strictEqual(c.getPublicState().spawnPowerups, false, "spawnPowerups false");
  assert.strictEqual(c.getPublicState().powerups.length, 0, "no powerups on field");
}

// (7) new-cell exploration: counted once per cell
{
  const c = makeRoute();
  for (let i = 0; i < 3; i++) { park(c, cc(2, 2).x, cc(2, 2).y); c.step(0, "none"); }   // same cell 3x
  const after1 = c.getPublicState().nav.newCells;
  park(c, cc(3, 2).x, cc(3, 2).y); const out = c.step(0, "none");                       // a new cell
  assert.strictEqual(after1, 1, "same cell counted once");
  assert.strictEqual(out.info.newCells, 2, "entering a new cell increments newCells");
  R.newCells = { afterSameCell: after1, afterNewCell: out.info.newCells };
}

// (8) wall hit increments wallHits with cooldown
{
  const c = makeRoute();
  c.state.tanks[0].angle = Math.PI;
  park(c, constants.WALL + constants.TANK_RADIUS + 2, 600);
  for (let i = 0; i < 60; i++) c.step({ throttle: 1, turn: 0, fire: false }, "none");
  const wh = c.getPublicState().nav.wallHits;
  assert.ok(wh >= 1 && wh < 60, "wall hits registered & cooldown-throttled");
  R.wallHits = wh;
}

// (9) stuck detection: idle past stuckWindow -> stuckEvents + penalty
{
  const c = makeRoute();   // blue idles at spawn (action 0 -> no movement)
  for (let i = 0; i < 60; i++) c.step(0, "none");
  const se = c.getPublicState().nav.stuckEvents;
  assert.ok(se >= 1, "idle/stuck registered stuckEvents");
  R.stuckEvents = se;
}

// (10,11) pathDist finite + decreases toward centre (BFS), bestPathDist tracks the min
{
  const c = makeRoute();
  park(c, cc(1, 1).x, cc(1, 1).y); const far = c.step(0, "none").info;
  park(c, cc(4, 2).x, cc(4, 2).y); const near = c.step(0, "none").info;
  assert.ok(far.pathDist >= 0, "pathDist finite at start");
  assert.ok(far.pathDist > near.pathDist, "BFS pathDist smaller for a cell closer to centre");
  assert.ok(near.bestPathDist <= near.pathDist, "bestPathDist tracks the minimum reached");
  R.pathDist = { far: far.pathDist, near: near.pathDist, best: near.bestPathDist };
}

// (12) first entering centre -> enterCenterReward
{
  const c = makeRoute();
  const cl = c.getPublicState().centerClear;
  park(c, cl.x + cl.w / 2, cl.y + cl.h / 2);
  const out = c.step(0, "none");
  assert.strictEqual(out.info.enteredCenter, true, "enteredCenter true");
  assert.ok(out.reward > 0.9, "enterCenterReward (~1.0) applied");
  R.enterCenter = { reward: Math.round(out.reward * 1000) / 1000 };
}

// (13) staying inside centre 2s -> route_success
{
  const c = makeRoute();
  const cl = c.getPublicState().centerClear;
  let info = null;
  for (let i = 0; i < 200; i++) {
    park(c, cl.x + cl.w / 2, cl.y + cl.h / 2);
    const out = c.step(0, "none");
    if (out.done) { info = out.info; break; }
  }
  assert.ok(info && info.result === "route_success", "2s in centre -> route_success");
  assert.ok(info.centerStayTime >= 2.0, "centerStayTime >= 2");
  R.success = { result: info.result, centerStayTime: info.centerStayTime, elapsed: info.elapsed };
}

// (14) timeout outside centre -> route_timeout
{
  const c = makeRoute({ maxSteps: 50 });
  let info = null;
  for (let i = 0; i < 80; i++) {
    park(c, cc(1, 1).x, cc(1, 1).y);   // far from centre, never succeeds; no poison so never dies
    const out = c.step(0, "none");
    if (out.done) { info = out.info; break; }
  }
  assert.ok(info && info.result === "route_timeout", "maxSteps -> route_timeout");
  assert.ok(info.routeTimeout === true, "routeTimeout flag");
  R.timeout = { result: info.result };
}

// (v2-1,2) no-progress: stalled-outside accrues noProgressEvents, cooldown-throttled (not every frame)
{
  const c = makeRoute();
  for (let i = 0; i < 120; i++) { park(c, cc(1, 1).x, cc(1, 1).y); c.step(0, "none"); }  // parked far, no progress
  const np = c.getPublicState().nav.noProgressEvents;
  assert.ok(np >= 1, "stalled bestPathDist -> noProgressEvents fired");
  assert.ok(np < 10, "noProgressEvents throttled by noProgressWindow (not every step)");
  R.noProgress = { over120steps: np };
}

// (v2-3) inside the centre: no stuck / no-progress penalty while waiting to win
{
  const c = makeRoute({ maxSteps: 2000 });
  const cl = c.getPublicState().centerClear;
  for (let i = 0; i < 70; i++) { c.step(0, "none"); }   // idle at spawn (outside) -> stuck + no-progress accrue
  const before = c.getPublicState().nav;
  const stuckB = before.stuckEvents, npB = before.noProgressEvents;
  assert.ok(stuckB > 0, "penalties accrued while outside (sanity)");
  let info = null;
  for (let i = 0; i < 200; i++) { park(c, cl.x + cl.w / 2, cl.y + cl.h / 2); const out = c.step(0, "none"); if (out.done) { info = out.info; break; } }
  assert.strictEqual(info.result, "route_success", "reaches success inside centre");
  assert.strictEqual(info.stuckEvents, stuckB, "no extra stuck penalty inside centre");
  assert.strictEqual(info.noProgressEvents, npB, "no extra no-progress penalty inside centre");
  R.centreGate = { stuckFrozen: info.stuckEvents === stuckB, noProgressFrozen: info.noProgressEvents === npB };
}

// (v2-5) a path-progress step refreshes the no-progress clock (delays the penalty)
{
  const A = makeRoute();
  for (let i = 0; i < 70; i++) { park(A, cc(1, 1).x, cc(1, 1).y); A.step(0, "none"); }
  const B = makeRoute();
  for (let i = 0; i < 70; i++) {
    park(B, i === 30 ? cc(4, 2).x : cc(1, 1).x, i === 30 ? cc(4, 2).y : cc(1, 1).y);  // one closer-cell step = progress
    B.step(0, "none");
  }
  const npA = A.getPublicState().nav.noProgressEvents, npB = B.getPublicState().nav.noProgressEvents;
  assert.ok(npA >= 1, "stalled run fired no-progress by step 70");
  assert.ok(npB < npA, "a progress step refreshed lastProgressAt and delayed no-progress");
  R.progressRefresh = { stalled: npA, withProgressStep: npB };
}

// regression: other scenarios untouched
{
  const b = new RicochetCore({ seed: SEED, arenaMode: "survival" });
  assert.strictEqual(b.reset(SEED).length, 105, "battle survival obs 105");
  assert.strictEqual(b.getPublicState().scenario, "battle", "default scenario battle");
  const pn = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "nav_powerup_poison" });
  pn.reset(SEED);
  assert.strictEqual(pn.getPublicState().scenario, "nav_powerup_poison", "poison-nav still works");
}

console.log("ROUTE SMOKE OK");
console.log(JSON.stringify(R, null, 2));
