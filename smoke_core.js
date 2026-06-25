"use strict";
// Headless check: game_core.js must run in plain Node with NO DOM/canvas shim.
const assert = require("assert");
const { RicochetCore, ACTION_TABLE, OBS_SIZE, RAY_OFFSETS, constants, makeSurvivalWorld } = require("./game_core.js");

assert.strictEqual(ACTION_TABLE.length, 18, "expected 18 discrete actions");
assert.strictEqual(RAY_OFFSETS.length, 32, "expected 32 wall radar rays");
assert.strictEqual(constants.WALL_RAY_COUNT, 32, "expected exported ray count");
assert.strictEqual(OBS_SIZE, 105, "expected 105-d observation (90 base + 11-wide survival poison block)");

// A tiny deterministic action stream so the test is reproducible.
function makeStream(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s % ACTION_TABLE.length; };
}

function assertCoreBasics(mode) {
  const core = new RicochetCore({ seed: 1307, arenaMode: mode });
  const obs = core.reset(1307);
  const state = core.getPublicState();
  assert.strictEqual(state.arenaMode, mode, `${mode}: public state arenaMode`);
  assert.strictEqual(state.constants.arenaMode, mode, `${mode}: constants arenaMode`);
  assert.strictEqual(state.constants.wallRayCount, 32, `${mode}: ray count`);
  assert.strictEqual(obs.length, OBS_SIZE, `${mode}: reset obs size`);
  assert.strictEqual(core.buildObservation(0).length, OBS_SIZE, `${mode}: buildObservation obs size`);
  if (mode === "maze") {
    assert.ok(state.maze, "maze mode should include maze cells");
    assert.ok(state.walls.length > 4, "maze mode should include internal walls");
    assert.strictEqual(state.spawnPowerups, true, "maze mode default powerups");
  } else {
    assert.strictEqual(state.maze, null, "open mode should not include maze cells");
    assert.strictEqual(state.walls.length, 4, "open mode should only include boundary walls");
    assert.strictEqual(state.spawnPowerups, false, "open mode default powerups");
    assert.strictEqual(state.tanks[0].x, 240, "open blue spawn x");
    assert.strictEqual(state.tanks[0].y, 320, "open blue spawn y");
    assert.strictEqual(state.tanks[1].x, 720, "open red spawn x");
    assert.strictEqual(state.tanks[1].y, 320, "open red spawn y");
  }
}

function runEpisode(seed, arenaMode) {
  const core = new RicochetCore({ seed, arenaMode, maxSteps: 2000 });
  let obs = core.reset(seed);
  assert.strictEqual(obs.length, OBS_SIZE, "reset obs size");
  const next = makeStream(seed ^ 0x5151);
  const obsTrace = [];
  let steps = 0, done = false, totalReward = 0, result = "running", lastInfo = null;
  while (!done && steps < 2000) {
    const a0 = next();
    const out = core.step(a0, arenaMode === "open" ? "stationary" : "laika");
    assert.strictEqual(out.obs0.length, OBS_SIZE, "step obs size");
    assert.strictEqual(typeof out.reward, "number", "reward is a number");
    assert.ok(Number.isFinite(out.reward), "reward is finite");
    assert.strictEqual(typeof out.done, "boolean", "done is boolean");
    obsTrace.push(out.obs0[0]);
    totalReward += out.reward;
    done = out.done;
    result = out.info.result;
    lastInfo = out.info;
    steps += 1;
  }
  return { steps, totalReward, result, obsTrace, lastInfo };
}

assertCoreBasics("maze");
assertCoreBasics("open");

const a = runEpisode(1307, "maze");
const b = runEpisode(1307, "maze");
assert.deepStrictEqual(a.obsTrace, b.obsTrace, "DETERMINISM FAILED: same seed produced different maze trajectories");
assert.strictEqual(a.steps, b.steps, "determinism: maze step count differs");

const c = runEpisode(1307, "open");
const d = runEpisode(1307, "open");
assert.deepStrictEqual(c.obsTrace, d.obsTrace, "DETERMINISM FAILED: same seed produced different open trajectories");
assert.strictEqual(c.steps, d.steps, "determinism: open step count differs");

// --- Level 1B: open-arena spawn jitter (deterministic per seed, varies across seeds) ---
function jitterSpawn(seed) {
  const core = new RicochetCore({ seed, arenaMode: "open", spawnJitter: true });
  core.reset(seed);
  const s = core.getPublicState();
  return { blueY: s.tanks[0].y, redY: s.tanks[1].y, blueAngle: s.tanks[0].angle, spawnJitter: s.spawnJitter };
}
function runOpenJitter(seed) {
  const core = new RicochetCore({ seed, arenaMode: "open", spawnJitter: true, maxSteps: 800 });
  core.reset(seed);
  const next = makeStream(seed ^ 0x77);
  const trace = [];
  let steps = 0, done = false;
  while (!done && steps < 800) {
    const out = core.step(next(), "stationary");
    assert.strictEqual(out.obs0.length, OBS_SIZE, "open-jitter obs size");
    assert.ok(Number.isFinite(out.reward), "open-jitter reward finite");
    trace.push(out.obs0[0], out.obs0[1]);
    done = out.done; steps += 1;
  }
  return trace;
}
const jA = jitterSpawn(1307);
const jB = jitterSpawn(1307);
const jC = jitterSpawn(20000);
assert.strictEqual(jA.spawnJitter, true, "jitter flag should be reported in public state");
assert.ok(Math.abs(jA.blueY - 320) > 1e-9 || Math.abs(jA.redY - 320) > 1e-9, "jitter should move spawns off-center");
assert.ok(jA.blueY === jB.blueY && jA.redY === jB.redY && jA.blueAngle === jB.blueAngle, "jitter must be deterministic per seed");
assert.ok(jA.redY !== jC.redY, "different seeds should give different jitter spawns");
const oj1 = runOpenJitter(4242);
const oj2 = runOpenJitter(4242);
assert.deepStrictEqual(oj1, oj2, "DETERMINISM FAILED: open-jitter episode not reproducible");

// open fixed-spawn must remain unchanged (Level 1A protection)
const fixed = new RicochetCore({ seed: 1307, arenaMode: "open" });
fixed.reset(1307);
const fs = fixed.getPublicState();
assert.strictEqual(fs.tanks[0].y, 320, "open fixed blue y unchanged (Level 1A)");
assert.strictEqual(fs.tanks[1].y, 320, "open fixed red y unchanged (Level 1A)");
assert.strictEqual(fs.spawnJitter, false, "open default spawnJitter must stay false");

// --- Level 1C: low-frequency turret opponent ---
function runTurret(seed, arenaMode, steps, blueMode) {
  const core = new RicochetCore({ seed, arenaMode, spawnJitter: arenaMode === "open", maxSteps: steps + 50 });
  core.reset(seed);
  const next = makeStream(seed ^ 0x33);
  const fireSteps = [];
  let redFiresNoLos = 0, noLosSteps = 0, done = false, i = 0;
  while (!done && i < steps) {
    const pre = core.getPublicState();
    const vis = core.canSee(pre.tanks[1], pre.tanks[0]);
    if (!vis) noLosSteps += 1;
    const prevReload = pre.tanks[1].reload;
    const blueAction = blueMode === "random" ? next() : 0;
    const out = core.step(blueAction, "turret");
    assert.strictEqual(out.obs0.length, OBS_SIZE, "turret step obs size");
    assert.ok(Number.isFinite(out.reward), "turret reward finite");
    const post = core.getPublicState();
    if (post.tanks[1].reload > prevReload + 1e-6) {  // red's reload jumped => it fired
      fireSteps.push(i);
      if (!vis) redFiresNoLos += 1;
    }
    done = out.done;
    i += 1;
  }
  let minGap = Infinity;
  for (let k = 1; k < fireSteps.length; k++) minGap = Math.min(minGap, fireSteps[k] - fireSteps[k - 1]);
  return { redFires: fireSteps.length, fireSteps, minGap, redFiresNoLos, noLosSteps, steps: i };
}

// open: no inner walls -> turret always has LoS; must fire, but rate-limited by cooldown
const tOpen1 = runTurret(4242, "open", 600, "still");
const tOpen2 = runTurret(4242, "open", 600, "still");
assert.ok(tOpen1.redFires >= 1, "turret should fire at least once with LoS + alignment (open)");
if (tOpen1.fireSteps.length >= 2) {
  assert.ok(tOpen1.minGap >= 25, `turret fired too frequently (cooldown not enforced): minGap=${tOpen1.minGap}`);
}
assert.deepStrictEqual(tOpen1.fireSteps, tOpen2.fireSteps, "DETERMINISM FAILED: turret fire pattern differs for same seed");

// maze: line-of-sight varies -> turret must never fire through a wall
const tMaze = runTurret(7, "maze", 500, "random");
assert.strictEqual(tMaze.redFiresNoLos, 0, "turret must NOT fire without line of sight");
assert.ok(tMaze.noLosSteps > 0, "maze turret test should hit some no-LoS steps to be meaningful");

// --- Survival (big-map) mode: 2x world + central safe zone + -20% walls ---
const W2 = constants.W * 2, H2 = constants.H * 2;
const sv = new RicochetCore({ seed: 1307, arenaMode: "survival", maxSteps: 600 });
const svObs = sv.reset(1307);
const svState = sv.getPublicState();
assert.strictEqual(svState.arenaMode, "survival", "survival arenaMode");
assert.strictEqual(svState.constants.worldW, W2, "survival worldW = 2x base");
assert.strictEqual(svState.constants.worldH, H2, "survival worldH = 2x base");
assert.strictEqual(svState.viewW, constants.W, "survival viewW = base W");
assert.deepStrictEqual(svState.centerClear, { x: W2 / 4, y: H2 / 4, w: constants.W, h: constants.H }, "centerClear rect");
assert.strictEqual(svObs.length, OBS_SIZE, "survival reset obs size");
assert.ok(svObs.every((v) => Number.isFinite(v) && v >= -1.000001 && v <= 1.000001), "survival obs in range");
// --- survival poison/safe-zone obs block (last 11 dims) ---
const PB = OBS_SIZE - 11;
const svPoison0 = svObs.slice(PB);
assert.strictEqual(svPoison0[0], 0, "survival reset: poison inactive (flag 0)");
assert.strictEqual(svPoison0[1], 1, "survival reset: whole world safe (inside=1)");
assert.ok(svPoison0[10] > 0, "survival reset: timeToPoison > 0");
// open mode zeros the poison block entirely (backward-neutral)
const openPB = new RicochetCore({ seed: 1307, arenaMode: "open", spawnPowerups: false }).reset(1307).slice(PB);
assert.ok(openPB.every((v) => v === 0), "open mode: poison block all zeros");
// idle survival episode until poison is active AND the corner tank is outside the ring
const svFlee = new RicochetCore({ seed: 1307, arenaMode: "survival", spawnPowerups: false, maxSteps: 6000 });
svFlee.reset(1307);
let svOutside = null;
for (let i = 0; i < 4000; i++) {
  const out = svFlee.step(0, "none");   // blue idles in its spawn corner (180,1120)
  const pb = out.obs0.slice(PB);
  if (pb[0] === 1 && pb[1] === -1) { svOutside = pb; break; }
  if (out.done) break;
}
assert.ok(svOutside, "poison eventually activates with the idle tank outside the safe zone");
assert.ok(svOutside[9] < 0, "outside the ring -> edgeMargin negative");
assert.ok(svOutside[2] > 0 && svOutside[3] < 0, "safe centre is up-and-right of the corner tank (dx>0, dy<0)");
const svStep = sv.step(5, "laika");
assert.strictEqual(svStep.obs0.length, OBS_SIZE, "survival step obs size");
assert.ok(Number.isFinite(svStep.reward), "survival reward finite");
// center safe zone must contain no internal walls
const cc = svState.centerClear;
const centerInside = (w) => { const mx = w.x + w.w / 2, my = w.y + w.h / 2; return mx > cc.x && mx < cc.x + cc.w && my > cc.y && my < cc.y + cc.h; };
// center safe zone keeps ~50% of the edge wall density (not empty)
const svInternal = svState.walls.slice(4); // drop the 4 world-boundary walls
const svCenterWalls = svInternal.filter(centerInside).length;
const svEdgeWalls = svInternal.length - svCenterWalls;
const svCenterDensity = svCenterWalls / (cc.w * cc.h);
const svEdgeDensity = svEdgeWalls / (W2 * H2 - cc.w * cc.h);
const svDensityRatio = svCenterDensity / svEdgeDensity;
assert.ok(svCenterWalls > 0, "center safe zone should keep some walls (not fully cleared)");
assert.ok(svDensityRatio > 0.30 && svDensityRatio < 0.75, `center wall density should be ~50% of the edge (got ${svDensityRatio.toFixed(2)})`);
// world boundary walls present
assert.ok(svState.walls.some((w) => Math.abs(w.x - (W2 - constants.WALL)) < 1 && w.h >= H2 - 1), "survival world-right boundary wall");
// wall density ~20% lower than full (same base maze, only removal differs)
const svFull = makeSurvivalWorld(2024, 0.0);
const svReduced = makeSurvivalWorld(2024, 0.20);
const fullInternal = svFull.walls.length - 4, reducedInternal = svReduced.walls.length - 4;
assert.ok(reducedInternal < fullInternal, "reduced map should have fewer internal walls than full");
const dropFrac = (fullInternal - reducedInternal) / fullInternal;
assert.ok(dropFrac > 0.10 && dropFrac < 0.35, `wall density drop should be ~20% (got ${(dropFrac * 100).toFixed(1)}%)`);

// --- Stage 2/3: continuous poison ring + fractional health + no draw window ---
function poisonProbe(seed) {
  const core = new RicochetCore({ seed, arenaMode: "survival", spawnPowerups: false, maxSteps: 9000 });
  core.reset(seed);
  const worldW0 = core.getPublicState().poison.safeRect.w;  // safe rect starts at the full world width
  let healthBefore15 = null, fractionalSeen = false, result = "running", minSafeW = Infinity, shrankInward = false;
  let safeAt18 = null, safeAt22 = null, t18 = null, t22 = null;
  let outsideStartT = null, outsideStartH = null, dps = null;
  for (let i = 0; i < 2000; i++) {
    const out = core.step(0, "stationary");  // both idle -> only the closing poison ring can damage
    result = out.info.result;
    const ps = core.getPublicState();
    const t = ps.elapsed, sr = ps.poison.safeRect, t0 = ps.tanks[0];
    if (t < 14.9) healthBefore15 = t0.health;
    if (!Number.isInteger(t0.health)) fractionalSeen = true;
    if (safeAt18 === null && t >= 18) { safeAt18 = sr; t18 = t; }
    if (safeAt22 === null && t >= 22) { safeAt22 = sr; t22 = t; }
    if (sr.w < minSafeW) minSafeW = sr.w;
    if (ps.poison.active && sr.x > 4 && sr.y > 4) shrankInward = true;   // ring closing in from the edges toward centre
    const inside = t0.x >= sr.x && t0.x <= sr.x + sr.w && t0.y >= sr.y && t0.y <= sr.y + sr.h;
    if (ps.poison.active && !inside && t0.alive) {
      if (outsideStartT === null) { outsideStartT = t; outsideStartH = t0.health; }
      else if (dps === null && t - outsideStartT >= 3.0) dps = (outsideStartH - t0.health) / (t - outsideStartT);
    }
    if (out.done) break;
  }
  const rateX = (safeAt18 && safeAt22) ? (safeAt22.x - safeAt18.x) / (t22 - t18) : null;
  return { healthBefore15, fractionalSeen, dps, result, rateX, minSafeW, worldW0, shrankInward };
}
const pp = poisonProbe(1307);
assert.strictEqual(pp.healthBefore15, constants.MAX_HEALTH, "no poison/combat damage before the 16s poison start");
assert.ok(pp.rateX !== null && pp.rateX > 9 && pp.rateX < 17, `continuous shrink ~13 px/s on X (1/3 of the old ~38; got ${pp.rateX})`);
assert.ok(pp.dps !== null && Math.abs(pp.dps - 0.35) < 0.06, `poison dps ~0.35 (got ${pp.dps})`);
assert.ok(pp.fractionalSeen, "poison should produce fractional health");
assert.ok(pp.shrankInward && pp.minSafeW < pp.worldW0 * 0.85, `poison ring shrinks inward toward the centre (got minSafeW ${pp.minSafeW} / ${pp.worldW0})`);
assert.ok(pp.result === "win" || pp.result === "loss", `survival must resolve to win/loss without a draw/timeout (got ${pp.result})`);

// Stage 3: power-up durations x2 and normal shells +20%
assert.strictEqual(constants.SHIELD_DURATION, 54, "shield duration x2");
assert.strictEqual(constants.POWER_DURATION, 36, "power duration x2");
assert.strictEqual(constants.POWERUP_FIELD_TTL, 36, "powerup field ttl x2");
assert.ok(Math.abs(constants.SHELL_SPEED - (146.4 * 1.32 * 1.2)) < 1e-6, "normal shell speed +20%");

// Stage 3: opponent "none" (navigation base) - red inert (no move, no fire)
{
  const nav = new RicochetCore({ seed: 1307, arenaMode: "survival", spawnPowerups: false, maxSteps: 400 });
  nav.reset(1307);
  const startRed = { ...nav.getPublicState().tanks[1] };
  let redShells = 0;
  for (let i = 0; i < 120; i++) {
    const out = nav.step(0, "none");
    assert.strictEqual(out.obs0.length, OBS_SIZE, "nav obs size");
    redShells += nav.getPublicState().shells.filter((s) => s.owner === 1).length;
    if (out.done) break;
  }
  const endRed = nav.getPublicState().tanks[1];
  assert.ok(Math.abs(endRed.x - startRed.x) < 1e-6 && Math.abs(endRed.y - startRed.y) < 1e-6, "opponent=none red must not move");
  assert.strictEqual(redShells, 0, "opponent=none red must not fire");
}

// Stage 3: heuristic falls back to laika without crashing
{
  const h = new RicochetCore({ seed: 1307, arenaMode: "open", maxSteps: 200 });
  h.reset(1307);
  for (let i = 0; i < 50; i++) {
    const out = h.step(0, "heuristic");
    assert.strictEqual(out.obs0.length, OBS_SIZE, "heuristic-fallback obs size");
    assert.ok(Number.isFinite(out.reward), "heuristic-fallback reward finite");
  }
}

// Stage 3: laika + laika-aggressive step in the new survival map without error
for (const opp of ["laika", "aggressive"]) {
  const c = new RicochetCore({ seed: 7, arenaMode: "survival", spawnPowerups: false, maxSteps: 300 });
  c.reset(7);
  for (let i = 0; i < 120; i++) {
    const out = c.step(5, opp);
    assert.strictEqual(out.obs0.length, OBS_SIZE, `${opp} survival obs size`);
    if (out.done) break;
  }
}

// Stage 3: a poisoned laika actively moves toward safety (doesn't sit and die)
function laikaFleeProbe(seed) {
  const core = new RicochetCore({ seed, arenaMode: "survival", spawnPowerups: false, maxSteps: 9000 });
  core.reset(seed);
  for (let i = 0; i < 1300; i++) {
    const out = core.step(0, "stationary"); // red sits, gets exposed; we query what laika WOULD do
    const ps = core.getPublicState();
    const red = ps.tanks[1], sr = ps.poison.safeRect;
    const inside = red.x >= sr.x && red.x <= sr.x + sr.w && red.y >= sr.y && red.y <= sr.y + sr.h;
    if (ps.poison.active && !inside && red.alive) {
      const ctrl = core.scriptedControl(1, "laika");
      return { found: true, throttle: ctrl.throttle };
    }
    if (out.done) break;
  }
  return { found: false };
}
const lf = laikaFleeProbe(1307);
assert.ok(lf.found, "red should get caught outside the safe zone for the laika-escape check");
assert.ok(lf.throttle > 0, `a poisoned laika must move toward safety, not sit (throttle=${lf.throttle})`);

// --- Pathfind must navigate the active grid (regression: survival used the 8x6 grid) ---
function pathfindReach(seed, mode) {
  const core = new RicochetCore({ seed, arenaMode: mode, spawnPowerups: false, maxSteps: 9000 });
  core.reset(seed);
  const ps0 = core.getPublicState();
  const goal = { x: ps0.tanks[1].x, y: ps0.tanks[1].y };
  const startDist = Math.hypot(ps0.tanks[0].x - goal.x, ps0.tanks[0].y - goal.y);
  let minDist = startDist;
  const steps = mode === "survival" ? 1100 : 700;
  for (let i = 0; i < steps; i++) {
    const ctrl = core.scriptedControl(0, "pathfind", { goal });
    const out = core.step(ctrl, "stationary");
    const me = core.getPublicState().tanks[0];
    minDist = Math.min(minDist, Math.hypot(me.x - goal.x, me.y - goal.y));
    if (out.done) break;
  }
  return { startDist: Math.round(startDist), minDist: Math.round(minDist) };
}
const pfMaze = pathfindReach(1307, "maze");
const pfSurv = pathfindReach(1307, "survival");
assert.ok(pfMaze.minDist < 220, `pathfind should reach the goal on maze (start ${pfMaze.startDist} -> min ${pfMaze.minDist})`);
assert.ok(pfSurv.minDist < 220, `pathfind should reach the goal on the survival grid (start ${pfSurv.startDist} -> min ${pfSurv.minDist})`);

// --- Experimental shell decay: 2x launch speed, linear decay to ~0 over TTL ---
function shellDecayProbe(seed) {
  const core = new RicochetCore({ seed, arenaMode: "survival", spawnPowerups: false, maxSteps: 3000 });
  core.reset(seed);
  let v0 = null, age0 = null, vLast = null, ageLast = null;
  for (let i = 0; i < 400; i++) {
    const out = core.step(i === 0 ? { throttle: 0, turn: 0, fire: true } : 0, "none");
    const sh = core.getPublicState().shells.find((s) => s.owner === 0);
    if (sh) {
      const spd = Math.hypot(sh.vx, sh.vy);
      if (v0 === null) { v0 = spd; age0 = sh.age; }
      vLast = spd; ageLast = sh.age;
    }
    if (out.done) break;
  }
  return { v0, age0, vLast, ageLast };
}
const sd = shellDecayProbe(1307);
const sdBase = constants.SHELL_SPEED;
const sdExpect = (age) => 2 * sdBase * Math.max(0, 1 - age / constants.SHELL_DECAY_DURATION);  // 2x at birth -> 0 at end of decay window
assert.ok(sd.v0 !== null, "a shell should be fired");
assert.ok(Math.abs(sd.v0 - sdExpect(sd.age0)) < 0.06 * sdBase, `launch ~2x base, decaying (got ${Math.round(sd.v0)} @${sd.age0.toFixed(2)}s, expected ${Math.round(sdExpect(sd.age0))})`);
assert.ok(sd.ageLast > sd.age0 + 0.3, "shell should persist long enough to show decay");
assert.ok(sd.vLast < sd.v0, "shell speed must be decreasing");
assert.ok(Math.abs(sd.vLast - sdExpect(sd.ageLast)) < 0.06 * sdBase, `speed follows linear decay (got ${Math.round(sd.vLast)} @${sd.ageLast.toFixed(2)}s, expected ${Math.round(sdExpect(sd.ageLast))})`);
// when disabled, speed stays constant at base
const noDecay = new RicochetCore({ seed: 1307, arenaMode: "survival", spawnPowerups: false, shellDecay: false });
noDecay.reset(1307);
noDecay.step({ throttle: 0, turn: 0, fire: true }, "none");
const ndShell = noDecay.getPublicState().shells.find((s) => s.owner === 0);
assert.ok(ndShell && Math.abs(Math.hypot(ndShell.vx, ndShell.vy) - constants.SHELL_SPEED) < 1e-6, "shellDecay:false keeps the original constant speed");

console.log(JSON.stringify({
  ok: true,
  source: "game_core.js (no DOM shim)",
  actions: ACTION_TABLE.length,
  obsSize: OBS_SIZE,
  wallRayCount: RAY_OFFSETS.length,
  poisonObsOutside: svOutside ? svOutside.map((v) => Math.round(v * 100) / 100) : null,
  openPoisonBlockAllZero: openPB.every((v) => v === 0),
  maze_seed1307: { steps: a.steps, result: a.result, totalReward: Number(a.totalReward.toFixed(3)) },
  open_seed1307: { steps: c.steps, result: c.result, totalReward: Number(c.totalReward.toFixed(3)) },
  open_jitter: { blueY: Number(jA.blueY.toFixed(2)), redY: Number(jA.redY.toFixed(2)), deterministic: true },
  survival: { worldW: W2, worldH: H2, internal_walls: svState.walls.length - 4, wall_drop_pct: Number(((fullInternal - reducedInternal) / fullInternal * 100).toFixed(1)), centerWalls: svCenterWalls, centerDensityRatio: Number(svDensityRatio.toFixed(2)) },
  poison: { shrinkRateX: pp.rateX === null ? null : Number(pp.rateX.toFixed(2)), dps: pp.dps === null ? null : Number(pp.dps.toFixed(3)), result: pp.result, neverOvershoot: !pp.everOvershoot },
  stage3: { shellSpeed: Number(constants.SHELL_SPEED.toFixed(1)), shieldDuration: constants.SHIELD_DURATION, powerupFieldTtl: constants.POWERUP_FIELD_TTL, laikaFleeThrottle: lf.throttle },
  pathfind: { maze: pfMaze, survival: pfSurv },
  shellDecay: { v0: Math.round(sd.v0), base: Math.round(constants.SHELL_SPEED), vLast: Math.round(sd.vLast), ageLast: Number(sd.ageLast.toFixed(1)) },
  turret: {
    open_fires: tOpen1.redFires,
    open_min_gap: tOpen1.minGap === Infinity ? null : tOpen1.minGap,
    maze_noLos_steps: tMaze.noLosSteps,
    fires_without_los: tMaze.redFiresNoLos,
    deterministic: true
  },
  deterministic_same_seed: true
}, null, 2));
