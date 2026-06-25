"use strict";
// moba_poison_run (跑毒课) headless smoke tests. Run: node smoke_moba_poison_run.js
// A NAV lesson: the moba map + shrinking poison ring + red removed, reusing the route
// lesson's dense BFS-to-centre reward/success. Designed to warm-start from a
// nav_route_to_center model (opponent obs stay zeroed, same obs/action space).
const assert = require("assert");
const { RicochetCore, constants, OBS_SIZE } = require("./game_core.js");

const SEED = 1307;
const CW = constants.CELL_W, CH = constants.CELL_H, WW = constants.WORLD_W, WH = constants.WORLD_H;
const SCOLS = constants.SCOLS, SROWS = constants.SROWS;
const makeRun = (extra) => { const c = new RicochetCore(Object.assign({ seed: SEED, arenaMode: "survival", scenario: "moba_poison_run", maxSteps: 1800 }, extra || {})); c.reset(SEED); return c; };
const cc = (x, y) => ({ x: x * CW + CW / 2, y: y * CH + CH / 2 });
const park = (c, x, y) => { c.state.tanks[0].x = x; c.state.tanks[0].y = y; };
const inCC = (cl, x, y) => x > cl.x && x < cl.x + cl.w && y > cl.y && y < cl.y + cl.h;
const cellOf = (px, py) => ({ cx: Math.max(0, Math.min(SCOLS - 1, Math.floor(px / CW))), cy: Math.max(0, Math.min(SROWS - 1, Math.floor(py / CH))) });
function bfs(cells, cl, sx, sy) {
  const s = cellOf(sx, sy), seen = Array.from({ length: SROWS }, () => new Array(SCOLS).fill(false));
  const q = [[s.cx, s.cy, 0]]; seen[s.cy][s.cx] = true;
  for (let h = 0; h < q.length; h++) { const [x, y, d] = q[h]; if (inCC(cl, x * CW + CW / 2, y * CH + CH / 2)) return d; const w = cells[y][x].walls;
    if (!w.n && y > 0 && !seen[y - 1][x]) { seen[y - 1][x] = true; q.push([x, y - 1, d + 1]); }
    if (!w.e && x < SCOLS - 1 && !seen[y][x + 1]) { seen[y][x + 1] = true; q.push([x + 1, y, d + 1]); }
    if (!w.s && y < SROWS - 1 && !seen[y + 1][x]) { seen[y + 1][x] = true; q.push([x, y + 1, d + 1]); }
    if (!w.w && x > 0 && !seen[y][x - 1]) { seen[y][x - 1] = true; q.push([x - 1, y, d + 1]); } }
  return Infinity;
}
const R = {};

// (1) reset + obs 105 + finite/range + scenario id
{
  const obs = makeRun().buildObservation(0);
  assert.strictEqual(OBS_SIZE, 105, "OBS_SIZE 105");
  assert.strictEqual(obs.length, 105, "obs length 105");
  assert.ok(obs.every((v) => Number.isFinite(v) && v >= -1.000001 && v <= 1.000001), "obs finite & in range");
  assert.strictEqual(makeRun().getPublicState().scenario, "moba_poison_run", "scenario id");
}

// (2) NAV lesson: red removed + opponent obs block zeroed (warm-start match with route)
{
  const c = makeRun();
  assert.strictEqual(c.getPublicState().tanks[1].alive, false, "red removed (nav)");
  assert.ok(c.buildObservation(0).slice(15, 25).every((v) => v === 0), "opponent 10 obs zeroed (route-compatible)");
}

// (3) uses the fixed moba map (same walls as fixed_moba) + survival centre + no powerups
{
  const ps = makeRun().getPublicState();
  const fixed = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "fixed_moba" }); fixed.reset(SEED);
  assert.strictEqual(ps.walls.length, fixed.getPublicState().walls.length, "reuses the moba wall layout");
  assert.deepStrictEqual({ x: ps.centerClear.x, y: ps.centerClear.y, w: ps.centerClear.w, h: ps.centerClear.h }, { x: 480, y: 320, w: 960, h: 640 }, "survival centre");
  assert.strictEqual(ps.worldW, 1920, "worldW 1920");
  assert.strictEqual(ps.spawnPowerups, false, "no power-ups");
  R.walls = ps.walls.length;
}

// (4) poison ring LOADED (unlike route) and activates past startTime
{
  assert.ok(makeRun().getPublicState().poison !== null, "poison enabled");
  const route = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "nav_route_to_center" }); route.reset(SEED);
  assert.strictEqual(route.getPublicState().poison, null, "nav_route_to_center stays poison-off");
  const c = makeRun(); let active = false;
  for (let i = 0; i < 360 && !active; i++) { c.step(0, "none"); active = c.getPublicState().poison.active; }
  assert.ok(active, "poison ring activates past startTime");
  R.poison = { enabled: true, activated: active };
}

// (5) blue spawns OUTSIDE the centre, BFS spawn->centre finite (must run in)
{
  const ps = makeRun().getPublicState(), cl = ps.centerClear, b = ps.tanks[0];
  assert.ok(!inCC(cl, b.x, b.y), "blue starts outside the centre safe zone");
  assert.ok(Number.isFinite(bfs(ps.maze, cl, b.x, b.y)), "blue->centre reachable on the moba map");
  R.blueSpawn = { x: Math.round(b.x), y: Math.round(b.y) };
}

// (6) route reward active: BFS pathDist finite + smaller closer to centre (dense shaping)
{
  const c = makeRun();
  park(c, cc(1, 1).x, cc(1, 1).y); const far = c.step(0, "none").info;
  park(c, cc(6, 5).x, cc(6, 5).y); const near = c.step(0, "none").info;
  assert.ok(far.pathDist >= 0 && far.pathDist > near.pathDist, "BFS pathDist smaller nearer the centre");
  R.pathDist = { far: far.pathDist, near: near.pathDist };
}

// (7) reaching the centre & staying 2s -> route_success (== surviving the poison)
{
  const c = makeRun(); const cl = c.getPublicState().centerClear; let info = null;
  for (let i = 0; i < 200; i++) { park(c, cl.x + cl.w / 2, cl.y + cl.h / 2); const o = c.step(0, "none"); if (o.done) { info = o.info; break; } }
  assert.ok(info && info.result === "route_success", "stay-in-centre -> route_success");
  assert.ok(info.routeSuccess === true, "routeSuccess flag set");
  R.success = { result: info.result, centerStayTime: info.centerStayTime };
}

// (8) timeout while never reaching centre (and not dying yet) -> route_timeout
{
  const c = makeRun({ maxSteps: 40 }); let info = null;
  for (let i = 0; i < 80; i++) { park(c, cc(1, 1).x, cc(1, 1).y); const o = c.step(0, "none"); if (o.done) { info = o.info; break; } }
  assert.ok(info && info.result === "route_timeout", "maxSteps -> route_timeout");
  R.timeout = info.result;
}

// (9) poison actually kills a tank that never reaches safety -> route_death (the 跑毒 stake)
{
  const c = makeRun();
  c.state.elapsed = 30; c.state.poisonActive = true;
  c.state.safeRect = { x: 880, y: 600, w: 160, h: 80 };
  let info = null;
  for (let i = 0; i < 400; i++) { park(c, 200, 200); const o = c.step(0, "none"); if (o.done) { info = o.info; break; } }
  assert.ok(info && info.result === "route_death", "poison kills a stuck-outside tank -> route_death");
  R.poisonDeath = { result: info.result, poisonDamageTaken: info.poisonDamageTaken };
}

// (10) regression: nav_route_to_center untouched (random maze, poison off) + battle intact
{
  const r = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "nav_route_to_center" }); r.reset(SEED);
  assert.strictEqual(r.getPublicState().poison, null, "route poison still off");
  const fixedWalls = (() => { const f = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "fixed_moba" }); f.reset(SEED); return f.getPublicState().walls.length; })();
  assert.notStrictEqual(r.getPublicState().walls.length, fixedWalls, "route still uses the RANDOM maze, not the moba map");
  const b = new RicochetCore({ seed: SEED, arenaMode: "survival" });
  assert.strictEqual(b.reset(SEED).length, 105, "battle obs 105");
  assert.strictEqual(b.getPublicState().scenario, "battle", "battle scenario intact");
}

console.log("MOBA_POISON_RUN SMOKE OK");
console.log(JSON.stringify(R, null, 2));
