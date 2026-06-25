"use strict";
// Fixed MOBA arena (scenario "fixed_moba") headless smoke tests. Run: node smoke_fixed_moba.js
// fixed_moba is a COMBAT scenario on a fixed 4-fold-symmetric survival-sized map.
const assert = require("assert");
const { RicochetCore, constants, OBS_SIZE } = require("./game_core.js");

const SEED = 1307;
const CW = constants.CELL_W, CH = constants.CELL_H;
const WW = constants.WORLD_W, WH = constants.WORLD_H;
const SCOLS = constants.SCOLS, SROWS = constants.SROWS;
const TANK_R = constants.TANK_RADIUS, MAXH = constants.MAX_HEALTH;

function makeMoba(extra) {
  const c = new RicochetCore(Object.assign(
    { seed: SEED, arenaMode: "survival", scenario: "fixed_moba", maxSteps: 1800 }, extra || {}));
  c.reset(SEED);
  return c;
}
const inCC = (cc, x, y) => x > cc.x && x < cc.x + cc.w && y > cc.y && y < cc.y + cc.h;
const cellOf = (px, py) => ({ cx: Math.max(0, Math.min(SCOLS - 1, Math.floor(px / CW))), cy: Math.max(0, Math.min(SROWS - 1, Math.floor(py / CH))) });
function circleRectHit(cx, cy, rad, r) {
  const nx = Math.max(r.x, Math.min(cx, r.x + r.w)), ny = Math.max(r.y, Math.min(cy, r.y + r.h));
  return (cx - nx) ** 2 + (cy - ny) ** 2 <= rad * rad;
}
function bfsToCenter(cells, cc, sx, sy) {
  const s = cellOf(sx, sy);
  const seen = Array.from({ length: SROWS }, () => new Array(SCOLS).fill(false));
  const q = [[s.cx, s.cy, 0]]; seen[s.cy][s.cx] = true;
  for (let h = 0; h < q.length; h++) {
    const [x, y, d] = q[h];
    if (inCC(cc, x * CW + CW / 2, y * CH + CH / 2)) return d;
    const w = cells[y][x].walls;
    if (!w.n && y > 0 && !seen[y - 1][x]) { seen[y - 1][x] = true; q.push([x, y - 1, d + 1]); }
    if (!w.e && x < SCOLS - 1 && !seen[y][x + 1]) { seen[y][x + 1] = true; q.push([x + 1, y, d + 1]); }
    if (!w.s && y < SROWS - 1 && !seen[y + 1][x]) { seen[y + 1][x] = true; q.push([x, y + 1, d + 1]); }
    if (!w.w && x > 0 && !seen[y][x - 1]) { seen[y][x - 1] = true; q.push([x - 1, y, d + 1]); }
  }
  return Infinity;
}
const R = {};

// (1) reset works + (2) obs length 105 + (3) finite & in range
{
  const obs = makeMoba().buildObservation(0);
  assert.strictEqual(OBS_SIZE, 105, "OBS_SIZE 105");
  assert.strictEqual(obs.length, 105, "fixed_moba reset obs length 105");
  assert.ok(obs.every((v) => Number.isFinite(v) && v >= -1.000001 && v <= 1.000001), "obs finite & in [-1,1]");
  assert.strictEqual(makeMoba().getPublicState().scenario, "fixed_moba", "scenario reported fixed_moba");
}

// (4) world size is survival-sized 1920x1280
{
  const ps = makeMoba().getPublicState();
  assert.strictEqual(ps.worldW, 1920, "worldW 1920");
  assert.strictEqual(ps.worldH, 1280, "worldH 1280");
  assert.strictEqual(SCOLS, 16, "SCOLS 16"); assert.strictEqual(SROWS, 12, "SROWS 12");
  R.world = { w: ps.worldW, h: ps.worldH, cols: SCOLS, rows: SROWS };
}

// (5) wall count is ~60% of a random survival maze: fewer than survival, but dense
//     enough to keep the shell-ricochet physics meaningful (not a near-empty arena).
{
  const moba = makeMoba().getPublicState().walls.length;
  const surv = new RicochetCore({ seed: SEED, arenaMode: "survival" }); surv.reset(SEED);
  const survN = surv.getPublicState().walls.length;
  assert.ok(moba < survN && moba >= survN * 0.5, `fixed_moba walls (${moba}) ~60% of survival (${survN}): fewer but dense`);
  R.walls = { fixedMoba: moba, survivalRandom: survN, ratioPct: Math.round(moba / survN * 100) };
}

// (6) centre keeps the survival geometry and carries only DISCRETE, gap-rich cover
//     (ricochet surfaces) — not empty, not walled-in. Traversability is asserted in (10).
{
  const ps = makeMoba().getPublicState(), cc = ps.centerClear;
  assert.deepStrictEqual({ x: cc.x, y: cc.y, w: cc.w, h: cc.h }, { x: 480, y: 320, w: 960, h: 640 }, "centerClear == survival centre");
  const inside = ps.walls.filter((r) => inCC(cc, r.x + r.w / 2, r.y + r.h / 2)).length;
  assert.ok(inside > 0 && inside <= 24, "centre carries discrete cover (not empty, not walled-in)");
  R.centerCover = inside;
}

// (7) blue & red spawns are not inside any wall
{
  const ps = makeMoba().getPublicState(), t = ps.tanks;
  assert.ok(!ps.walls.some((r) => circleRectHit(t[0].x, t[0].y, TANK_R, r)), "blue spawn not in wall");
  assert.ok(!ps.walls.some((r) => circleRectHit(t[1].x, t[1].y, TANK_R, r)), "red spawn not in wall");
  // mirror-symmetric spawns about the world centre
  assert.ok(Math.abs((t[0].x + t[1].x) - WW) < 1 && Math.abs(t[0].y - t[1].y) < 1, "blue/red spawns left-right symmetric");
  R.spawns = { blue: { x: Math.round(t[0].x), y: Math.round(t[0].y) }, red: { x: Math.round(t[1].x), y: Math.round(t[1].y) } };
}

// (8,9) blue->centre and red->centre BFS distances finite
{
  const ps = makeMoba().getPublicState(), cc = ps.centerClear, t = ps.tanks;
  const bd = bfsToCenter(ps.maze, cc, t[0].x, t[0].y), rd = bfsToCenter(ps.maze, cc, t[1].x, t[1].y);
  assert.ok(Number.isFinite(bd), "blue->centre BFS finite");
  assert.ok(Number.isFinite(rd), "red->centre BFS finite");
  R.bfsSpawn = { blue: bd, red: rd };
}

// (10) all 4 corners + 4 lane mid-edges reach the centre, and EVERY cell is reachable
{
  const ps = makeMoba().getPublicState(), cc = ps.centerClear;
  const probes = {
    TL: [CW * 1.5, CH * 1.5], TR: [WW - CW * 1.5, CH * 1.5], BL: [CW * 1.5, WH - CH * 1.5], BR: [WW - CW * 1.5, WH - CH * 1.5],
    leftMid: [CW * 0.5, WH / 2], rightMid: [WW - CW * 0.5, WH / 2], topMid: [WW / 2, CH * 0.5], botMid: [WW / 2, WH - CH * 0.5]
  };
  for (const [k, [x, y]] of Object.entries(probes)) assert.ok(Number.isFinite(bfsToCenter(ps.maze, cc, x, y)), `${k}->centre reachable`);
  let unreachable = 0;
  for (let y = 0; y < SROWS; y++) for (let x = 0; x < SCOLS; x++) if (!Number.isFinite(bfsToCenter(ps.maze, cc, x * CW + CW / 2, y * CH + CH / 2))) unreachable++;
  assert.strictEqual(unreachable, 0, "every cell reaches the centre (no isolated pockets)");
  // no deep dead ends: no cell walled on >=3 sides (incl boundary)
  let deadends = 0;
  for (let y = 0; y < SROWS; y++) for (let x = 0; x < SCOLS; x++) {
    const w = ps.maze[y][x].walls; let b = 0;
    if (w.n || y === 0) b++; if (w.e || x === SCOLS - 1) b++; if (w.s || y === SROWS - 1) b++; if (w.w || x === 0) b++;
    if (b >= 3) deadends++;
  }
  assert.strictEqual(deadends, 0, "no deep dead-ends (no cell blocked on >=3 sides)");
  R.unreachable = unreachable; R.deadends = deadends;
}

// (11) the map is up/down AND left/right symmetric (wall-rect set closed under both reflections)
{
  const ps = makeMoba().getPublicState();
  const key = (r) => `${Math.round(r.x)},${Math.round(r.y)},${Math.round(r.w)},${Math.round(r.h)}`;
  const set = new Set(ps.walls.map(key));
  const mh = (r) => ({ x: WW - r.x - r.w, y: r.y, w: r.w, h: r.h });
  const mv = (r) => ({ x: r.x, y: WH - r.y - r.h, w: r.w, h: r.h });
  assert.ok(ps.walls.every((r) => set.has(key(mh(r)))), "left-right symmetric");
  assert.ok(ps.walls.every((r) => set.has(key(mv(r)))), "up-down symmetric");
  R.symmetric = { leftRight: true, upDown: true };
}

// (12) red=none -> no valid enemy: red is inert (never moves, deals no damage)
{
  const c = makeMoba();
  const ps0 = c.getPublicState(); const r0 = { x: ps0.tanks[1].x, y: ps0.tanks[1].y };
  let info = null;
  for (let i = 0; i < 90; i++) info = c.step(0, "none").info;   // blue idle, red "none"
  const ps = c.getPublicState();
  assert.ok(Math.hypot(ps.tanks[1].x - r0.x, ps.tanks[1].y - r0.y) < 1, "red(none) does not move");
  assert.strictEqual(ps.tanks[0].health, MAXH, "blue takes no damage from an inert red");
  assert.strictEqual(info.result, "running", "no premature win/loss with red=none");
  R.redNone = { redMoved: false, blueHealth: ps.tanks[0].health };
}

// (13) combat path works (watch/human use it): red=laika is a live enemy that moves
{
  const c = makeMoba();
  const r0 = { x: c.getPublicState().tanks[1].x, y: c.getPublicState().tanks[1].y };
  let moved = 0;
  for (let i = 0; i < 60; i++) {
    c.step(0, "laika");
    const r = c.getPublicState().tanks[1];
    moved = Math.max(moved, Math.hypot(r.x - r0.x, r.y - r0.y));
    if (c.isDone()) break;
  }
  assert.ok(moved > TANK_R, "red(laika) actively moves -> combat scenario live");
  assert.strictEqual(c.getPublicState().tanks[1].alive || c.isDone(), true, "red alive (or round resolved) under laika");
  R.combat = { redMaxDisplacement: Math.round(moved) };
}

// (14) regression: other scenarios/arenas untouched
{
  for (const mode of ["open", "maze", "survival"]) {
    const c = new RicochetCore({ seed: SEED, arenaMode: mode });
    assert.strictEqual(c.reset(SEED).length, 105, `${mode} obs 105`);
    assert.strictEqual(c.getPublicState().scenario, "battle", `${mode} default scenario battle`);
  }
  const route = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "nav_route_to_center" }); route.reset(SEED);
  assert.strictEqual(route.getPublicState().scenario, "nav_route_to_center", "route scenario intact");
  assert.strictEqual(route.getPublicState().tanks[1].alive, false, "route still removes red");
  const poison = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "nav_powerup_poison" }); poison.reset(SEED);
  assert.strictEqual(poison.getPublicState().poison !== null, true, "poison-nav still has poison enabled");
  // fixed_moba leaves poison OFF by default but can opt in via config
  assert.strictEqual(makeMoba().getPublicState().poison, null, "fixed_moba poison off by default");
  const mobaPoison = makeMoba({ poisonEnabled: true });
  assert.strictEqual(mobaPoison.getPublicState().poison !== null, true, "fixed_moba poison opt-in via config");
}

console.log("FIXED_MOBA SMOKE OK");
console.log(JSON.stringify(R, null, 2));
