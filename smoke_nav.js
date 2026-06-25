"use strict";
// Lesson 1 (nav_powerup_poison) headless smoke tests. Run: node smoke_nav.js
const assert = require("assert");
const { RicochetCore, makeSurvivalWorld, constants, OBS_SIZE } = require("./game_core.js");

const SEED = 1307;
function makeNav(extra) {
  const c = new RicochetCore(Object.assign(
    { seed: SEED, arenaMode: "survival", scenario: "nav_powerup_poison", spawnPowerups: true, maxSteps: 2200 },
    extra || {}));
  c.reset(SEED);
  return c;
}
const R = {};

// (1,2) obs size + range; open/maze still 105
{
  const obs = makeNav().buildObservation(0);
  assert.strictEqual(OBS_SIZE, 105, "OBS_SIZE === 105");
  assert.strictEqual(obs.length, 105, "nav reset obs length === 105");
  assert.ok(obs.every((v) => Number.isFinite(v) && v >= -1.000001 && v <= 1.000001), "nav obs finite & in [-1,1]");
  for (const m of ["open", "maze"]) {
    assert.strictEqual(new RicochetCore({ seed: SEED, arenaMode: m }).reset(SEED).length, 105, m + " obs 105");
  }
}

// (3) centerClear internal walls: 0 in nav, >0 in normal survival
{
  const w0 = makeSurvivalWorld(SEED, 0.20, true);
  const w1 = makeSurvivalWorld(SEED, 0.20, false);
  const cc = w0.centerClear;
  const centerInside = (w) => { const mx = w.x + w.w / 2, my = w.y + w.h / 2; return mx > cc.x && mx < cc.x + cc.w && my > cc.y && my < cc.y + cc.h; };
  const nav = w0.walls.slice(4).filter(centerInside).length;
  const normal = w1.walls.slice(4).filter(centerInside).length;
  assert.strictEqual(nav, 0, "nav centerClear internal walls === 0");
  assert.ok(normal > 0, "normal survival keeps center walls (50%)");
  R.centerWalls = { nav, normal };
}

// (4) blue spawns outside centerClear
{
  const ps = makeNav().getPublicState();
  const b = ps.tanks[0], cc = ps.centerClear;
  const inside = b.x > cc.x && b.x < cc.x + cc.w && b.y > cc.y && b.y < cc.y + cc.h;
  assert.ok(!inside, "blue spawn is outside centerClear");
  R.blueSpawn = { x: Math.round(b.x), y: Math.round(b.y), insideCenter: inside };
}

// (5) opponent 10 obs features zeroed + red dead, no win/loss
{
  const c = makeNav();
  const opp = c.buildObservation(0).slice(15, 25); // self(9)+powerOneHot(6) -> opponent[15,25)
  assert.ok(opp.every((v) => v === 0), "opponent 10 obs features all zero in nav");
  assert.strictEqual(c.getPublicState().tanks[1].alive, false, "red is dead in nav");
}

// (7,8,9) poison reaches min circle + survive 12s parked at safe centre -> nav_success
{
  const c = makeNav({ maxSteps: 2500 });
  const cc = c.getPublicState().centerClear;
  const ccx = cc.x + cc.w / 2, ccy = cc.y + cc.h / 2;
  let info = null, sawMin = false;
  for (let i = 0; i < 2500; i++) {
    c.state.tanks[0].x = ccx; c.state.tanks[0].y = ccy; // keep blue parked at the safe centre
    const out = c.step(0, "none");
    if (out.info.poisonAtMinCircle) sawMin = true;
    if (out.done) { info = out.info; break; }
  }
  assert.ok(sawMin, "poisonAtMinCircle became true");
  assert.ok(info && info.navSuccess === true, "navSuccess true");
  assert.ok(/^nav_success/.test(info.result), "result is nav_success_*");
  assert.ok(info.survivalAfterMinCircle >= 12, "survivalAfterMinCircle >= 12");
  R.success = { result: info.result, step: info.steps, elapsed: info.elapsed, afterMin: info.survivalAfterMinCircle };
}

// (10) parked outside -> nav_death
{
  const c = makeNav({ maxSteps: 2500 });
  let info = null;
  for (let i = 0; i < 2500; i++) {
    c.state.tanks[0].x = 100; c.state.tanks[0].y = 100;
    const out = c.step(0, "none");
    if (out.done) { info = out.info; break; }
  }
  assert.ok(info && info.result === "nav_death", "parked-outside blue -> nav_death");
  R.death = { result: info.result, elapsed: info.elapsed };
}

// (11) wall hit increments wallHits with cooldown (not every frame)
{
  const c = makeNav();
  c.state.tanks[0].angle = Math.PI; // face -x into the left boundary
  c.state.tanks[0].x = constants.WALL + constants.TANK_RADIUS + 2;
  c.state.tanks[0].y = 600;
  const STEPS = 60;
  for (let i = 0; i < STEPS; i++) c.step({ throttle: 1, turn: 0, fire: false }, "none");
  const wh = c.getPublicState().nav.wallHits;
  assert.ok(wh >= 1, "wall hits registered");
  assert.ok(wh < STEPS, "wall hits throttled by cooldown, not every step");
  R.wallHits = { overSteps: STEPS, hits: wh };
}

// (12) empty-handed pickup -> pickupsWhenEmpty + reward
{
  const c = makeNav();
  const b = c.state.tanks[0];
  c.state.powerups.push({ x: b.x, y: b.y, radius: 13, type: "rapid", label: "R", color: "#fff", ttl: 36 });
  const out = c.step(0, "none");
  assert.strictEqual(out.info.pickupsWhenEmpty, 1, "empty pickup counted");
  assert.strictEqual(out.info.pickedAnyPowerup, true, "pickedAnyPowerup true");
  assert.ok(out.reward > 0.5, "pickupWhenEmpty reward (~0.8) applied");
  R.pickup = { pickupsWhenEmpty: out.info.pickupsWhenEmpty, reward: Math.round(out.reward * 1000) / 1000 };
}

// (13) poison damage -> poisonDamageTaken + negative step reward
{
  const c = makeNav({ maxSteps: 3000 });
  let info = null, rew = null;
  for (let i = 0; i < 3000; i++) {
    c.state.tanks[0].x = 120; c.state.tanks[0].y = 120;
    const out = c.step(0, "none");
    if (out.info.poisonDamageTaken > 0) { info = out.info; rew = out.reward; break; }
    if (out.done) break;
  }
  assert.ok(info && info.poisonDamageTaken > 0, "poisonDamageTaken > 0 while outside");
  assert.ok(rew < 0, "step taking poison damage has negative reward");
  R.poisonDamage = { taken: Math.round(info.poisonDamageTaken * 1000) / 1000, stepReward: Math.round(rew * 1000) / 1000 };
}

// (extra) fire action is inert in nav -> no shells, no self-suicide
{
  const c = makeNav();
  for (let i = 0; i < 200; i++) c.step({ throttle: 0, turn: 0, fire: true }, "none");
  const ps = c.getPublicState();
  assert.strictEqual(ps.shells.length, 0, "no shells spawned in nav (fire is inert)");
  assert.strictEqual(ps.tanks[0].alive, true, "blue cannot suicide by firing in nav");
  R.fireInert = { shells: ps.shells.length, blueAlive: ps.tanks[0].alive };
}

// regression: battle survival untouched by nav code
{
  const c = new RicochetCore({ seed: SEED, arenaMode: "survival" });
  assert.strictEqual(c.reset(SEED).length, 105, "battle survival obs 105");
  assert.strictEqual(c.getPublicState().scenario, "battle", "default scenario === battle");
  assert.strictEqual(c.getPublicState().tanks[1].alive, true, "battle survival red alive");
}

R.maxStepsNote = "core honors any maxSteps; training-script default is 2200";
console.log("NAV SMOKE OK");
console.log(JSON.stringify(R, null, 2));
