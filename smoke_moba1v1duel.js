"use strict";
// moba1v1duel headless smoke tests. Run: node smoke_moba1v1duel.js
// moba1v1duel = the fixed symmetric MOBA map + poison ring + duel mechanics:
//   breathing regen, a 5-shot 2x weapon burst, and -30% turn sensitivity.
const assert = require("assert");
const { RicochetCore, constants, OBS_SIZE } = require("./game_core.js");

const SEED = 1307;
const WW = constants.WORLD_W, WH = constants.WORLD_H, MAXH = constants.MAX_HEALTH;
const TAU = Math.PI * 2;
const makeDuel = (extra) => { const c = new RicochetCore(Object.assign({ seed: SEED, arenaMode: "survival", scenario: "moba1v1duel", maxSteps: 4000 }, extra || {})); c.reset(SEED); return c; };
const makeFixed = () => { const c = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "fixed_moba", maxSteps: 1800 }); c.reset(SEED); return c; };
const park = (c, x, y) => { c.state.tanks[0].x = x; c.state.tanks[0].y = y; };
const R = {};

// (1) reset + obs 105 + finite/range + scenario id
{
  const obs = makeDuel().buildObservation(0);
  assert.strictEqual(OBS_SIZE, 105, "OBS_SIZE 105");
  assert.strictEqual(obs.length, 105, "duel obs length 105");
  assert.ok(obs.every((v) => Number.isFinite(v) && v >= -1.000001 && v <= 1.000001), "obs finite & in range");
  assert.strictEqual(makeDuel().getPublicState().scenario, "moba1v1duel", "scenario id");
}

// (2) same fixed symmetric map as fixed_moba, red alive (combat), edge-symmetric spawns
{
  const d = makeDuel().getPublicState(), f = makeFixed().getPublicState();
  assert.strictEqual(d.walls.length, f.walls.length, "duel reuses the fixed_moba wall layout");
  const key = (r) => `${Math.round(r.x)},${Math.round(r.y)},${Math.round(r.w)},${Math.round(r.h)}`;
  const set = new Set(d.walls.map(key));
  assert.ok(d.walls.every((r) => set.has(key({ x: WW - r.x - r.w, y: r.y, w: r.w, h: r.h }))), "left-right symmetric");
  assert.ok(d.walls.every((r) => set.has(key({ x: r.x, y: WH - r.y - r.h, w: r.w, h: r.h }))), "up-down symmetric");
  assert.strictEqual(d.tanks[1].alive, true, "red is a live opponent");
  const t = d.tanks;
  assert.ok(Math.abs((t[0].x + t[1].x) - WW) < 1 && Math.abs(t[0].x - WW * 0.12) < 1, "spawns edge-ward & mirror-symmetric");
  R.spawns = { blue: { x: Math.round(t[0].x), y: Math.round(t[0].y) }, red: { x: Math.round(t[1].x), y: Math.round(t[1].y) } };
  R.walls = d.walls.length;
}

// (3) poison ring is LOADED onto the moba map (off for fixed_moba) and activates after startTime
{
  assert.ok(makeDuel().getPublicState().poison !== null, "poison enabled on moba1v1duel");
  assert.strictEqual(makeFixed().getPublicState().poison, null, "poison still OFF for fixed_moba");
  const c = makeDuel();
  let active = false;
  for (let i = 0; i < 360 && !active; i++) { c.step(0, "none"); active = c.getPublicState().poison.active; }  // ~20s -> ring starts
  assert.ok(active, "poison ring activates past startTime");
  R.poison = { enabled: true, activated: active };
}

// (4) breathing regen: undamaged for 5s -> heals ~0.5 HP/s up to MAX
{
  const c = makeDuel();
  c.state.tanks[0].health = 1.0; c.state.tanks[0].lastHitAt = 0;
  for (let i = 0; i < 150; i++) { park(c, 240, 640); c.step(0, "none"); }  // ~10s idle, no damage (poison not yet active)
  const hp = c.state.tanks[0].health;
  assert.ok(hp > 1.0, "health regenerated after 5s without damage");
  assert.ok(hp <= MAXH + 1e-9, "regen capped at MAX_HEALTH");
  R.regen = { from: 1.0, to: Math.round(hp * 100) / 100 };
}

// (5) regen is interrupted by a hit: no regen within regenDelay (5s) of taking damage
{
  const c = makeDuel();
  // advance ~6s so the clock is well past regenDelay, then "take a hit" now.
  for (let i = 0; i < 90; i++) { park(c, 240, 640); c.step(0, "none"); }
  c.state.tanks[0].health = 1.0; c.state.tanks[0].lastHitAt = c.state.elapsed;  // just hit
  for (let i = 0; i < 60; i++) { park(c, 240, 640); c.step(0, "none"); }        // ~4s < 5s since hit
  assert.strictEqual(c.state.tanks[0].health, 1.0, "no regen within 5s of a hit (interrupted)");
  R.interrupt = { heldAt: c.state.tanks[0].health };
}

// (6) poison damage also interrupts regen (and actually damages outside the safe zone)
{
  const c = makeDuel();
  c.state.elapsed = 30; c.state.poisonActive = true;
  c.state.safeRect = { x: 880, y: 600, w: 160, h: 80 };          // tiny safe zone near centre
  c.state.tanks[0].health = 1.5; c.state.tanks[0].lastHitAt = 0; // would regen (30s>5s) if not for poison
  for (let i = 0; i < 12; i++) { park(c, 200, 200); c.step(0, "none"); }  // parked OUTSIDE the safe zone
  assert.ok(c.state.tanks[0].health < 1.5, "poison damages & interrupts regen outside the safe zone");
  R.poisonDmg = { to: Math.round(c.state.tanks[0].health * 1000) / 1000 };
}

// (7) ONLY the Rapid (lightning) skill becomes a 5-shot burst; every other power keeps 2
{
  const pickup = (c, type) => { const b = c.state.tanks[0]; c.state.powerups.push({ x: b.x, y: b.y, radius: 13, type, label: type, color: "#fff", ttl: 30 }); c.step(0, "none"); return b.powerShots; };
  assert.strictEqual(pickup(makeDuel(), "rapid"), 5, "duel Rapid = 5-shot burst");
  assert.strictEqual(pickup(makeDuel(), "triple"), 2, "duel Triple unchanged (2)");
  assert.strictEqual(pickup(makeDuel(), "missile"), 2, "duel Missile unchanged (2)");
  assert.strictEqual(pickup(makeFixed(), "rapid"), 2, "fixed_moba Rapid still 2");
  R.weapon = { rapidDuel: 5, otherDuel: 2, rapidFixed: 2 };
}

// (7b) Rapid skill fires 2x-speed bullets in the duel; other shots stay normal speed
{
  const fireSpeed = (power) => { const c = makeDuel(); const b = c.state.tanks[0]; b.power = power; b.reload = 0; if (power) b.powerShots = 5; c.step({ throttle: 0, turn: 0, fire: true }, "none"); const s = c.state.shells[0]; return s ? Math.hypot(s.vx, s.vy) : 0; };
  const rapidSpeed = fireSpeed("rapid"), normSpeed = fireSpeed(null);
  assert.ok(rapidSpeed > 0 && normSpeed > 0, "both shots fired");
  const ratio = rapidSpeed / normSpeed;
  assert.ok(Math.abs(ratio - 2) < 0.15, `Rapid bullets ~2x normal speed (got ${ratio.toFixed(2)})`);
  R.rapid2x = { ratio: Math.round(ratio * 100) / 100 };
}

// (8) turn sensitivity reduced 30%: duel turn delta ≈ 0.7x a normal survival battle
{
  const turnFor = (c) => { c.state.tanks[0].angle = 0; for (let i = 0; i < 5; i++) c.step({ throttle: 0, turn: 1, fire: false }, "none"); return c.state.tanks[0].angle; };
  const duel = makeDuel();
  const batt = new RicochetCore({ seed: SEED, arenaMode: "survival" }); batt.reset(SEED);
  const dAng = turnFor(duel), bAng = turnFor(batt);
  const ratio = dAng / bAng;
  assert.ok(Math.abs(ratio - 0.7) < 0.02, `duel turn rate ~0.7x battle (got ${ratio.toFixed(3)})`);
  R.turn = { duelRad: Math.round(dAng * 1000) / 1000, battleRad: Math.round(bAng * 1000) / 1000, ratio: Math.round(ratio * 1000) / 1000 };
}

// (9) regression: other modes untouched
{
  for (const mode of ["open", "maze", "survival"]) {
    const c = new RicochetCore({ seed: SEED, arenaMode: mode });
    assert.strictEqual(c.reset(SEED).length, 105, `${mode} obs 105`);
    assert.strictEqual(c.getPublicState().scenario, "battle", `${mode} default scenario battle`);
  }
  // a normal survival battle keeps the vanilla turn rate (not slowed)
  const batt = new RicochetCore({ seed: SEED, arenaMode: "survival" }); batt.reset(SEED);
  batt.state.tanks[0].angle = 0; for (let i = 0; i < 5; i++) batt.step({ throttle: 0, turn: 1, fire: false }, "none");
  assert.ok(batt.state.tanks[0].angle > 1.0, "battle turn rate unchanged (fast)");
}

// (10) easy_laika: weaker combat opponent — fires + moves less than laika, but still acts
{
  const measure = (opp) => {
    const c = makeDuel({ maxSteps: 6000 });
    c.state.tanks[0].x = 960; c.state.tanks[0].y = 720; c.state.tanks[0].health = 99;
    c.state.tanks[1].x = 960; c.state.tanks[1].y = 540; c.state.tanks[1].health = 99;
    const r0 = { x: c.state.tanks[1].x, y: c.state.tanks[1].y };
    let fires = 0, moved = 0;
    for (let i = 0; i < 150; i++) { const o = c.step(0, opp); fires += o.events.filter((e) => e.type === "fire").length; const r = c.state.tanks[1]; moved = Math.max(moved, Math.hypot(r.x - r0.x, r.y - r0.y)); }
    return { fires, moved: Math.round(moved) };
  };
  const easy = measure("easy_laika"), hard = measure("laika");
  assert.ok(easy.fires >= 1, "easy_laika fires with LOS (not inert)");
  assert.ok(easy.moved > 10, "easy_laika moves (not a sitting duck)");
  assert.ok(easy.fires < hard.fires, `easy_laika fires less than laika (${easy.fires} < ${hard.fires})`);
  assert.ok(easy.moved < hard.moved, `easy_laika moves less than laika (${easy.moved} < ${hard.moved})`);
  R.easyVsLaika = { easy, laika: hard };
}

// (11) easy_laika prioritises poison escape when caught outside the safe zone
{
  const c = makeDuel();
  c.state.elapsed = 30; c.state.poisonActive = true;
  c.state.safeRect = { x: 880, y: 600, w: 160, h: 80 };
  const bot = c.state.tanks[1]; bot.x = 300; bot.y = 300; bot.health = 3;
  const d0 = Math.hypot(bot.x - 960, bot.y - 640);
  for (let i = 0; i < 30; i++) { c.state.tanks[0].x = 960; c.state.tanks[0].y = 640; c.step(0, "easy_laika"); }
  const r = c.getPublicState().tanks[1];
  assert.ok(Math.hypot(r.x - 960, r.y - 640) < d0, "easy_laika flees toward the safe centre under poison");
  R.easyPoisonEscape = { to: { x: Math.round(r.x), y: Math.round(r.y) } };
}

// (12) fire is effective in moba1v1duel (not inert) + combat stats surface in info
{
  const c = makeDuel(); const b = c.state.tanks[0]; b.x = 960; b.y = 640; b.reload = 0;
  const out = c.step({ throttle: 0, turn: 0, fire: true }, "none");
  assert.ok(c.getPublicState().shells.length >= 1, "blue fire spawns a shell (combat fire active)");
  assert.ok(typeof out.info.hitsDealt === "number" && typeof out.info.learnerHealth === "number", "combat stats present in info");
}

// (13) easy_laika doesn't disturb the other scripted opponents (all stay callable, obs 105)
{
  const c = new RicochetCore({ seed: SEED, arenaMode: "survival" }); c.reset(SEED);
  for (const k of ["stationary", "turret", "laika", "laika-aggressive", "none", "easy_laika"]) {
    c.step(0, k);
    assert.strictEqual(c.buildObservation(0).length, 105, `${k} obs 105`);
  }
}

console.log("MOBA1V1DUEL SMOKE OK");
console.log(JSON.stringify(R, null, 2));
