"use strict";
// laika-aggressive-pro expert smoke tests. Run: node smoke_script_bot.js
const assert = require("assert");
const { RicochetCore, constants, OBS_SIZE, ACTION_TABLE, controlToAction, actionToControl } = require("./game_core.js");

const SEED = 1307;
const R = {};

// (9,10) OBS / ACTION unchanged
assert.strictEqual(OBS_SIZE, 105, "OBS_SIZE 105");
assert.strictEqual(ACTION_TABLE.length, 18, "ACTION_TABLE 18");

// (1) pro is callable and yields a valid control -> valid Discrete(18) id
{
  const c = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "moba1v1duel", maxSteps: 600, spawnPowerups: true });
  c.reset(SEED);
  for (let i = 0; i < 40; i++) {
    const ctrl = c.scriptedControl(0, "laika-aggressive-pro");
    assert.ok(ctrl && typeof ctrl.throttle === "number" && typeof ctrl.turn === "number" && typeof ctrl.fire === "boolean", "control shape");
    const a = controlToAction(ctrl.throttle, ctrl.turn, ctrl.fire);
    assert.ok(Number.isInteger(a) && a >= 0 && a < ACTION_TABLE.length, "valid action id");
    c.step(a, "easy_laika", { dt: 1 / 30, repeat: 2 });
    if (c.isDone()) break;
  }
  assert.strictEqual(c.buildObservation(0).length, 105, "obs 105");
}

// (2) "pro" alias resolves to the same controller
{
  const c = new RicochetCore({ seed: SEED, arenaMode: "survival", scenario: "moba1v1duel", maxSteps: 100, spawnPowerups: true });
  c.reset(SEED);
  const a = c.scriptedControl(0, "laika-aggressive-pro"), b = c.scriptedControl(0, "pro");
  assert.deepStrictEqual(a, b, "'pro' == 'laika-aggressive-pro'");
}

// (3) pro is strong vs easy_laika and self-disciplined (a few episodes)
{
  const run = (red, eps) => {
    let wins = 0, hd = 0, sh = 0, fire = 0, total = 0;
    for (let ep = 0; ep < eps; ep++) {
      const c = new RicochetCore({ seed: SEED + ep, arenaMode: "survival", scenario: "moba1v1duel", maxSteps: 1800, spawnPowerups: true });
      c.reset(SEED + ep);
      let info = null;
      while (!c.isDone()) {
        const ctrl = c.scriptedControl(0, "laika-aggressive-pro");
        const a = controlToAction(ctrl.throttle, ctrl.turn, ctrl.fire);
        const o = c.step(a, red, { dt: 1 / 30, repeat: 2 });
        info = o.info; total += 1; if (actionToControl(a).fire) fire += 1;
      }
      if (info.result === "win") wins += 1;
      hd += info.hitsDealt || 0; sh += info.selfHits || 0;
    }
    return { win: wins / eps, hitsDealt: hd / eps, selfHits: sh / eps, firePct: 100 * fire / Math.max(1, total) };
  };
  const easy = run("easy_laika", 6);
  assert.ok(easy.win >= 0.8, `pro beats easy_laika (win ${easy.win})`);
  assert.ok(easy.hitsDealt > easy.selfHits * 2, `hits_dealt(${easy.hitsDealt.toFixed(2)}) > 2x self_hits(${easy.selfHits.toFixed(2)})`);
  assert.ok(easy.firePct > 3, `fire% (${easy.firePct.toFixed(1)}) > 3`);
  R.vsEasy = { win: easy.win, hitsDealt: Math.round(easy.hitsDealt * 100) / 100, selfHits: Math.round(easy.selfHits * 100) / 100, firePct: Math.round(easy.firePct * 10) / 10 };
}

// (4) other scripted opponents still work (regression)
{
  const c = new RicochetCore({ seed: SEED, arenaMode: "survival" }); c.reset(SEED);
  for (const k of ["laika", "laika-aggressive", "easy_laika", "stationary", "turret", "none", "laika-aggressive-pro"]) {
    c.step(0, k);
    assert.strictEqual(c.buildObservation(0).length, 105, `${k} obs 105`);
  }
}

// (5) shooting-lab: pro vs RANDOM-position turret on the OPEN map -> clean kills, varied geometry
{
  const xs = new Set(), ys = new Set(), lens = new Set();
  let wins = 0, hd = 0, sh = 0, fire = 0, total = 0; const eps = 12;
  for (let ep = 0; ep < eps; ep++) {
    const c = new RicochetCore({ seed: 4100 + ep, arenaMode: "open", scenario: "battle", maxSteps: 600, randomTurret: true });
    c.reset(4100 + ep);
    const t = c.getPublicState().tanks[1];                     // turret spawn position
    xs.add(Math.round(t.x)); ys.add(Math.round(t.y));
    assert.strictEqual(c.buildObservation(0).length, 105, "obs 105 (open)");
    let info = null, steps = 0;
    while (!c.isDone()) {
      const ctrl = c.scriptedControl(0, "laika-aggressive-pro");
      const a = controlToAction(ctrl.throttle, ctrl.turn, ctrl.fire);
      const o = c.step(a, "turret", { dt: 1 / 30, repeat: 2 });
      info = o.info; total += 1; steps += 1; if (actionToControl(a).fire) fire += 1;
    }
    lens.add(steps);
    if (info.result === "win") wins += 1;
    hd += info.hitsDealt || 0; sh += info.selfHits || 0;
  }
  assert.ok(xs.size >= eps - 1 && ys.size >= eps - 1, `turret pos varies (x:${xs.size} y:${ys.size} of ${eps})`);
  assert.ok(lens.size >= eps / 2, `episode lengths vary (${lens.size} distinct of ${eps}) -> aim-acquisition present`);
  assert.ok(wins / eps >= 0.8, `pro beats random turret (win ${wins / eps})`);
  assert.ok(hd / eps > sh / eps, `hits_dealt(${(hd / eps).toFixed(2)}) > self_hits(${(sh / eps).toFixed(2)})`);
  R.shootingLab = { win: wins / eps, hitsDealt: Math.round(hd / eps * 100) / 100, selfHits: Math.round(sh / eps * 100) / 100, firePct: Math.round(1000 * fire / Math.max(1, total)) / 10, posVariety: { x: xs.size, y: ys.size }, lenVariety: lens.size };
}

console.log("SCRIPT_BOT SMOKE OK");
console.log(JSON.stringify(R, null, 2));
