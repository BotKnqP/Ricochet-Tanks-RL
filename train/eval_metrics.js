"use strict";
// Rich-metrics evaluator: runs an EXPORTED policy (live_policy.json format) vs scripted opponents and
// reports robustness metrics beyond win rate -- the signals the self-play pilot must improve:
//   win, poison_death (learner died to poison: the "camp the old safe zone" failure), time_to_kill,
//   poison_dmg, powerups (self) vs enemy_powerups (powerup-exploitation), hit_per_shot (precision vs
//   volume), self_hit_rate, contact_frac (pressure/stuck style), ep_len.
//   node train/eval_metrics.js --policy runs/auto_live/live_policy.json --opps laika,easy_laika --episodes 60
const path = require("path");
const fs = require("fs");
const { RicochetCore, policyForward } = require(path.join(process.cwd(), "game_core.js"));

function arg(n, d) { const i = process.argv.indexOf("--" + n); return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : d; }
const POLICY = JSON.parse(fs.readFileSync(arg("policy", "runs/auto_live/live_policy.json"), "utf8"));
const OPPS = arg("opps", "stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro").split(",");
const EPISODES = parseInt(arg("episodes", "60"), 10);
const SEEDBASES = arg("seeds", "300000,900000").split(",").map(Number);
const MAX = parseInt(arg("max-steps", "3000"), 10);   // safety cap; poison always ends it well before (v2 fights are longer)
const RULESET = arg("ruleset", "");                    // "" -> survival_v1; "survival_v2" -> long-form combat rules
const SPAWN = arg("spawn", "");                         // override spawnMode (fixed/half_random/full_random)
const HP = arg("hp", "");                               // override tankMaxHp

function evalOpp(opp) {
  const agg = { n: 0, win: 0, poisonDeath: 0, ttkSum: 0, ttkN: 0, poisonDmg: 0, pups: 0, epups: 0,
    hits: 0, shots: 0, selfHits: 0, contact: 0, steps: 0, epEnemyPup: 0, winEnemyPup: 0, trunc: 0 };
  for (const sb of SEEDBASES) for (let e = 0; e < EPISODES; e++) {
    const seed = (sb + e) >>> 0;
    const opts = { seed, arenaMode: "survival", scenario: "moba1v1duel", spawnPowerups: true, shellDecay: true, maxSteps: MAX };
    if (RULESET) opts.ruleset = RULESET;
    if (SPAWN) opts.spawnMode = SPAWN;
    if (HP) opts.tankMaxHp = Number(HP);
    const c = new RicochetCore(opts);
    c.reset(seed);
    let info = null, n = 0, done = false;
    while (n < MAX) { const a = policyForward(c.observe(0), POLICY); const o = c.step(a, opp, { dt: 1 / 30, repeat: 2 }); info = o.info; n++; if (o.done) { done = true; break; } }
    if (!done) agg.trunc++;   // hit the cap without a death-resolution (should be ~0 under survival_v2 max_steps)
    agg.n++;
    if (info.result === "win") { agg.win++; agg.ttkSum += info.elapsed; agg.ttkN++; }
    if (info.loserId === 0 && info.deathCause === "poison") agg.poisonDeath++;
    agg.poisonDmg += info.poisonDamageTaken || 0; agg.pups += info.powerups || 0; agg.epups += info.enemyPowerups || 0;
    agg.hits += info.hitsDealt || 0; agg.shots += info.shotsFired || 0; agg.selfHits += info.selfHits || 0;
    agg.contact += info.contactSteps || 0; agg.steps += n;
    if ((info.enemyPowerups || 0) > 0) { agg.epEnemyPup++; if (info.result === "win") agg.winEnemyPup++; }
  }
  const N = agg.n;
  return {
    win: agg.win / N, poison_death: agg.poisonDeath / N,
    ttk: agg.ttkN ? agg.ttkSum / agg.ttkN : null, poison_dmg: agg.poisonDmg / N,
    pups: agg.pups / N, enemy_pups: agg.epups / N,
    hit_per_shot: agg.shots ? agg.hits / agg.shots : 0, self_hits: agg.selfHits / N,
    contact_frac: agg.steps ? agg.contact / (agg.steps * 2) : 0, ep_len: agg.steps / N,
    // powerup VULNERABILITY (GPT): when enemy got a powerup, did we still win? (null = enemy never got one)
    win_enemy_pup: agg.epEnemyPup ? agg.winEnemyPup / agg.epEnemyPup : null, enemy_pup_eps: agg.epEnemyPup / N,
    trunc_rate: agg.trunc / N,   // fraction of episodes that hit the cap without a death-resolution
  };
}

const out = {};
const hdr = "opponent".padEnd(24) + ["win", "psnDth", "ttk", "psnDmg", "pups", "ePups", "hit/sh", "selfHit", "contact", "epLen"].map(h => h.padStart(8)).join("");
console.log(`metrics (${EPISODES}ep x ${SEEDBASES.length} seeds)`);
console.log(hdr);
for (const opp of OPPS) {
  const m = evalOpp(opp); out[opp] = m;
  const row = [m.win, m.poison_death, m.ttk == null ? -1 : m.ttk, m.poison_dmg, m.pups, m.enemy_pups, m.hit_per_shot, m.self_hits, m.contact_frac, m.ep_len];
  console.log(opp.padEnd(24) + row.map(v => v.toFixed(2).padStart(8)).join(""));
}
const wins = OPPS.map(o => out[o].win);
console.log("-".repeat(hdr.length));
console.log(`MAXIMIN win=${Math.min(...wins).toFixed(2)}  MEAN win=${(wins.reduce((a, b) => a + b, 0) / wins.length).toFixed(2)}  ` +
  `mean poison_death=${(OPPS.reduce((a, o) => a + out[o].poison_death, 0) / OPPS.length).toFixed(2)}  ` +
  `mean hit/shot=${(OPPS.reduce((a, o) => a + out[o].hit_per_shot, 0) / OPPS.length).toFixed(2)}`);
const vuln = OPPS.map(o => out[o].win_enemy_pup).filter(v => v != null);
if (vuln.length) console.log(`powerup-vuln: mean win when enemy got a powerup=${(vuln.reduce((a, b) => a + b, 0) / vuln.length).toFixed(2)} (over ${vuln.length} opps where enemy grabbed one)`);
console.log(`mean pups=${(OPPS.reduce((a, o) => a + out[o].pups, 0) / OPPS.length).toFixed(2)} (powerups the LEARNER picked up — the resource-play signal)  ` +
  `mean trunc_rate=${(OPPS.reduce((a, o) => a + out[o].trunc_rate, 0) / OPPS.length).toFixed(3)} (should be ~0)`);
const dest = arg("out", "");
if (dest) { fs.writeFileSync(dest, JSON.stringify(out, null, 2)); console.log("wrote " + dest); }
