"use strict";
// Moving-target shooting gallery (Experiment 1): isolate "can the shooter LEAD a moving
// target" from full-moba noise. blue = a shooter (script via --blue), red = `mover` strafing
// vertically at a swept speed, never firing. Open arena, no powerups/poison. We measure how
// hit performance falls off as the target's transverse velocity rises.
//
//   node eval_shooting_gallery.js --blue laika-aggressive --episodes 30
const { RicochetCore, controlToAction } = require("./game_core.js");

function arg(name, def) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const BLUE = arg("blue", "laika-aggressive");
const EPISODES = parseInt(arg("episodes", "30"), 10);
const MAX = parseInt(arg("max-steps", "600"), 10);
const SPEEDS = (arg("speeds", "0,0.15,0.3,0.5,0.7,1.0")).split(",").map(Number);
const DT = 1 / 30, REPEAT = 2;
const FIRE = new Set([2, 4, 6, 8, 9, 11, 13, 15, 17]);   // Discrete(18) ids with fire=true
const r2 = (x) => Math.round(x * 100) / 100;

function run(speed) {
  let shots = 0, hits = 0, self = 0, kills = 0, vsum = 0, vcount = 0, hpsum = 0, steps = 0;
  for (let ep = 0; ep < EPISODES; ep++) {
    const c = new RicochetCore({ seed: 7000 + ep, arenaMode: "open", scenario: "battle",
      maxSteps: MAX, spawnPowerups: false, moverSpeed: speed });
    c.reset(7000 + ep);
    let info = null, prevY = null;
    while (!c.isDone()) {
      const ctrl = c.scriptedControl(0, BLUE);
      const a = controlToAction(ctrl.throttle, ctrl.turn, ctrl.fire);
      if (FIRE.has(a)) shots++;
      const o = c.step(a, "mover", { dt: DT, repeat: REPEAT });
      info = o.info; steps++;
      const ry = c.getPublicState().tanks[1].y;
      if (prevY !== null) { vsum += Math.abs(ry - prevY); vcount++; }
      prevY = ry;
    }
    hits += info.hitsDealt || 0;
    self += info.selfHits || 0;
    hpsum += info.opponentHealth || 0;
    if ((info.opponentHealth || 0) <= 0.01) kills++;     // mover has ~3 HP; killed = led well enough 3x
  }
  const transVel = (vsum / Math.max(1, vcount)) / (REPEAT * DT);   // px/s the target actually strafed
  return {
    moverSpeed: speed,
    target_transverse_vel: Math.round(transVel),
    shots_per_ep: r2(shots / EPISODES),
    hits_dealt: r2(hits / EPISODES),
    hit_per_shot: r2(hits / Math.max(1, shots)),
    self_hits: r2(self / EPISODES),
    kill_rate: r2(kills / EPISODES),
    target_final_hp: r2(hpsum / EPISODES),
  };
}

console.log(`# shooting gallery: blue=${BLUE}  episodes=${EPISODES}  max-steps=${MAX}`);
console.log("# speed  transV(px/s)  shots  hits  hit/shot  self  kill%  target_hp");
for (const s of SPEEDS) {
  const r = run(s);
  console.log("%s  %s  %s  %s  %s  %s  %s  %s",
    String(r.moverSpeed).padEnd(5), String(r.target_transverse_vel).padEnd(11),
    String(r.shots_per_ep).padEnd(5), String(r.hits_dealt).padEnd(4),
    String(r.hit_per_shot).padEnd(8), String(r.self_hits).padEnd(4),
    String(r.kill_rate).padEnd(5), r.target_final_hp);
}
