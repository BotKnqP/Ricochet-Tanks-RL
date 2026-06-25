"use strict";
// Batch script-demo generator for behaviour cloning.
//
// Records an EXPERT scripted bot's obs->action demonstrations vs a scripted opponent, in
// the SAME JSONL schema as the browser human recorder (plus an "expert" field). The
// expert's continuous control is discretised through controlToAction() into the exact
// Discrete(18) ids, and `obs` is the state BEFORE the action (s_t -> a_t). Streams to disk
// so 1000s of episodes stay memory-safe. `good_demo` is true only for WON episodes, so a
// `--filter good_wins` BC run learns from the expert's winning play only.
//
//   node gen_script_demos.js --blue laika-aggressive-pro --red easy_laika \
//       --scenario moba1v1duel --episodes 1000 --out data/script_demos/pro_vs_easy_1000.jsonl
const fs = require("fs");
const path = require("path");
const { RicochetCore, controlToAction, actionToControl } = require("./game_core.js");

function arg(name, def) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const BLUE = arg("blue", "laika-aggressive-pro");
const RED = arg("red", "easy_laika");
const SCEN = arg("scenario", "moba1v1duel");
const ARENA = arg("arena", "survival");                       // "open" for the shooting lab
const RANDOM_TURRET = process.argv.includes("--random-turret"); // open arena: random turret spawn
const LENIENT = process.argv.includes("--lenient");             // keep wins despite high self-hits (chaotic duels)
const MOVER_SPEED = arg("mover-speed", "");                    // number, or "rand" (cycle 0.2..1.0 per episode)
const moverSpeedFor = (ep) => MOVER_SPEED === "rand"
  ? [0.2, 0.4, 0.6, 0.8, 1.0][ep % 5]
  : (MOVER_SPEED === "" ? undefined : parseFloat(MOVER_SPEED));
const EPISODES = parseInt(arg("episodes", "1000"), 10);
const SEED = parseInt(arg("seed", "20240618"), 10);
const MAX = parseInt(arg("max-steps", "1800"), 10);
const OUT = arg("out", `data/script_demos/${BLUE}_vs_${RED}.jsonl`);
const REC_DT = 1 / 30, REC_REPEAT = 2;

const outPath = path.isAbsolute(OUT) ? OUT : path.join(__dirname, OUT);
fs.mkdirSync(path.dirname(outPath), { recursive: true });
// Synchronous per-episode writes: the episode loop is synchronous and would starve an
// async stream (buffering ALL output in memory until the end -> GBs / OOM for big runs).
// fs.writeSync flushes each episode to disk immediately, so memory stays bounded.
const fd = fs.openSync(outPath, "w");

const tally = { win: 0, loss: 0, draw: 0, timeout: 0, other: 0 };
let totalTx = 0;

for (let ep = 0; ep < EPISODES; ep++) {
  const c = new RicochetCore({ seed: SEED + ep, arenaMode: ARENA, scenario: SCEN, maxSteps: MAX,
    spawnPowerups: ARENA === "open" ? false : true, randomTurret: RANDOM_TURRET, moverSpeed: moverSpeedFor(ep),
    shellDecay: true });   // CANONICAL physics, explicit (matches game_core default + TankEnv)
  c.reset(SEED + ep);
  let step = 0, totalReward = 0, info = null, lines = "";
  while (!c.isDone()) {
    const obsBefore = c.observe(0);                               // s_t (BEFORE the action)
    const bc = c.scriptedControl(0, BLUE);                        // the expert's control
    const action = controlToAction(bc.throttle, bc.turn, bc.fire);
    const control = actionToControl(action);                     // canonical discretised == ACTION_TABLE[action]
    const o = c.step(action, RED, { dt: REC_DT, repeat: REC_REPEAT });
    info = o.info;
    lines += JSON.stringify({
      type: "transition", episode: ep, step, scenario: SCEN, opponent: RED, expert: BLUE,
      obs: obsBefore.map((v) => Math.round(v * 1e6) / 1e6),
      action, control,
      reward: Math.round(o.reward * 1e6) / 1e6,
      done: Boolean(o.done), result: o.info.result,
      elapsed: Math.round(o.info.elapsed * 1000) / 1000,
    }) + "\n";
    step += 1;
    totalReward += o.reward;
    totalTx += 1;
  }
  const result = (info && info.result) || c.result();
  const hitsDealt = (info && info.hitsDealt) || 0;
  const selfHits = (info && info.selfHits) || 0;
  // quality gate: a "good" demo is a clean win — actually fought (>=2 hits) and not
  // sloppy (self-hits no more than hits dealt). --filter good_wins reads only these.
  // --lenient keeps ANY win: vs reckless laika-aggressive the expert wins ~0.89 by landing ONE
  // decisive hit and letting aggro finish itself off via self-hits (all such wins have hitsDealt=1
  // and high selfHits), so BOTH strict sub-gates (hitsDealt>=2, selfHits<=hitsDealt) rejected them.
  const goodDemo = LENIENT
    ? (result === "win")
    : (result === "win" && hitsDealt >= 2 && selfHits <= hitsDealt);
  tally[result in tally ? result : "other"] += 1;
  lines += JSON.stringify({
    type: "episode_summary", episode: ep, scenario: SCEN, opponent: RED, expert: BLUE,
    result, elapsed: Math.round(c.getPublicState().elapsed * 1000) / 1000,
    totalReward: Math.round(totalReward * 1e6) / 1e6, steps: step,
    hits_dealt: hitsDealt, self_hits: selfHits,
    good_demo: goodDemo,
  }) + "\n";
  fs.writeSync(fd, lines);                                        // flush this episode to disk
  if ((ep + 1) % 100 === 0) console.error(`  ...${ep + 1}/${EPISODES} episodes (${totalTx} transitions)`);
}

fs.closeSync(fd);
console.log(`wrote ${outPath}`);
console.log(JSON.stringify({ blue: BLUE, red: RED, scenario: SCEN, episodes: EPISODES, transitions: totalTx, outcomes: tally }, null, 2));
