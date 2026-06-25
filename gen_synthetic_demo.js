"use strict";
// Synthetic demonstration generator for the BC pipeline.
//
// Produces JSONL in the SAME schema as the browser human recorder
// (human_demo_moba1v1duel_easy_laika_*.jsonl), but the "demonstrator" is a scripted bot
// (default: laika) whose continuous control is DISCRETISED through controlToAction() into
// the exact Discrete(18) action ids — identical to how a human's WASD+Space are recorded.
// Useful for (a) smoke-testing train_bc.py / evaluate_bc.py without manual recording, and
// (b) bootstrapping a BC warm-start before you sit down to record real human demos.
//
//   node gen_synthetic_demo.js --episodes 8 --out data/human_demos/synthetic_laika.jsonl --blue laika
//
// Args: --episodes N  --out PATH  --blue <laika|laika-aggressive|...>  --seed S  --max-steps M
const fs = require("fs");
const path = require("path");
const { RicochetCore, controlToAction, actionToControl } = require("./game_core.js");

function arg(name, def) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const EPISODES = parseInt(arg("episodes", "8"), 10);
const OUT = arg("out", "data/human_demos/synthetic_laika.jsonl");
const BLUE = arg("blue", "laika");           // the scripted "demonstrator" controlling blue
const SEED = parseInt(arg("seed", "1307"), 10);
const MAX_STEPS = parseInt(arg("max-steps", "1800"), 10);
const REC_DT = 1 / 30, REC_REPEAT = 2;       // match training / the human recorder cadence

const outPath = path.isAbsolute(OUT) ? OUT : path.join(__dirname, OUT);
fs.mkdirSync(path.dirname(outPath), { recursive: true });

const lines = [];
const tally = { win: 0, loss: 0, draw: 0, timeout: 0, other: 0 };
let totalTransitions = 0;

for (let ep = 0; ep < EPISODES; ep++) {
  const core = new RicochetCore({
    seed: SEED + ep, arenaMode: "survival", scenario: "moba1v1duel",
    maxSteps: MAX_STEPS, spawnPowerups: true,
  });
  core.reset(SEED + ep);
  let step = 0, totalReward = 0, info = null;
  while (!core.isDone()) {
    const obsBefore = core.observe(0);                              // s_t BEFORE the action
    const raw = core.scriptedControl(0, BLUE);                      // continuous scripted control
    const action = controlToAction(raw.throttle, raw.turn, raw.fire);
    const control = actionToControl(action);                       // canonical discretised control == ACTION_TABLE[action]
    const keys = { w: control.throttle > 0, s: control.throttle < 0, a: control.turn < 0, d: control.turn > 0, space: !!control.fire };
    const out = core.step(action, "easy_laika", { dt: REC_DT, repeat: REC_REPEAT });
    info = out.info;
    // obs rounded to 6 decimals and `control` stored as the discretised action's control
    // (== ACTION_TABLE[action]) — IDENTICAL to the browser human recorder (game_render.js
    // recordStep), so train_bc.py reads synthetic and human demos through one schema.
    lines.push(JSON.stringify({
      type: "transition", episode: ep, step,
      scenario: "moba1v1duel", opponent: "easy_laika",
      obs: obsBefore.map((v) => Math.round(v * 1e6) / 1e6),
      action, control, keys,
      reward: Math.round(out.reward * 1e6) / 1e6,
      done: Boolean(out.done), result: out.info.result,
      elapsed: Math.round(out.info.elapsed * 1000) / 1000,
    }));
    step += 1;
    totalReward += out.reward;
    totalTransitions += 1;
  }
  const result = (info && info.result) || core.result();
  tally[result in tally ? result : "other"] += 1;
  lines.push(JSON.stringify({
    type: "episode_summary", episode: ep,
    scenario: "moba1v1duel", opponent: "easy_laika",
    result, elapsed: Math.round(core.getPublicState().elapsed * 1000) / 1000,
    totalReward: Math.round(totalReward * 1e6) / 1e6, steps: step, good_demo: true,
  }));
}

fs.writeFileSync(outPath, lines.join("\n") + "\n");
console.log(`wrote ${outPath}`);
console.log(JSON.stringify({ episodes: EPISODES, transitions: totalTransitions, outcomes: tally, blue: BLUE }, null, 2));
