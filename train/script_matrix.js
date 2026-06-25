"use strict";
// FAST in-process script-vs-script win matrix. One node process, pure RicochetCore — no
// per-episode subprocess spawning (eval_script_bot.js is ~9s/episode via IPC; this is ~0.05ms/step).
// blue = discretised expert action (matches the demo/eval label path); red = scripted opponent.
//
//   node train/script_matrix.js --episodes 60 --max-steps 900 \
//     --blues laika,laika-aggressive,laika-aggressive-pro,easy_laika \
//     --reds stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro

const path = require("path");
const { RicochetCore, controlToAction } = require(path.join(__dirname, "..", "game_core.js"));

function arg(name, def) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
const BLUES = arg("blues", "laika,laika-aggressive,laika-aggressive-pro").split(",");
const REDS = arg("reds", "stationary,easy_laika,laika,laika-aggressive,laika-aggressive-pro").split(",");
const EPISODES = parseInt(arg("episodes", "60"), 10);
const SEED = parseInt(arg("seed", "300000"), 10);
const MAX = parseInt(arg("max-steps", "900"), 10);
const SCEN = arg("scenario", "moba1v1duel");
const ARENA = arg("arena", "survival");

// Uses core.advance (physics only, NO obs raycasts — step() builds obs0+obs1 every call, ~20x
// slower and useless here). Manual step-count + maxSteps timeout, since advance doesn't truncate.
// Controls computed once per decision and held across actionRepeat updates, matching step().
function play(blue, red) {
  let win = 0, loss = 0, draw = 0, timeout = 0;
  for (let ep = 0; ep < EPISODES; ep++) {
    const c = new RicochetCore({ seed: SEED + ep, arenaMode: ARENA, scenario: SCEN, maxSteps: MAX,
      spawnPowerups: true, shellDecay: true });
    c.reset(SEED + ep);
    let steps = 0, done = false, result = null;
    while (steps < MAX && !done) {
      const bc = c.scriptedControl(0, blue);
      const a = controlToAction(bc.throttle, bc.turn, bc.fire);
      const rc = c.scriptedControl(1, red);
      for (let k = 0; k < 2; k++) {
        const out = c.advance(1 / 30, a, rc);
        if (out.done) { done = true; result = out.result; break; }
      }
      steps++;
    }
    if (!done) result = "timeout";
    if (result === "win") win++;
    else if (result === "loss") loss++;
    else if (result === "timeout") timeout++;
    else draw++;
  }
  return { win: win / EPISODES, loss: loss / EPISODES, draw: draw / EPISODES,
           timeout: timeout / EPISODES, hd: 0 };
}

console.log(`WIN MATRIX (blue beats red) ${EPISODES}ep seed=${SEED} ${SCEN}/${ARENA} maxSteps=${MAX}`);
const pad = (s, n) => String(s).padEnd(n);
const padl = (s, n) => String(s).padStart(n);
console.log(pad("blue\\red", 24) + REDS.map((r) => padl(r.slice(0, 13), 14)).join(""));
const best = {};
const cells = {};
for (const blue of BLUES) {
  const row = [];
  for (const red of REDS) {
    const r = play(blue, red);
    cells[`${blue}|${red}`] = r;
    row.push(r.win);
    if (!(red in best) || r.win > best[red].win) best[red] = { win: r.win, blue };
    console.error(`  [cell] ${blue} vs ${red}: win=${r.win.toFixed(2)} loss=${r.loss.toFixed(2)} to=${r.timeout.toFixed(2)}`);
  }
  console.log(pad(blue, 24) + row.map((w) => padl(w.toFixed(2), 14)).join(""));
}
console.log("-".repeat(24 + REDS.length * 14));
console.log("BEST expert per opponent (DAgger expert-map):");
for (const red of REDS) console.log(`  ${pad(red, 24)} -> ${pad(best[red].blue, 22)} (win ${best[red].win.toFixed(2)})`);
console.log("\nself-hit / timeout detail (blue vs red):");
for (const blue of BLUES) for (const red of REDS) {
  const r = cells[`${blue}|${red}`];
  console.log(`  ${pad(blue + " vs " + red, 40)} win=${r.win.toFixed(2)} loss=${r.loss.toFixed(2)} to=${r.timeout.toFixed(2)} hd=${r.hd.toFixed(1)}`);
}
