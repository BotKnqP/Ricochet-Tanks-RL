"use strict";
// One parallel worker of the fixed-spawn ELO ladder. Plays the ordered (blue,red) pairs assigned to this
// slice (pairIndex % nslices === slice) for G seeded games each and writes runs/ladder/slice_<k>.json with
// per-cell [blueWins, draws, games]. A model acts via policyForward(observe(seat)); the RED model uses
// observe(1) raw -- exactly how this project's self-play fed policy opponents (rl_bridge_selfplay.js:60),
// and exactly what happens in play.html, so the ladder is faithful to deployment.
//   node _ladder_worker.js --slice 0 --nslices 12 --games 60
const path = require("path");
const fs = require("fs");
const ROOT = __dirname;
const { RicochetCore, policyForward } = require(path.join(ROOT, "game_core.js"));
global.window = global.window || {};
require(path.join(ROOT, "ladder_weights.js"));
const POL = global.window.RICOCHET_POLICIES;

function arg(n, d) { const i = process.argv.indexOf("--" + n); return i >= 0 ? process.argv[i + 1] : d; }
const SLICE = parseInt(arg("slice", "0"), 10);
const NSLICES = parseInt(arg("nslices", "1"), 10);
const G = parseInt(arg("games", "60"), 10);
const MAX = parseInt(arg("max-steps", "3000"), 10);
const SEEDBASE = parseInt(arg("seed", "424242"), 10);
const OUTDIR = path.join(ROOT, "runs/ladder");
fs.mkdirSync(OUTDIR, { recursive: true });

const MODELS = [
  ["fixed_bc", "S1 裸BC"], ["fixed_dagger", "S2 DAgger"], ["fixed_league", "S3 联赛PPO"],
  ["fixed_dr_league", "S4 域随机联赛"], ["v3champ", "固定出生点冠军"], ["v15", "v1.5随机出生体"],
];
const SCRIPTS = ["stationary", "easy_laika", "laika", "laika-aggressive", "laika-aggressive-pro"];
const PLAYERS = [];
for (const [name, label] of MODELS) { if (!POL[name]) { console.error("MISSING policy:", name); process.exit(1); } PLAYERS.push({ name, label, kind: "model", pol: POL[name] }); }
for (const s of SCRIPTS) PLAYERS.push({ name: s, label: s, kind: "script" });
const N = PLAYERS.length;

function actionFor(p, c, seat) { return p.kind === "script" ? p.name : policyForward(c.observe(seat), p.pol); }
function playGame(blue, red, seed) {
  const c = new RicochetCore({ seed, arenaMode: "survival", scenario: "moba1v1duel", spawnPowerups: true,
    shellDecay: true, ruleset: "survival_v1", spawnMode: "fixed", maxSteps: MAX });
  c.reset(seed);
  let info = null, k = 0;
  while (k < MAX) {
    const o = c.step(actionFor(blue, c, 0), actionFor(red, c, 1), { dt: 1 / 30, repeat: 2 });
    info = o.info; k++;
    if (o.done) break;
  }
  if (info && info.loserId === 1) return "blue";
  if (info && info.loserId === 0) return "red";
  return "draw";
}

const cells = {};
const t0 = Date.now();
let pairIdx = 0, played = 0;
for (let i = 0; i < N; i++) {
  for (let j = 0; j < N; j++) {
    if (i === j) continue;
    const mine = (pairIdx % NSLICES) === SLICE;
    pairIdx++;
    if (!mine) continue;
    let bw = 0, dr = 0;
    for (let g = 0; g < G; g++) {
      const seed = (SEEDBASE + (i * 131 + j) * 100003 + g * 97) >>> 0;
      const r = playGame(PLAYERS[i], PLAYERS[j], seed);
      if (r === "blue") bw++; else if (r === "draw") dr++;
      played++;
    }
    cells[`${i},${j}`] = [bw, dr, G];
  }
}
const secs = (Date.now() - t0) / 1000;
fs.writeFileSync(path.join(OUTDIR, `slice_${SLICE}.json`), JSON.stringify({
  slice: SLICE, nslices: NSLICES, games_per_cell: G, played, secs: Math.round(secs),
  roster: PLAYERS.map((p) => ({ name: p.name, label: p.label, kind: p.kind })), cells,
}));
console.error(`slice ${SLICE}/${NSLICES}: ${played} games in ${secs.toFixed(0)}s`);
