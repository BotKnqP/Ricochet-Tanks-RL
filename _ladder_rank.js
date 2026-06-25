"use strict";
// Aggregate the worker slices and fit STABLE ratings: the Bradley-Terry maximum-likelihood fixed point
// (the rating an Elo ranked-ladder converges to once it stops moving), reported on the Elo scale
// (Elo_i = 1000 + 400*log10(gamma_i), gamma geo-mean pinned to 1). Seats are balanced upstream so the
// blue(left) first-mover edge cancels; BT uses each pair's total games/wins. Writes leaderboard.md + results.json.
//   node _ladder_rank.js
const path = require("path");
const fs = require("fs");
const ROOT = __dirname;
const OUTDIR = path.join(ROOT, "runs/ladder");

const slices = fs.readdirSync(OUTDIR).filter((f) => /^slice_\d+\.json$/.test(f))
  .map((f) => JSON.parse(fs.readFileSync(path.join(OUTDIR, f), "utf8")));
if (!slices.length) { console.error("no slice_*.json in", OUTDIR); process.exit(1); }
const roster = slices[0].roster;
const G = slices[0].games_per_cell;
const N = roster.length;

// assemble full matrices: bw[i][j]=blue(i) wins vs red(j), dr draws, bn games
const bw = Array.from({ length: N }, () => new Array(N).fill(0));
const dr = Array.from({ length: N }, () => new Array(N).fill(0));
const bn = Array.from({ length: N }, () => new Array(N).fill(0));
let totalGames = 0;
for (const sl of slices) for (const key in sl.cells) {
  const [i, j] = key.split(",").map(Number); const [w, d, n] = sl.cells[key];
  bw[i][j] += w; dr[i][j] += d; bn[i][j] += n; totalGames += n;
}
// sanity: every off-diagonal cell present
let missing = 0;
for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j && bn[i][j] === 0) missing++;
if (missing) console.error(`WARNING: ${missing} empty cells (incomplete slices)`);

// per-pair total games n_ij and total score W_i (wins + 0.5 draws across BOTH seats)
const nij = Array.from({ length: N }, () => new Array(N).fill(0));
const W = new Array(N).fill(0);
for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
  if (i === j) continue;
  nij[i][j] = bn[i][j] + bn[j][i];
  // i's score vs j: as blue (bw+0.5dr) + as red (red wins = bn[j][i]-bw[j][i]-dr[j][i], +0.5 dr[j][i])
  W[i] += bw[i][j] + 0.5 * dr[i][j] + (bn[j][i] - bw[j][i] - dr[j][i]) + 0.5 * dr[j][i];
}

// Bradley-Terry MM iteration to the MLE fixed point
let g = new Array(N).fill(1);
let iters = 0, delta = Infinity;
for (; iters < 100000 && delta > 1e-12; iters++) {
  const ng = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    let denom = 0;
    for (let j = 0; j < N; j++) { if (i === j || nij[i][j] === 0) continue; denom += nij[i][j] / (g[i] + g[j]); }
    ng[i] = denom > 0 ? Math.max(W[i], 1e-9) / denom : g[i];   // tiny floor so a 0-win player stays finite
  }
  const logmean = ng.reduce((a, x) => a + Math.log(x), 0) / N;  // pin geometric mean to 1
  const norm = Math.exp(-logmean);
  for (let i = 0; i < N; i++) ng[i] *= norm;
  delta = Math.max(...ng.map((x, i) => Math.abs(Math.log(x) - Math.log(g[i]))));
  g = ng;
}
const elo = g.map((x) => 1000 + 400 * Math.log10(x));

// aggregates
const LAIKA = new Set(["stationary", "easy_laika", "laika", "laika-aggressive", "laika-aggressive-pro"]);
const rows = roster.map((p, i) => {
  let bwS = 0, bnS = 0, rwS = 0, rnS = 0, lw = 0, ln = 0;
  for (let j = 0; j < N; j++) {
    if (j === i) continue;
    bwS += bw[i][j] + 0.5 * dr[i][j]; bnS += bn[i][j];
    rwS += (bn[j][i] - bw[j][i] - dr[j][i]) + 0.5 * dr[j][i]; rnS += bn[j][i];
    if (LAIKA.has(roster[j].name)) {
      lw += bw[i][j] + 0.5 * dr[i][j] + (bn[j][i] - bw[j][i] - dr[j][i]) + 0.5 * dr[j][i];
      ln += bn[i][j] + bn[j][i];
    }
  }
  return { i, name: p.name, label: p.label, kind: p.kind, elo: elo[i],
    games: bnS + rnS, winrate: (bwS + rwS) / (bnS + rnS),
    blue_wr: bwS / bnS, red_wr: rwS / rnS, vs_laika: ln ? lw / ln : 0 };
});
rows.sort((a, b) => b.elo - a.elo);

// ---- console ----
console.log(`\n=== FIXED-spawn ELO ladder — STABLE (Bradley-Terry MLE) ===`);
console.log(`${N} players · ${G} games/cell · ${totalGames} games · BT converged in ${iters} iters (delta<1e-12)\n`);
console.log("rank  player                 type    ELO   games  win%  blue% red%  vsLaika%");
rows.forEach((r, k) => {
  console.log(`${String(k + 1).padStart(2)}.  ${r.name.padEnd(21)} ${r.kind.padEnd(6)} ${r.elo.toFixed(0).padStart(5)}  ${String(r.games).padStart(5)} ${(100 * r.winrate).toFixed(1).padStart(5)} ${(100 * r.blue_wr).toFixed(0).padStart(4)} ${(100 * r.red_wr).toFixed(0).padStart(4)}  ${(100 * r.vs_laika).toFixed(1).padStart(6)}`);
});
console.log("\n--- head-to-head: row(blue) win% vs col(red) ---");
console.log("".padEnd(21) + roster.map((p) => p.name.slice(0, 6).padStart(7)).join(""));
for (let i = 0; i < N; i++) {
  let line = roster[i].name.padEnd(21);
  for (let j = 0; j < N; j++) line += (i === j ? "—" : (100 * bw[i][j] / bn[i][j]).toFixed(0)).padStart(7);
  console.log(line);
}

// ---- artifacts ----
fs.writeFileSync(path.join(OUTDIR, "results.json"), JSON.stringify({
  meta: { players: roster, games_per_cell: G, total_games: totalGames, spawn: "fixed", ruleset: "survival_v1",
    method: "Bradley-Terry MLE -> Elo scale", bt_iters: iters },
  ladder: rows.map((r, k) => ({ rank: k + 1, name: r.name, kind: r.kind, elo: Math.round(r.elo),
    games: r.games, winrate: +r.winrate.toFixed(4), blue_wr: +r.blue_wr.toFixed(4),
    red_wr: +r.red_wr.toFixed(4), vs_laika: +r.vs_laika.toFixed(4) })),
  blue_wins: bw, draws: dr, blue_n: bn,
}, null, 2));

const md = [];
md.push(`# 固定出生点 ELO 排位赛排行榜\n`);
md.push(`> 规则 survival_v1 · 固定出生点 · obs 105 · **${N} 名选手** · 每格 ${G} 局(双向各 ${G},座位均衡抵消蓝方先手)· 共 **${totalGames}** 局`);
md.push(`> 评分 = Bradley–Terry 极大似然不动点(排位赛收敛后的稳定 Elo),起点 1000,几何均值锚定 · BT 迭代 ${iters} 步收敛\n`);
md.push(`| 名次 | 选手 | 类型 | **ELO** | 总场 | 胜率 | 蓝方胜率 | 红方胜率 | 对laika家族 |`);
md.push(`|---|---|---|---|---|---|---|---|---|`);
rows.forEach((r, k) => md.push(`| ${k + 1} | \`${r.name}\` | ${r.kind === "model" ? "模型" : "脚本"} | **${r.elo.toFixed(0)}** | ${r.games} | ${(100 * r.winrate).toFixed(1)}% | ${(100 * r.blue_wr).toFixed(0)}% | ${(100 * r.red_wr).toFixed(0)}% | ${(100 * r.vs_laika).toFixed(1)}% |`));
md.push(`\n**说明**:座位均衡后蓝方(左/先手)优势已抵消;模型只在蓝方训练过,红方用 \`observe(1)\` 原始观测(与 play.html 部署一致),故其红方胜率偏低属真实离群弱点。\n`);
fs.writeFileSync(path.join(OUTDIR, "leaderboard.md"), md.join("\n") + "\n");
console.log(`\nwrote ${path.join(OUTDIR, "leaderboard.md")} and results.json`);
