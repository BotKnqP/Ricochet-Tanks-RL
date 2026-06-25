"use strict";
// Inference-time OPPONENT ROUTER. The multi-task conflict (no single identity-blind policy beats BOTH evasive
// laika AND reckless laika-aggressive) is attacked by ROUTING: each step, estimate the enemy's aggression from
// the PUBLIC game state (closing-rate + close-contact over a few frames -- OUTSIDE the fixed 101-dim obs) and
// run the reckless-SPECIALIST when the enemy is charging, else the GENERALIST. The opponent's behaviour is fixed
// per episode, so the EMA estimate converges in ~1-2s; before that, default to the generalist.
//   node train/eval_router.js --specialist spec.json --generalist gen.json --ruleset survival_v2 \
//        --opps laika,laika-aggressive,charger,... --episodes 30 --aggr-thresh 0.5 --warmup 20
const path = require("path"); const fs = require("fs");
const { RicochetCore, policyForward, controlToAction } = require(path.join(process.cwd(), "game_core.js"));
function arg(n, d) { const i = process.argv.indexOf("--" + n); return i >= 0 ? process.argv[i + 1] : d; }
const SPEC_SCRIPT = arg("spec-script", "");           // if set, the RECKLESS route uses this SCRIPT (e.g. laika-aggressive) instead of a policy
const SPEC = SPEC_SCRIPT ? null : JSON.parse(fs.readFileSync(arg("specialist"), "utf8"));
const GEN_SCRIPT = arg("gen-script", "");             // if set, the EVASIVE route uses this script instead of a policy
const GEN = GEN_SCRIPT ? null : JSON.parse(fs.readFileSync(arg("generalist"), "utf8"));
const OPPS = arg("opps", "laika,laika-aggressive,charger,easy_laika,stationary,laika-aggressive-pro,randomized").split(",");
const EPISODES = parseInt(arg("episodes", "30"), 10);
const SEEDBASES = arg("seeds", "300000,900000").split(",").map(Number);
const RULESET = arg("ruleset", "");
const SPAWN = arg("spawn", "");
const MAX = parseInt(arg("max-steps", "3000"), 10);
// Classifier: CLASSIFY-THEN-LOCK to avoid the feedback loop (routing to the aggressive specialist closes distance,
// which would keep the router stuck on spec). During a WARMUP window the GENERALIST drives blue (consistent
// behaviour), and we accumulate the enemy's APPROACH SPEED (its own velocity projected toward blue -- blue-position
// -robust) + close-contact. At WARMUP end we classify reckless-vs-evasive and LOCK the expert for the rest of the
// episode (the opponent's behaviour is fixed per episode, so early classification is enough).
const APPROACH_THRESH = parseFloat(arg("approach", "8"));  // mean enemy approach speed (px/step) above this -> reckless
const CLOSE_FRAC = parseFloat(arg("close-frac", "0.30"));  // OR warmup close-fraction above this -> reckless
const CLOSE = parseFloat(arg("close", "220"));             // "close" range (px)
const WARMUP = parseInt(arg("warmup", "45"), 10);          // steps to observe (generalist) before classify+lock

function evalOpp(opp) {
  let wins = 0, n = 0, specSteps = 0, totSteps = 0, correctRoute = 0, ttkSum = 0, ttkN = 0;
  const reckless = (opp === "laika-aggressive" || opp === "charger");   // ground truth for routing accuracy
  for (const sb of SEEDBASES) for (let e = 0; e < EPISODES; e++) {
    const seed = (sb + e) >>> 0;
    const opts = { seed, arenaMode: "survival", scenario: "moba1v1duel", spawnPowerups: true, shellDecay: true, maxSteps: MAX };
    if (RULESET) opts.ruleset = RULESET;
    if (SPAWN) opts.spawnMode = SPAWN;
    const c = new RicochetCore(opts); c.reset(seed);
    let info = null, k = 0, closeCount = 0, approachSum = 0, approachN = 0, locked = false, useSpec = false, finalSpec = false, prevEn = null;
    while (k < MAX) {
      const st = c.getPublicState(); const me = st.tanks[0], en = st.tanks[1];
      const d = Math.hypot(me.x - en.x, me.y - en.y) || 1;
      if (k < WARMUP) {                                  // OBSERVE (generalist drives blue, consistent)
        if (d < CLOSE) closeCount++;
        if (prevEn) {                                    // enemy approach speed = its velocity toward blue
          const ux = (me.x - en.x) / d, uy = (me.y - en.y) / d;
          approachSum += (en.x - prevEn.x) * ux + (en.y - prevEn.y) * uy; approachN++;
        }
      } else if (!locked) {                              // CLASSIFY + LOCK
        const closeFrac = closeCount / WARMUP, approach = approachN ? approachSum / approachN : 0;
        useSpec = (closeFrac > CLOSE_FRAC) || (approach > APPROACH_THRESH);   // reckless if it stayed close OR charged
        locked = true;
      }
      prevEn = { x: en.x, y: en.y };
      finalSpec = useSpec;
      if (useSpec) specSteps++; totSteps++;
      let a;
      if (useSpec) a = SPEC_SCRIPT ? (() => { const b = c.scriptedControl(0, SPEC_SCRIPT); return controlToAction(b.throttle, b.turn, b.fire); })() : policyForward(c.observe(0), SPEC);
      else a = GEN_SCRIPT ? (() => { const b = c.scriptedControl(0, GEN_SCRIPT); return controlToAction(b.throttle, b.turn, b.fire); })() : policyForward(c.observe(0), GEN);
      const o = c.step(a, opp, { dt: 1 / 30, repeat: 2 }); info = o.info; k++;
      if (o.done) break;
    }
    n++;
    if (info.result === "win") { wins++; ttkSum += info.elapsed; ttkN++; }
    if (finalSpec === reckless) correctRoute++;
  }
  return { win: wins / n, specFrac: specSteps / totSteps, routeAcc: correctRoute / n, ttk: ttkN ? ttkSum / ttkN : null };
}

console.log(`OPPONENT ROUTER classify@${WARMUP} (ruleset=${RULESET || "v1"}, spawn=${SPAWN || "fixed"}, close-frac>${CLOSE_FRAC} OR approach>${APPROACH_THRESH}, close=${CLOSE}px)`);
console.log("opponent".padEnd(22) + ["win", "specFrac", "routeAcc", "ttk"].map(h => h.padStart(9)).join(""));
const out = {}, wins = [];
for (const opp of OPPS) {
  const m = evalOpp(opp); out[opp] = m; wins.push(m.win);
  console.log(opp.padEnd(22) + [m.win, m.specFrac, m.routeAcc, m.ttk == null ? -1 : m.ttk].map(v => v.toFixed(2).padStart(9)).join(""));
}
console.log("-".repeat(58));
console.log(`MAXIMIN win=${Math.min(...wins).toFixed(2)}  MEAN win=${(wins.reduce((a, b) => a + b, 0) / wins.length).toFixed(2)}`);
const dest = arg("out", "");
if (dest) fs.writeFileSync(dest, JSON.stringify(out, null, 2));
