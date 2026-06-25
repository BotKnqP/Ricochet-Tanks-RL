"use strict";
// Human-demonstration recorder smoke tests. Run: node smoke_record.js
// Verifies keysToControl + that human keys map to the SAME Discrete(18) action ids the
// PPO agent uses, and that a recorded JSONL is well-formed (obs length == OBS_SIZE).
const assert = require("assert");
const fs = require("fs");
const { RicochetCore, constants, OBS_SIZE, ACTION_TABLE, keysToControl, controlToAction } = require("./game_core.js");

const kc = (o) => keysToControl(o);
const R = {};

// (1-7) keysToControl mappings (spec's truth table)
assert.deepStrictEqual(kc({ w: true }), { throttle: 1, turn: 0, fire: false }, "W");
assert.deepStrictEqual(kc({ s: true }), { throttle: -1, turn: 0, fire: false }, "S");
assert.deepStrictEqual(kc({ a: true }), { throttle: 0, turn: -1, fire: false }, "A");
assert.deepStrictEqual(kc({ d: true }), { throttle: 0, turn: 1, fire: false }, "D");
assert.deepStrictEqual(kc({ space: true }), { throttle: 0, turn: 0, fire: true }, "Space");
assert.strictEqual(kc({ w: true, s: true }).throttle, 0, "W+S -> throttle 0");
assert.strictEqual(kc({ a: true, d: true }).turn, 0, "A+D -> turn 0");
assert.deepStrictEqual(kc({ w: true, space: true }), { throttle: 1, turn: 0, fire: true }, "W+Space");
assert.deepStrictEqual(kc({ w: true, a: true }), { throttle: 1, turn: -1, fire: false }, "W+A");
assert.deepStrictEqual(kc({ w: true, a: true, space: true }), { throttle: 1, turn: -1, fire: true }, "W+A+Space");
assert.deepStrictEqual(kc({ d: true, space: true }), { throttle: 0, turn: 1, fire: true }, "D+Space");

// (8) every key combo -> a valid Discrete(18) action id
let maxAction = -1;
for (const w of [0, 1]) for (const a of [0, 1]) for (const s of [0, 1]) for (const d of [0, 1]) for (const sp of [0, 1]) {
  const c = kc({ w: !!w, a: !!a, s: !!s, d: !!d, space: !!sp });
  const id = controlToAction(c.throttle, c.turn, c.fire);
  assert.ok(Number.isInteger(id) && id >= 0 && id < ACTION_TABLE.length, `action id in [0,${ACTION_TABLE.length}) for ${w}${a}${s}${d}${sp}`);
  maxAction = Math.max(maxAction, id);
}
R.maxActionSeen = maxAction;

// (12) OBS_SIZE / ACTION_TABLE unchanged (Discrete(18))
assert.strictEqual(OBS_SIZE, 105, "OBS_SIZE 105");
assert.strictEqual(ACTION_TABLE.length, 18, "ACTION_TABLE length 18 (Discrete(18))");

// (9-10) record a real moba1v1duel vs easy_laika episode -> obs len 105, JSON round-trips
const core = new RicochetCore({ seed: 1307, arenaMode: "survival", scenario: "moba1v1duel", maxSteps: 300, spawnPowerups: true });
core.reset(1307);
const keyseq = [{ w: true }, { w: true, a: true }, { space: true }, { d: true }, { w: true, space: true }, { s: true }];
const lines = [];
for (let i = 0; i < 60; i++) {
  const k = keyseq[i % keyseq.length];
  const obsBefore = core.observe(0);                         // s_t BEFORE the action
  const control = kc(k);
  const action = controlToAction(control.throttle, control.turn, control.fire);
  const out = core.step(action, "easy_laika", { dt: 1 / 30, repeat: 2 });
  assert.strictEqual(obsBefore.length, OBS_SIZE, "obs length == OBS_SIZE");
  assert.ok(action >= 0 && action < ACTION_TABLE.length, "action id in range");
  lines.push(JSON.stringify({
    type: "transition", episode: 0, step: i, scenario: "moba1v1duel", opponent: "easy_laika",
    obs: obsBefore.map((v) => Math.round(v * 1e6) / 1e6), action, control,
    keys: { w: !!k.w, a: !!k.a, s: !!k.s, d: !!k.d, space: !!k.space },
    reward: out.reward, done: out.done, result: out.info.result, elapsed: out.info.elapsed,
  }));
  if (out.done) break;
}
lines.push(JSON.stringify({
  type: "episode_summary", episode: 0, scenario: "moba1v1duel", opponent: "easy_laika",
  result: core.result(), elapsed: core.getPublicState().elapsed, totalReward: 0, steps: lines.length, good_demo: true,
}));

// every line is valid JSON; transitions carry OBS_SIZE-length obs
let tx = 0;
for (const ln of lines) {
  const o = JSON.parse(ln);                                  // (10) JSON.parse round-trips
  if (o.type === "transition") { assert.strictEqual(o.obs.length, OBS_SIZE, "parsed obs length"); tx += 1; }
}
assert.ok(tx > 0, "recorded at least one transition");
R.transitions = tx;

// write a sample JSONL for the Python read test (smoke runner reads + deletes it)
fs.writeFileSync("_record_sample.jsonl", lines.join("\n") + "\n");

console.log("RECORD SMOKE OK");
console.log(JSON.stringify(R, null, 2));
