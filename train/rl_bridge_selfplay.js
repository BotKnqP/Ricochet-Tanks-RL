"use strict";
// Self-play vec bridge: like rl_bridge_vec.js, but each env's opponent can be driven EITHER by
// a script name (env.opponent) OR by an external action supplied per-step (opponentActions[i],
// produced in Python by a frozen "self"/past-checkpoint policy). To let Python compute that red
// action it needs the RED perspective, so every response also carries obs1 (observe(1)) per env.
// Kept SEPARATE from rl_bridge_vec.js so the proven DAgger/league path is never touched.

const path = require("path");
const readline = require("readline");

const { RicochetCore, OBS_SIZE, ACTION_TABLE } = require(path.join(__dirname, "..", "game_core.js"));

let envs = [];
let opponent = "stationary";
let seedIncrement = true;
let randomizeSeed = false;

function makeRng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

function nextSeed(env) {
  if (randomizeSeed) return Math.floor(env.rng() * 0x7fffffff);
  if (seedIncrement) return (env.seedBase + env.episode) & 0x7fffffff;
  return env.seedBase & 0x7fffffff;
}

function handleInit(msg) {
  const n = msg.n >>> 0;
  if (!n) throw new Error("init requires n >= 1");
  opponent = msg.opponent || "stationary";
  const perEnv = Array.isArray(msg.opponents) ? msg.opponents : null;
  seedIncrement = msg.seedIncrement !== false;
  randomizeSeed = Boolean(msg.randomizeSeed);
  const baseSeed = (msg.baseSeed === undefined ? 1307 : msg.baseSeed) >>> 0;
  const stride = Number.isInteger(msg.seedStride) ? msg.seedStride : 10000;
  const coreCfg = msg.core || {};
  envs = [];
  for (let i = 0; i < n; i++) {
    const seedBase = (baseSeed + i * stride) >>> 0;
    envs.push({
      core: new RicochetCore({ ...coreCfg, seed: seedBase }),
      seedBase,
      episode: 0,
      rng: makeRng(seedBase ^ 0x9e3779b9),
      opponent: perEnv ? (perEnv[i] || opponent) : opponent
    });
  }
  return { ok: true, n, obsSize: OBS_SIZE, actionSize: ACTION_TABLE.length, opponents: envs.map((e) => e.opponent) };
}

function handleReset() {
  const obs = new Array(envs.length);
  const obs1 = new Array(envs.length);
  for (let i = 0; i < envs.length; i++) {
    const env = envs[i];
    env.episode = 0;
    obs[i] = env.core.reset(nextSeed(env));
    obs1[i] = env.core.observe(1);            // red perspective for the first opponent action
  }
  return { obs, obs1 };
}

function handleStep(msg) {
  const actions = msg.actions;
  if (!Array.isArray(actions) || actions.length !== envs.length) {
    throw new Error(`step expects ${envs.length} actions, got ${Array.isArray(actions) ? actions.length : typeof actions}`);
  }
  // opponentActions[i]: integer red action for a policy-controlled env, or null/undefined to fall
  // back to that env's scripted opponent. Lets a single rollout MIX scripts and frozen selves.
  const oppActions = Array.isArray(msg.opponentActions) ? msg.opponentActions : null;
  const obs = new Array(envs.length);
  const obs1 = new Array(envs.length);
  const reward = new Array(envs.length);
  const done = new Array(envs.length);
  const info = new Array(envs.length);
  for (let i = 0; i < envs.length; i++) {
    const env = envs[i];
    const ra = oppActions ? oppActions[i] : null;
    const input1 = (ra === null || ra === undefined) ? env.opponent : (ra | 0);
    const out = env.core.step(actions[i] | 0, input1);
    reward[i] = out.reward;
    if (out.done) {
      const terminal = out.obs0;
      env.episode += 1;
      obs[i] = env.core.reset(nextSeed(env));
      obs1[i] = env.core.observe(1);          // fresh episode's red obs (terminal red obs not needed)
      done[i] = true;
      info[i] = {
        terminal_observation: terminal,
        "TimeLimit.truncated": Boolean(out.truncated),
        result: out.info.result
      };
    } else {
      obs[i] = out.obs0;
      obs1[i] = out.obs1;
      done[i] = false;
      info[i] = {};
    }
  }
  return { obs, obs1, reward, done, info };
}

function handle(msg) {
  if (!msg || typeof msg !== "object") throw new Error("request must be a JSON object");
  switch (msg.cmd) {
    case "init": return handleInit(msg);
    case "reset": return handleReset(msg);
    case "step": return handleStep(msg);
    default: throw new Error(`unknown cmd: ${msg.cmd}`);
  }
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let response;
  try {
    response = handle(JSON.parse(trimmed));
  } catch (err) {
    response = { error: err && err.message ? err.message : String(err) };
  }
  process.stdout.write(`${JSON.stringify(response)}\n`);
});
