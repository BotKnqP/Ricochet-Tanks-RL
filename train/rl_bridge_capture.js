"use strict";
// Capture bridge: like rl_bridge.js but also returns a compact VIEW (tank/shell/poison positions)
// each step, so a Python driver can record a real battle trajectory for visualization. Read-only
// w.r.t. training; never used in a training loop.

const path = require("path");
const readline = require("readline");
const { RicochetCore, OBS_SIZE } = require(path.join(__dirname, "..", "game_core.js"));

let core = null;

function view(includeWalls) {
  const s = core.getPublicState();
  const v = {
    step: s.stepCount,
    result: s.result,
    done: s.done,
    tanks: s.tanks.map((t) => ({
      id: t.id, x: Math.round(t.x), y: Math.round(t.y),
      a: Math.round(t.angle * 1000) / 1000, h: t.health, alive: !!t.alive, pw: t.power || 0,
    })),
    shells: s.shells.map((sh) => ({ x: Math.round(sh.x), y: Math.round(sh.y), o: sh.owner })),
    poison: s.poison ? { active: !!s.poison.active, rect: s.poison.safeRect || null } : null,
    powerups: (s.powerups || []).map((p) => ({ x: Math.round(p.x), y: Math.round(p.y), kind: p.kind || p.type || "" })),
  };
  if (includeWalls) {
    v.walls = (s.walls || []).map((w) => ({ x: Math.round(w.x), y: Math.round(w.y), w: Math.round(w.w), h: Math.round(w.h) }));
    v.worldW = s.worldW; v.worldH = s.worldH;
  }
  return v;
}

function handle(msg) {
  if (msg.cmd === "init") {
    const cfg = {
      seed: msg.seed || 300000, arenaMode: msg.arenaMode || "survival",
      scenario: msg.scenario || "moba1v1duel", maxSteps: msg.maxSteps || 900,
      spawnPowerups: msg.spawnPowerups !== false, shellDecay: true,
      actionRepeat: 2, stepDt: 1 / 30,
    };
    core = new RicochetCore(cfg);
    return { ok: true, obsSize: OBS_SIZE };
  }
  if (msg.cmd === "reset") {
    const obs = core.reset(msg.seed >>> 0);
    return { obs, view: view(true) };
  }
  if (msg.cmd === "step") {
    const out = core.step(msg.action | 0, msg.opponent);
    return { obs: out.obs0, reward: out.reward, done: !!out.done, result: out.info.result, view: view(false) };
  }
  throw new Error("unknown cmd " + msg.cmd);
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  const t = line.trim();
  if (!t) return;
  let resp;
  try { resp = handle(JSON.parse(t)); } catch (e) { resp = { error: e && e.message ? e.message : String(e) }; }
  process.stdout.write(JSON.stringify(resp) + "\n");
});
