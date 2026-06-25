"use strict";

const path = require("path");
const readline = require("readline");

const {
  RicochetCore,
  ACTION_TABLE,
  OBS_SIZE,
  controlToAction
} = require(path.join(__dirname, "..", "game_core.js"));

let currentConfig = {
  seed: 1307,
  arenaMode: "maze",
  maxSteps: 1000,
  actionRepeat: 2,
  stepDt: 1 / 30
};

let core = new RicochetCore(currentConfig);
let hasReset = false;

function sameConfig(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function normalizeArenaMode(value, fallback) {
  if (value === undefined || value === null) return fallback;
  if (value === "maze" || value === "open" || value === "survival") return value;
  throw new Error(`arenaMode must be "maze", "open", or "survival", got ${value}`);
}

function configFromMessage(msg) {
  const arenaMode = normalizeArenaMode(msg.arenaMode, currentConfig.arenaMode || "maze");
  const config = {
    seed: msg.seed === undefined || msg.seed === null ? currentConfig.seed : msg.seed >>> 0,
    arenaMode,
    maxSteps: Number.isInteger(msg.maxSteps) ? msg.maxSteps : currentConfig.maxSteps,
    actionRepeat: Number.isInteger(msg.actionRepeat) ? msg.actionRepeat : currentConfig.actionRepeat,
    stepDt: typeof msg.stepDt === "number" && Number.isFinite(msg.stepDt) ? msg.stepDt : currentConfig.stepDt
  };
  if (msg.spawnPowerups !== undefined) {
    config.spawnPowerups = Boolean(msg.spawnPowerups);
  } else if (currentConfig.spawnPowerups !== undefined && msg.arenaMode === undefined) {
    config.spawnPowerups = currentConfig.spawnPowerups;
  }
  if (msg.spawnJitter !== undefined) {
    config.spawnJitter = Boolean(msg.spawnJitter);
  } else if (currentConfig.spawnJitter !== undefined && msg.arenaMode === undefined) {
    config.spawnJitter = currentConfig.spawnJitter;
  }
  if (msg.shellDecay !== undefined) {
    config.shellDecay = Boolean(msg.shellDecay);
  } else if (currentConfig.shellDecay !== undefined && msg.arenaMode === undefined) {
    config.shellDecay = currentConfig.shellDecay;
  }
  if (msg.randomTurret !== undefined) {
    config.randomTurret = Boolean(msg.randomTurret);
  } else if (currentConfig.randomTurret !== undefined && msg.arenaMode === undefined) {
    config.randomTurret = currentConfig.randomTurret;
  }
  if (msg.scenario !== undefined) {
    config.scenario = msg.scenario;
  } else if (currentConfig.scenario !== undefined && msg.arenaMode === undefined) {
    config.scenario = currentConfig.scenario;
  }
  // survival_v2 / combat_v2 rules (default off -> old survival_v1 behaviour)
  if (msg.ruleset !== undefined) {
    config.ruleset = msg.ruleset;
  } else if (currentConfig.ruleset !== undefined && msg.arenaMode === undefined) {
    config.ruleset = currentConfig.ruleset;
  }
  if (msg.spawnMode !== undefined) {
    config.spawnMode = msg.spawnMode;
  } else if (currentConfig.spawnMode !== undefined && msg.arenaMode === undefined) {
    config.spawnMode = currentConfig.spawnMode;
  }
  if (msg.tankMaxHp !== undefined) {
    config.tankMaxHp = Number(msg.tankMaxHp);
  } else if (currentConfig.tankMaxHp !== undefined && msg.arenaMode === undefined) {
    config.tankMaxHp = currentConfig.tankMaxHp;
  }
  if (msg.reward !== undefined) {
    if (!msg.reward || typeof msg.reward !== "object" || Array.isArray(msg.reward)) {
      throw new Error("reward must be an object when provided");
    }
    config.reward = msg.reward;
  } else if (currentConfig.reward !== undefined) {
    config.reward = currentConfig.reward;
  }
  return config;
}

function ensureCore(config) {
  if (!sameConfig(currentConfig, config)) {
    currentConfig = config;
    core = new RicochetCore(currentConfig);
    hasReset = false;
  }
}

function navInfoFields(state) {
  if (!state.nav) return {};
  if (state.scenario === "nav_route_to_center") {
    return {
      centerStayTime: state.nav.centerStayTime,
      enteredCenter: state.nav.enteredCenter,
      wallHits: state.nav.wallHits,
      newCells: state.nav.newCells,
      stuckEvents: state.nav.stuckEvents,
      noProgressEvents: state.nav.noProgressEvents,
      pathDist: Number.isFinite(state.nav.pathDist) ? state.nav.pathDist : -1,
      bestPathDist: Number.isFinite(state.nav.bestPathDist) ? state.nav.bestPathDist : -1,
      routeSuccess: state.nav.navSuccess,
      routeTimeout: state.result === "route_timeout"
    };
  }
  return {
    survivalAfterMinCircle: state.nav.survivalAfterMinCircle,
    wallHits: state.nav.wallHits,
    pickups: state.nav.pickups,
    pickupsWhenEmpty: state.nav.pickupsWhenEmpty,
    poisonDamageTaken: state.nav.poisonDamageTaken,
    pickedAnyPowerup: state.nav.pickedAnyPowerup,
    navSuccess: state.nav.navSuccess
  };
}

function publicInfo(extra = {}) {
  const state = core.getPublicState();
  return {
    seed: state.seed,
    result: state.result,
    steps: state.stepCount,
    elapsed: Math.round(state.elapsed * 1000) / 1000,
    done: state.done,
    truncated: Boolean(state.done && (state.result === "timeout" || state.result === "nav_timeout")),
    arenaMode: state.arenaMode,
    scenario: state.scenario,
    spawnPowerups: state.spawnPowerups,
    spawnJitter: state.spawnJitter,
    obsSize: state.constants.obsSize,
    actionSize: state.constants.actionSize,
    wallRayCount: state.constants.wallRayCount,
    worldW: state.constants.worldW,
    worldH: state.constants.worldH,
    viewW: state.constants.viewW,
    viewH: state.constants.viewH,
    poisonEnabled: state.constants.poisonEnabled,
    poisonStartTime: state.constants.poisonStartTime,
    poisonDamagePerSecond: state.constants.poisonDamagePerSecond,
    poisonActive: state.poison ? Boolean(state.poison.active) : false,
    poisonAtMinCircle: state.poison ? Boolean(state.poison.atMinCircle) : false,
    shellDecay: state.constants.shellDecay,
    learnerAlive: state.tanks[0] ? Boolean(state.tanks[0].alive) : false,
    opponentAlive: state.tanks[1] ? Boolean(state.tanks[1].alive) : false,
    ...navInfoFields(state),
    ...extra
  };
}

function sanitizeObs(obs) {
  if (!Array.isArray(obs) || obs.length !== OBS_SIZE) {
    throw new Error(`expected obs length ${OBS_SIZE}, got ${Array.isArray(obs) ? obs.length : typeof obs}`);
  }
  return obs.map((value, index) => {
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`non-finite obs at ${index}: ${value}`);
    if (number < -1.000001 || number > 1.000001) {
      throw new Error(`obs out of [-1,1] at ${index}: ${number}`);
    }
    return Math.max(-1, Math.min(1, number));
  });
}

function sanitizeReward(value) {
  const reward = Number(value);
  if (!Number.isFinite(reward)) throw new Error(`non-finite reward: ${value}`);
  return reward;
}

// DAgger oracle: the expert's discretised action for player 0 in the CURRENT core state
// (no advance, no mutation). Returned alongside obs so a learner rollout can be labelled
// on-policy by the expert script. undefined when no expert requested.
function expertActionFor(expert) {
  if (!expert || typeof expert !== "string") return undefined;
  const ctrl = core.scriptedControl(0, expert);
  return controlToAction(ctrl.throttle, ctrl.turn, ctrl.fire);
}

function reset(msg) {
  const config = configFromMessage(msg);
  ensureCore(config);
  const seed = msg.seed === undefined || msg.seed === null ? config.seed : msg.seed >>> 0;
  const obs = sanitizeObs(core.reset(seed));
  hasReset = true;
  const info = publicInfo();
  const expertAction = expertActionFor(msg.expert);
  if (expertAction !== undefined) info.expertAction = expertAction;
  return {
    obs,
    reward: 0,
    done: false,
    truncated: false,
    info
  };
}

function step(msg) {
  if (!hasReset) reset({ cmd: "reset", seed: currentConfig.seed });

  const action = Number.isInteger(msg.action) ? msg.action : 0;
  if (action < 0 || action >= ACTION_TABLE.length) {
    throw new Error(`action must be an integer in [0, ${ACTION_TABLE.length - 1}], got ${msg.action}`);
  }
  const opponent = msg.opponent === undefined ? "stationary" : msg.opponent;
  const out = core.step(action, opponent);

  const info = { ...out.info, opponent };
  const expertAction = expertActionFor(msg.expert);   // expert's action for the post-step state
  if (expertAction !== undefined) info.expertAction = expertAction;
  return {
    obs: sanitizeObs(out.obs0),
    reward: sanitizeReward(out.reward),
    done: Boolean(out.done),
    truncated: Boolean(out.truncated),
    info
  };
}

function handleMessage(msg) {
  if (!msg || typeof msg !== "object") throw new Error("request must be a JSON object");
  switch (msg.cmd) {
    case "reset": return reset(msg);
    case "step": return step(msg);
    default: throw new Error(`unknown cmd: ${msg.cmd}`);
  }
}

function writeResponse(response) {
  process.stdout.write(`${JSON.stringify(response)}\n`);
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity
});

rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  try {
    writeResponse(handleMessage(JSON.parse(trimmed)));
  } catch (err) {
    writeResponse({
      error: err && err.message ? err.message : String(err)
    });
  }
});
