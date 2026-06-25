// game_render.js — browser front-end for Ricochet Tanks.
//
// Owns ONLY presentation + input: canvas drawing, audio, explosion/beam
// particles, keyboard/pointer input, the requestAnimationFrame loop, the live
// "watch" overlay, and choosing which controller drives each tank. All physics
// comes from game_core.js (loaded first): it creates a RicochetCore, drives it
// with advance(), draws from getPublicState(), and turns the per-tick `events`
// list into sound + particles. No simulation lives here.

(() => {
  "use strict";

  const Core = window.RicochetCoreModule;
  if (!Core) throw new Error("game_core.js must be loaded before game_render.js");
  const { RicochetCore, policyForward, keysToControl, controlToAction } = Core;

  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  const p1ScoreEl = document.getElementById("p1Score");
  const p2ScoreEl = document.getElementById("p2Score");
  const modeLabel = document.getElementById("modeLabel");
  const arenaSeedEl = document.getElementById("arenaSeed");
  const powerupLabel = document.getElementById("powerupLabel");

  const urlParams = new URLSearchParams(window.location.search);
  const scenarioParam = urlParams.get("scenario");
  const scenario = (scenarioParam === "nav_powerup_poison" || scenarioParam === "nav_route_to_center" || scenarioParam === "fixed_moba" || scenarioParam === "moba1v1duel" || scenarioParam === "moba_poison_run") ? scenarioParam : "battle";
  const isNavScenario = scenario === "nav_powerup_poison" || scenario === "nav_route_to_center" || scenario === "moba_poison_run";  // nav = red removed
  const usesFixedMap = scenario === "fixed_moba" || scenario === "moba1v1duel" || scenario === "moba_poison_run";  // fixed symmetric map
  const arenaMode = (isNavScenario || usesFixedMap) ? "survival"   // nav lessons + fixed maps require the survival world
    : urlParams.get("arena") === "open" ? "open"
    : urlParams.get("arena") === "survival" ? "survival"
    : "maze";
  const powerupsParam = urlParams.get("powerups");
  const spawnPowerups = powerupsParam === "0" || powerupsParam === "false"
    ? false
    : powerupsParam === "1" || powerupsParam === "true"
      ? true
      : arenaMode === "open" ? false : true;
  const jitterParam = urlParams.get("jitter");
  const spawnJitter = (jitterParam === "1" || jitterParam === "true") && arenaMode === "open";
  const cameraParam = urlParams.get("camera");
  const decayParam = urlParams.get("decay");
  const shellDecay = decayParam !== "0" && decayParam !== "false";  // experimental shell decay, on by default
  // survival_v2 / combat_v2 long-form combat: ?ruleset=survival_v2 (HP x2 / slower regen / random spawn).
  // ?spawn=fixed|half_random|full_random overrides the spawn mode.
  const ruleset = urlParams.get("ruleset") === "survival_v2" ? "survival_v2" : undefined;
  const spawnModeParam = urlParams.get("spawn");
  const coreOpts = { seed: 1307, arenaMode, spawnPowerups, spawnJitter, shellDecay, scenario };
  if (ruleset) coreOpts.ruleset = ruleset;
  if (spawnModeParam) coreOpts.spawnMode = spawnModeParam;
  const core = new RicochetCore(coreOpts);
  const {
    W, H, WALL, COLS, ROWS, CELL_W, CELL_H, TANK_RADIUS, TANK_SCALE,
    MAX_HEALTH, FIRE_DELAY, ARENA_DIAG, TAU, worldW, worldH
  } = core.constants;

  let view = core.getPublicState();
  const explosions = [];
  const beams = [];
  const keys = new Set();

  const HUMAN_KEYS = [
    { up: "w", down: "s", left: "a", right: "d", fire: " " },
    { up: "ArrowUp", down: "ArrowDown", left: "ArrowLeft", right: "ArrowRight", fire: "Enter" }
  ];

  const app = {
    scores: [0, 0],
    botEnabled: true,
    botType: "laika",
    botModel: null,
    blueController: "human",
    cameraMode: "god",
    speedMultiplier: 1,
    navGoal: null,
    navPath: null,
    roundMessage: "",
    roundMessageTimer: 0,
    pendingReset: 0,
    watch: { enabled: false, runDir: "runs/live_demo", pollMs: 2000 },
    play: { enabled: false, humanSide: 1 },   // human-vs-agent: you drive one tank, the agent the other (humanSide 1=red, 0=blue)
    record: { enabled: false, recording: false, episode: 0, step: 0, totalReward: 0, goodDemo: true, accum: 0, episodeFinalized: false, lines: [], lastActionId: 0, lastResult: "running" },
    lastTime: performance.now()
  };

  // ----------------------------- audio -----------------------------
  const audio = { ctx: null, lastBounce: 0 };
  function ensureAudio() {
    if (!audio.ctx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return null;
      audio.ctx = new AudioContext();
    }
    if (audio.ctx.state === "suspended") audio.ctx.resume();
    return audio.ctx;
  }
  function playSound(type) {
    const actx = audio.ctx;
    if (!actx) return;
    if (type === "bounce" && actx.currentTime - audio.lastBounce < 0.045) return;
    if (type === "bounce") audio.lastBounce = actx.currentTime;
    const gain = actx.createGain();
    const osc = actx.createOscillator();
    gain.connect(actx.destination);
    osc.connect(gain);
    let start = 260, end = 160, volume = 0.07, duration = 0.08, wave = "square";
    if (type === "fire") { start = 220; end = 90; duration = 0.06; volume = 0.06; wave = "sawtooth"; }
    else if (type === "laser") { start = 920; end = 420; duration = 0.14; volume = 0.08; wave = "triangle"; }
    else if (type === "missile") { start = 130; end = 240; duration = 0.16; volume = 0.065; wave = "sawtooth"; }
    else if (type === "bounce") { start = 640; end = 380; duration = 0.035; volume = 0.035; wave = "square"; }
    else if (type === "power") { start = 420; end = 760; duration = 0.12; volume = 0.06; wave = "sine"; }
    else if (type === "shield") { start = 560; end = 280; duration = 0.16; volume = 0.07; wave = "triangle"; }
    else if (type === "explode") { start = 90; end = 34; duration = 0.28; volume = 0.1; wave = "sawtooth"; }
    osc.type = wave;
    osc.frequency.setValueAtTime(start, actx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(Math.max(20, end), actx.currentTime + duration);
    gain.gain.setValueAtTime(volume, actx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + duration);
    osc.start();
    osc.stop(actx.currentTime + duration);
  }

  // --------------------------- particles ---------------------------
  function createExplosion(x, y, color, count) {
    if (count >= 20) playSound("explode");
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * TAU;
      const speed = 70 + Math.random() * 210;
      explosions.push({
        x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
        radius: 2 + Math.random() * 4, ttl: 0.45 + Math.random() * 0.5, maxTtl: 0.95, color
      });
    }
  }

  // core events -> sound + particles (the renderer's only job for game logic)
  function handleEvents(events) {
    for (const e of events) {
      switch (e.type) {
        case "fire":
          playSound(e.weapon === "missile" ? "missile" : e.weapon === "snipe" ? "laser" : "fire");
          break;
        case "bounce": playSound("bounce"); break;
        case "shield": createExplosion(e.x, e.y, "#56d6c9", 14); playSound("shield"); break;
        case "hit": createExplosion(e.x, e.y, e.color, 10); playSound("bounce"); break;
        case "defeat": createExplosion(e.x, e.y, e.color, 34); break;
        case "missileExplode": createExplosion(e.x, e.y, "#ff9f43", 16); break;
        case "power": playSound("power"); break;
        case "laser": beams.push({ owner: e.owner, path: e.path, ttl: 0.16 }); break;
        default: break;
      }
    }
  }

  function updateVisuals(dt) {
    for (let i = beams.length - 1; i >= 0; i--) {
      beams[i].ttl -= dt;
      if (beams[i].ttl <= 0) beams.splice(i, 1);
    }
    for (let i = explosions.length - 1; i >= 0; i--) {
      const p = explosions[i];
      p.ttl -= dt;
      p.x += p.vx * dt; p.y += p.vy * dt;
      p.vx *= 0.92; p.vy *= 0.92;
      if (p.ttl <= 0) explosions.splice(i, 1);
    }
  }

  // ------------------------ neural policies ------------------------
  function availableModels() { return Object.keys(window.RICOCHET_POLICIES || {}); }
  function neuralReady() {
    return availableModels().length > 0 && Boolean(window.RICOCHET_POLICIES[app.botModel]);
  }
  function neuralAction(playerId, modelName) {
    const obs = core.buildObservation(playerId);
    const policy = modelName ? window.RICOCHET_POLICIES[modelName] : window.RICOCHET_POLICIES[app.botModel];
    return policyForward(obs, policy);
  }
  // OPPONENT ROUTER (cracks the multi-task conflict): classify the enemy as reckless vs evasive from its APPROACH
  // SPEED over a warmup window (blue runs the generalist meanwhile), then LOCK the route -- reckless -> the
  // aggressive-mirror script ("aggressive"=laika-aggressive, the v2 reckless-counter), evasive -> the generalist policy.
  const ROUTER_WARMUP = 35, ROUTER_APPROACH = 4.5;
  function routerAction() {
    const r = app.router || (app.router = { k: 0, aSum: 0, aN: 0, prevEn: null, locked: false, useScript: false });
    const me = view.tanks[0], en = view.tanks[1];
    const gen = app.routerGen || "v2";
    if (!me || !en) return window.RICOCHET_POLICIES[gen] ? neuralAction(0, gen) : "laika";
    const d = Math.hypot(me.x - en.x, me.y - en.y) || 1;
    if (r.k < ROUTER_WARMUP) {
      if (r.prevEn) { const ux = (me.x - en.x) / d, uy = (me.y - en.y) / d; r.aSum += (en.x - r.prevEn.x) * ux + (en.y - r.prevEn.y) * uy; r.aN++; }
    } else if (!r.locked) {
      r.useScript = (r.aN ? r.aSum / r.aN : 0) > ROUTER_APPROACH;   // reckless -> aggressive mirror
      r.locked = true;
    }
    r.prevEn = { x: en.x, y: en.y }; r.k++;
    return r.useScript ? "aggressive" : (window.RICOCHET_POLICIES[gen] ? neuralAction(0, gen) : "laika");
  }

  // --------------------- controller selection ----------------------
  function humanControl(id) {
    const c = HUMAN_KEYS[id];
    return {
      throttle: (keys.has(c.up) ? 1 : 0) + (keys.has(c.down) ? -1 : 0),
      turn: (keys.has(c.right) ? 1 : 0) + (keys.has(c.left) ? -1 : 0),
      fire: keys.has(c.fire)
    };
  }
  // The AGENT driving an arbitrary seat (used by play-vs-agent for whichever side the human is NOT on).
  function agentAction(playerId, model) {
    // NEURAL (105-d) policies loaded into RICOCHET_POLICIES drive via policyForward on the live 105-d obs; this is how
    // the v1.5 DAgger clone plays. Everything else is SCRIPT-based (obs-independent, works either seat): aggro =
    // laika-aggressive (the verified champion, maximin ~0.46-0.50), or any scripted style by name (ace/charger/laika/pro).
    if (model && window.RICOCHET_POLICIES && window.RICOCHET_POLICIES[model]) return neuralAction(playerId, model);
    if (!model || model === "aggro") return "aggressive";
    return model;   // core runs scriptedControl(seat, model)
  }
  // A controller for either seat: "player" -> keyboard, a loaded RICOCHET_POLICIES key -> neural agent, else a script name.
  function controllerInput(seat, ctrl) {
    if (ctrl === "player") return humanControl(seat);
    if (ctrl && window.RICOCHET_POLICIES && window.RICOCHET_POLICIES[ctrl]) return neuralAction(seat, ctrl);
    if (ctrl === "aggro" || ctrl === "laika-aggressive") return "aggressive";
    return ctrl || "stationary";   // any other string -> core.scriptedControl(seat, ctrl)
  }
  function inputForBlue() {
    if (app.matchActive) return controllerInput(0, app.blueCtrl);
    if (app.play.enabled) return app.play.humanSide === 0 ? humanControl(0) : agentAction(0, app.liveModel);
    if (app.watch.enabled) return agentAction(0, app.liveModel);   // watch: the (script-based) agent drives blue
    if (app.blueController !== "human") {
      if (app.blueController === "pathfind") {
        const me = view.tanks[0];
        const goal = app.navGoal || (view.tanks[1] ? { x: view.tanks[1].x, y: view.tanks[1].y } : null);
        if (me && goal) app.navPath = core.planPath(me.x, me.y, goal.x, goal.y);
        return core.scriptedControl(0, "pathfind", { goal: app.navGoal });
      }
      if (app.blueController === "laika-aggressive") return "aggressive";
      if (app.blueController === "laika-aggressive-pro") return "laika-aggressive-pro";
      if (app.blueController === "laika") return "laika";
      if (window.RICOCHET_POLICIES && window.RICOCHET_POLICIES[app.blueController]) {
        return neuralAction(0, app.blueController);
      }
    }
    return humanControl(0);
  }
  function inputForRed() {
    if (app.matchActive) return controllerInput(1, app.redCtrl);
    if (app.play.enabled) return app.play.humanSide === 1 ? humanControl(1) : agentAction(1, app.liveModel);
    if (app.botEnabled) {
      if (app.botType === "none") return "none";
      if (app.botType === "neural" && neuralReady()) return neuralAction(1, app.botModel);
      if (app.botType === "laika-aggressive") return "aggressive";   // legacy alias -> core "aggressive"
      // pass any other script name straight to core.scriptedControl: laika/stationary/turret/easy_laika/
      // laika-aggressive-pro AND the parameterized weird family (charger/spammer/precision/turtle/...).
      return app.botType || "laika";
    }
    return humanControl(1);
  }

  // --------------------------- main loop ---------------------------
  function tick(dt) {
    if (core.isDone()) {
      if (app.pendingReset <= 0) {
        const res = core.result();
        if (res && res.indexOf("nav_success") === 0) { app.roundMessage = "Reached safety!"; }
        else if (res === "route_success") { app.roundMessage = "Reached centre!"; }
        else if (res === "nav_death") { app.roundMessage = "Poisoned"; }
        else if (res === "route_death") { app.roundMessage = "Down"; }
        else if (res === "nav_timeout" || res === "route_timeout") { app.roundMessage = "Time up"; }
        else if (res === "win") { app.scores[0] += 1; app.roundMessage = "Blue scores"; }
        else if (res === "loss") { app.scores[1] += 1; app.roundMessage = "Red scores"; }
        else app.roundMessage = "Draw";
        app.roundMessageTimer = 1.5;
        app.pendingReset = 1.2;
        updateHud();
      }
      app.pendingReset -= dt;
      app.roundMessageTimer = Math.max(0, app.roundMessageTimer - dt);
      updateVisuals(dt);
      if (app.pendingReset <= 0) {
        core.reset();
        view = core.getPublicState();
        app.navGoal = null;
        arenaSeedEl.textContent = String(view.seed);
        updateHud();
      }
      return;
    }
    if (app.matchActive || (app.watch.enabled && window.RICOCHET_POLICIES && window.RICOCHET_POLICIES[app.liveModel || "live"])) {
      // Drive the policy at the EXACT training/eval cadence: one decision per RL step, held across
      // action_repeat=2 updates of dt=1/30. The old advance(dt)+decide-every-frame path re-decided
      // BOTH tanks every physics update, so laika reacted at 2x and the agent (tuned for repeat=2)
      // badly underperformed (browser 18% vs the true ~64%).
      const out = core.step(inputForBlue(), inputForRed(), { dt: 1 / 30, repeat: 2 });
      view = core.getPublicState();
      handleEvents(out.events);
      updateVisuals(dt);
      app.roundMessageTimer = Math.max(0, app.roundMessageTimer - dt);
      return;
    }
    const out = core.advance(dt, inputForBlue(), inputForRed());
    view = out.publicState;
    handleEvents(out.events);
    updateVisuals(dt);
    app.roundMessageTimer = Math.max(0, app.roundMessageTimer - dt);
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - app.lastTime) / 1000);
    app.lastTime = now;
    if (app.record.enabled) {
      recordFrame(dt);
    } else if (app.matchActive || (app.watch.enabled && window.RICOCHET_POLICIES && window.RICOCHET_POLICIES[app.liveModel || "live"])) {
      // Watch mode runs at the TRAINING cadence (one decision per RL step = action_repeat=2 updates,
      // RL_DT s of game time). Advance exactly dt*speed of game-time via an accumulator, so 1x is
      // real-time (not 2x). Cap steps/frame so high speed can't saturate the obs raycast + wedge the
      // renderer; drop the backlog past the cap rather than spiralling.
      const RL_DT = 2 / 30;
      app.watchAccum = (app.watchAccum || 0) + dt * app.speedMultiplier;
      let steps = 0;
      while (app.watchAccum >= RL_DT && steps < 8) { tick(RL_DT); app.watchAccum -= RL_DT; steps++; }
      if (steps >= 8) app.watchAccum = 0;
    } else {
      let remaining = dt * app.speedMultiplier;
      let guard = 0;
      while (remaining > 0 && guard < 32) {
        const step = Math.min(0.033, remaining);
        tick(step);
        remaining -= step;
        guard++;
      }
    }
    draw();
    requestAnimationFrame(frame);
  }

  // ----------------- human demonstration recording (record=human) ----------------
  // Drives the sim with FIXED RL steps (dt=1/30, action_repeat=2 — same as training),
  // recording obs_before -> human action id per RL step. Human WASD+Space are mapped via
  // keysToControl()+controlToAction() to the SAME Discrete(18) ids the PPO agent uses, so
  // the JSONL is drop-in for behaviour cloning (target = action id, not raw key events).
  const REC_STEP_DT = 1 / 30, REC_REPEAT = 2, REC_RL_DT = REC_STEP_DT * REC_REPEAT;
  function recKeys() {
    return { w: keys.has("w"), a: keys.has("a"), s: keys.has("s"), d: keys.has("d"), space: keys.has(" ") };
  }
  function recordStep() {
    const k = recKeys();
    const obsBefore = core.observe(0);                                  // s_t (BEFORE the action)
    const control = keysToControl(k);
    const action = controlToAction(control.throttle, control.turn, control.fire);
    const out = core.step(action, "easy_laika", { dt: REC_STEP_DT, repeat: REC_REPEAT });
    view = core.getPublicState();
    handleEvents(out.events);
    app.record.lines.push({
      type: "transition", episode: app.record.episode, step: app.record.step,
      scenario: "moba1v1duel", opponent: "easy_laika",
      obs: obsBefore.map((v) => Math.round(v * 1e6) / 1e6),
      action, control, keys: k,
      reward: Math.round(out.reward * 1e6) / 1e6,
      done: Boolean(out.done), result: out.info.result,
      elapsed: Math.round(out.info.elapsed * 1000) / 1000
    });
    app.record.step += 1;
    app.record.totalReward += out.reward;
    app.record.lastActionId = action;
    app.record.lastResult = out.info.result;
    if (out.done) finalizeEpisode();
    updateHud();
  }
  function recordFrame(dt) {
    if (!app.record.recording) return;                 // paused: freeze the sim
    if (core.isDone()) { finalizeEpisode(); return; }
    app.record.accum += dt;
    let guard = 0;
    while (app.record.accum >= REC_RL_DT && !core.isDone() && guard < 6) {
      app.record.accum -= REC_RL_DT;
      recordStep();
      guard += 1;
    }
  }
  function finalizeEpisode() {
    if (app.record.episodeFinalized) return;
    app.record.episodeFinalized = true;
    app.record.recording = false;                      // auto-pause at episode end
    app.record.lines.push({
      type: "episode_summary", episode: app.record.episode,
      scenario: "moba1v1duel", opponent: "easy_laika",
      result: app.record.lastResult, elapsed: Math.round((view.elapsed || 0) * 1000) / 1000,
      totalReward: Math.round(app.record.totalReward * 1e6) / 1e6,
      steps: app.record.step, good_demo: app.record.goodDemo
    });
    app.roundMessage = app.record.lastResult === "win" ? "WIN — saved (N=next)"
      : app.record.lastResult === "loss" ? "LOSS — saved (N=next)" : "Episode saved (N=next)";
    app.roundMessageTimer = 2.2;
    updateHud();
  }
  function newRecordEpisode() {
    if (!app.record.episodeFinalized && app.record.step > 0) finalizeEpisode();
    core.reset();
    view = core.getPublicState();
    arenaSeedEl.textContent = String(view.seed);
    app.record.episode += 1;
    app.record.step = 0;
    app.record.totalReward = 0;
    app.record.goodDemo = true;
    app.record.accum = 0;
    app.record.episodeFinalized = false;
    app.record.lastResult = "running";
    resetRoundUi();
    updateHud();
  }
  function exportJSONL() {
    const txCount = app.record.lines.filter((l) => l.type === "transition").length;
    if (!app.record.lines.length) { app.roundMessage = "No data to export"; app.roundMessageTimer = 1.6; updateHud(); return; }
    const body = app.record.lines.map((o) => JSON.stringify(o)).join("\n") + "\n";
    const blob = new Blob([body], { type: "application/x-ndjson" });
    const d = new Date(), pad = (n) => String(n).padStart(2, "0");
    const ts = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `human_demo_moba1v1duel_easy_laika_${ts}.jsonl`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    app.roundMessage = `Exported ${txCount} transitions`;
    app.roundMessageTimer = 2.2; updateHud();
  }
  function drawRecordOverlay() {
    if (!app.record.enabled) return;
    const r = app.record;
    ctx.save();
    ctx.fillStyle = "rgba(8,10,16,0.82)";
    roundedRect(W - 234, 10, 224, 96, 8); ctx.fill();
    ctx.textBaseline = "middle";
    ctx.font = "800 14px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = r.recording ? "#ff5d5d" : "#9aa4b2";
    ctx.fillText(r.recording ? "● REC" : "❚❚ PAUSED", W - 222, 28);
    ctx.textAlign = "right";
    ctx.fillStyle = r.goodDemo ? "#7cfc84" : "#ffb86b";
    ctx.font = "700 11px system-ui, sans-serif";
    ctx.fillText(r.goodDemo ? "good_demo" : "BAD_demo", W - 20, 28);
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(230,237,243,0.92)";
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.fillText(`ep ${r.episode} · step ${r.step} · a=${r.lastActionId}`, W - 222, 50);
    ctx.fillText(`result: ${r.lastResult} · demos ${r.lines.filter((l) => l.type === "transition").length}`, W - 222, 68);
    ctx.fillStyle = "rgba(230,237,243,0.55)";
    ctx.font = "600 10px system-ui, sans-serif";
    ctx.fillText("R rec · N next · E export · G/B mark", W - 222, 90);
    ctx.restore();
  }

  // ----------------------------- draw ------------------------------
  // Minimal "god" camera: scale the (possibly 2x) world to fit the canvas,
  // centered. For old 960x640 modes this is identity (s=1). Follow-blue/red
  // modes and screen-space UI come in the render stage.
  function normalizeCamera(c) {
    c = (c || "").toLowerCase();
    if (c === "god") return "god";
    if (c === "blue" || c === "followblue") return "blue";
    if (c === "red" || c === "followred") return "red";
    return null;
  }

  // god: fit the whole world to the canvas. blue/red: follow that tank at 1:1, the
  // followed tank near screen centre, clamped to the world edges. For old 960x640
  // worlds every mode reduces to the full view (identity).
  function worldCamera() {
    if (app.cameraMode === "blue" || app.cameraMode === "red") {
      const t = app.cameraMode === "blue" ? view.tanks[0] : view.tanks[1];
      const cx = t ? t.x : worldW / 2, cy = t ? t.y : worldH / 2;
      // Always centre the followed tank (no clamp), so an edge tank still sits
      // mid-screen; the area beyond the world is filled with a void by drawFloor.
      return { s: 1, ox: -(cx - W / 2), oy: -(cy - H / 2) };
    }
    const s = Math.min(W / worldW, H / worldH);
    return { s, ox: (W - worldW * s) / 2, oy: (H - worldH * s) / 2 };
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const cam = worldCamera();
    ctx.save();
    ctx.translate(cam.ox, cam.oy);
    ctx.scale(cam.s, cam.s);
    drawFloor();
    drawPoison();
    drawNavPath();
    drawPowerups();
    drawLaserAims();
    drawSnipeAims();
    drawShells();
    drawBeams();
    drawTanks();
    drawWalls();
    drawExplosions();
    ctx.restore();
    drawTankHealthBars(cam);   // screen-space UI: not scaled by the camera
    drawMinimap(cam);
    drawOverlay();
    drawRecordOverlay();
  }

  // MOBA-style minimap (follow views only): the whole world shrunk into a corner
  // with tanks/poison/powerups and a rectangle marking the magnified main view.
  function drawMinimap(cam) {
    if (app.cameraMode === "god") return;
    const mmW = 200, mmH = mmW * worldH / worldW, pad = 12;
    const mx = W - mmW - pad, my = H - mmH - pad, ms = mmW / worldW;
    const p = view.poison;
    ctx.save();
    ctx.fillStyle = "rgba(10,12,18,0.80)";
    ctx.fillRect(mx - 3, my - 3, mmW + 6, mmH + 6);
    ctx.strokeStyle = "rgba(255,255,255,0.28)";
    ctx.lineWidth = 1;
    ctx.strokeRect(mx - 3, my - 3, mmW + 6, mmH + 6);
    ctx.beginPath(); ctx.rect(mx, my, mmW, mmH); ctx.clip();
    ctx.fillStyle = "#20242d"; ctx.fillRect(mx, my, mmW, mmH);
    ctx.fillStyle = "rgba(130,140,160,0.35)";
    for (const wall of view.walls) ctx.fillRect(mx + wall.x * ms, my + wall.y * ms, Math.max(1, wall.w * ms), Math.max(1, wall.h * ms));
    if (p && p.safeRect) {
      const sr = p.safeRect;
      ctx.fillStyle = "rgba(150,40,200,0.30)";
      ctx.beginPath();
      ctx.rect(mx, my, mmW, mmH);
      ctx.rect(mx + sr.x * ms, my + sr.y * ms, sr.w * ms, sr.h * ms);
      ctx.fill("evenodd");
      ctx.strokeStyle = "rgba(214,130,255,0.9)"; ctx.lineWidth = 1;
      ctx.strokeRect(mx + sr.x * ms, my + sr.y * ms, sr.w * ms, sr.h * ms);
    }
    for (const pw of view.powerups) { ctx.fillStyle = pw.color || "#fff"; ctx.beginPath(); ctx.arc(mx + pw.x * ms, my + pw.y * ms, 2, 0, TAU); ctx.fill(); }
    for (const t of view.tanks) {
      if (!t.alive) continue;
      ctx.fillStyle = t.id === 0 ? "#58a6ff" : "#ff6b6b";
      ctx.beginPath(); ctx.arc(mx + t.x * ms, my + t.y * ms, 3, 0, TAU); ctx.fill();
    }
    // the magnifier lens: rectangle of the currently magnified main view
    const vx = -cam.ox / cam.s, vy = -cam.oy / cam.s, vw = W / cam.s, vh = H / cam.s;
    ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 1.5;
    ctx.strokeRect(mx + vx * ms, my + vy * ms, vw * ms, vh * ms);
    ctx.restore();
  }

  function drawFloor() {
    // out-of-bounds void (seen when the follow camera centres on an edge tank)
    ctx.fillStyle = "#15171c";
    ctx.fillRect(-worldW, -worldH, worldW * 3, worldH * 3);
    ctx.fillStyle = "#20242d";
    ctx.fillRect(0, 0, worldW, worldH);
    ctx.strokeStyle = "rgba(255,255,255,0.035)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= worldW; x += CELL_W / 2) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, worldH); ctx.stroke();
    }
    for (let y = 0; y <= worldH; y += CELL_H / 2) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(worldW, y); ctx.stroke();
    }
  }

  function drawPoison() {
    const p = view.poison;
    if (!p || !p.safeRect) return;
    const sr = p.safeRect;
    ctx.save();
    // tint the danger area (world minus safe rect) via an even-odd hole
    ctx.fillStyle = p.active ? "rgba(150, 40, 200, 0.24)" : "rgba(150, 40, 200, 0.07)";
    ctx.beginPath();
    ctx.rect(0, 0, worldW, worldH);
    ctx.rect(sr.x, sr.y, sr.w, sr.h);
    ctx.fill("evenodd");
    // safe-zone border
    ctx.strokeStyle = p.active ? "rgba(214, 130, 255, 0.9)" : "rgba(214, 130, 255, 0.45)";
    ctx.lineWidth = 4;
    ctx.strokeRect(sr.x, sr.y, sr.w, sr.h);
    ctx.restore();
  }

  function drawWalls() {
    for (const wall of view.walls) {
      const gradient = ctx.createLinearGradient(wall.x, wall.y, wall.x + wall.w, wall.y + wall.h);
      gradient.addColorStop(0, "#3a404d");
      gradient.addColorStop(1, "#171b22");
      ctx.fillStyle = gradient;
      ctx.fillRect(wall.x, wall.y, wall.w, wall.h);
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.strokeRect(wall.x + 0.5, wall.y + 0.5, wall.w - 1, wall.h - 1);
    }
  }

  function drawNavPath() {
    if (app.blueController !== "pathfind" || !app.navPath || app.navPath.length < 2) return;
    ctx.save();
    ctx.strokeStyle = "rgba(120, 200, 255, 0.55)";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.setLineDash([6, 7]);
    ctx.beginPath();
    ctx.moveTo(app.navPath[0].x, app.navPath[0].y);
    for (let i = 1; i < app.navPath.length; i++) ctx.lineTo(app.navPath[i].x, app.navPath[i].y);
    ctx.stroke();
    ctx.setLineDash([]);
    const g = app.navGoal || (view.tanks[1] ? { x: view.tanks[1].x, y: view.tanks[1].y } : null);
    if (g) {
      ctx.fillStyle = "rgba(120, 200, 255, 0.85)";
      ctx.beginPath(); ctx.arc(g.x, g.y, 6, 0, TAU); ctx.fill();
    }
    ctx.restore();
  }

  function drawLaserAims() {
    for (const tank of view.tanks) {
      if (!tank.alive || tank.power !== "laser") continue;
      const path = core.traceLaser(tank);
      ctx.save();
      ctx.lineCap = "round";
      ctx.setLineDash([10, 10]);
      for (const segment of path) {
        ctx.strokeStyle = tank.id === 0 ? "rgba(183, 220, 255, 0.38)" : "rgba(255, 208, 208, 0.38)";
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(segment.x1, segment.y1); ctx.lineTo(segment.x2, segment.y2); ctx.stroke();
        ctx.strokeStyle = "rgba(255, 79, 216, 0.32)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(segment.x1, segment.y1); ctx.lineTo(segment.x2, segment.y2); ctx.stroke();
      }
      ctx.restore();
    }
  }

  function drawSnipeAims() {
    for (const tank of view.tanks) {
      if (!tank.alive || tank.power !== "bounce") continue;
      ctx.save();
      ctx.lineCap = "round";
      ctx.setLineDash([14, 10]);
      for (const [offset, width, alpha] of [[-0.14, 2, 0.34], [0, 3, 0.52], [0.14, 2, 0.34]]) {
        const angle = tank.angle + offset;
        const x1 = tank.x + Math.cos(angle) * (tank.radius + 16);
        const y1 = tank.y + Math.sin(angle) * (tank.radius + 16);
        const x2 = x1 + Math.cos(angle) * ARENA_DIAG * 1.7;
        const y2 = y1 + Math.sin(angle) * ARENA_DIAG * 1.7;
        ctx.strokeStyle = `rgba(242, 231, 255, ${alpha})`;
        ctx.lineWidth = width + 2;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.strokeStyle = `rgba(199, 146, 234, ${Math.min(0.9, alpha + 0.18)})`;
        ctx.lineWidth = width;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }
      ctx.restore();
    }
  }

  function roundedRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
  }

  function drawTanks() {
    for (const tank of view.tanks) {
      ctx.save();
      ctx.translate(tank.x, tank.y);
      ctx.rotate(tank.angle);
      ctx.scale(TANK_SCALE, TANK_SCALE);
      ctx.globalAlpha = tank.alive ? 1 : 0.35;
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      ctx.beginPath(); ctx.ellipse(0, 5, 22, 15, 0, 0, TAU); ctx.fill();
      ctx.fillStyle = tank.color;
      ctx.strokeStyle = "#090b10";
      ctx.lineWidth = 3;
      roundedRect(-19, -14, 38, 28, 7);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = tank.accent;
      ctx.fillRect(0, -5, 28, 10);
      ctx.strokeRect(0, -5, 28, 10);
      ctx.fillStyle = "#111318";
      ctx.fillRect(-14, -19, 9, 7);
      ctx.fillRect(5, -19, 9, 7);
      ctx.fillRect(-14, 12, 9, 7);
      ctx.fillRect(5, 12, 9, 7);
      drawPowerSkin(tank);
      if (tank.shield > 0) {
        ctx.strokeStyle = "#56d6c9";
        for (let i = 0; i < tank.shield; i++) {
          ctx.lineWidth = 3 - i * 0.5;
          ctx.beginPath(); ctx.arc(0, 0, 27 + i * 6, 0, TAU); ctx.stroke();
        }
      }
      ctx.restore();
    }
  }

  // Health + reload bars, drawn in screen space (projected via the camera) so they
  // keep a constant pixel size regardless of camera zoom. Shown for both tanks, in
  // play and watch. Reload bar is always visible: green = ready, amber = reloading.
  function drawTankHealthBars(cam) {
    for (const tank of view.tanks) {
      if (!tank.alive) continue;
      const width = 46, height = 7;
      const scrX = cam.ox + tank.x * cam.s;
      const scrY = cam.oy + tank.y * cam.s;
      const x = scrX - width / 2;
      const y = scrY - tank.radius * cam.s - 24;
      const maxHp = tank.maxHealth || MAX_HEALTH;   // survival_v2 doubles HP -> bar + per-HP ticks must follow the real pool
      const ratio = Math.max(0, Math.min(1, tank.health / maxHp));
      ctx.save();
      ctx.fillStyle = "rgba(4, 6, 10, 0.82)";
      ctx.fillRect(x - 1, y - 1, width + 2, height + 2);
      // colour by remaining fraction (green->amber->red) so the v2 long fights read at a glance
      ctx.fillStyle = ratio > 0.5 ? (tank.id === 0 ? "#58a6ff" : "#ff6b6b") : ratio > 0.25 ? "#f0b132" : "#e5484d";
      ctx.fillRect(x, y, width * ratio, height);
      ctx.strokeStyle = "rgba(255,255,255,0.62)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
      ctx.strokeStyle = "rgba(255,255,255,0.42)";
      for (let i = 1; i < maxHp; i++) {   // one tick per HP point (6 segments under v2, 3 under v1)
        const sx = x + (width * i) / maxHp;
        ctx.beginPath(); ctx.moveTo(sx, y + 1); ctx.lineTo(sx, y + height - 1); ctx.stroke();
      }
      // Ammo bar: up to ammoLimit (5) live shells. Each segment is a shot; green =
      // available, dark = a live shell still occupying that slot. Empties as you fire
      // and refills as shells despawn/hit -> this IS the "fired all 5" cooldown.
      const ammoLimit = tank.ammoLimit || 5;
      const liveShells = view.shells.filter((s) => s.owner === tank.id).length;
      const available = Math.max(0, ammoLimit - liveShells);
      const ammoY = y + height + 3, ammoH = 4, gap = 2;
      const segW = (width - gap * (ammoLimit - 1)) / ammoLimit;
      for (let i = 0; i < ammoLimit; i++) {
        const sx = x + i * (segW + gap);
        ctx.fillStyle = "rgba(4, 6, 10, 0.82)";
        ctx.fillRect(sx - 0.5, ammoY - 1, segW + 1, ammoH + 2);
        ctx.fillStyle = i < available ? "#7cfc84" : "#3a2330";
        ctx.fillRect(sx, ammoY, segW, ammoH);
      }
      if (tank.shield > 0) {
        ctx.fillStyle = "#56d6c9";
        for (let i = 0; i < tank.shield; i++) {
          ctx.beginPath(); ctx.arc(x + width + 7 + i * 7, y + height / 2, 2.4, 0, TAU); ctx.fill();
        }
      }
      ctx.restore();
    }
  }

  function drawPowerSkin(tank) {
    if (!tank.power) return;
    if (tank.power === "rapid") {
      ctx.strokeStyle = "#f2c94c"; ctx.lineWidth = 3;
      for (const y of [-10, 0, 10]) { ctx.beginPath(); ctx.moveTo(-15, y); ctx.lineTo(15, y - 5); ctx.stroke(); }
      ctx.fillStyle = "#f2c94c"; ctx.beginPath(); ctx.arc(24, 0, 4, 0, TAU); ctx.fill();
    } else if (tank.power === "triple") {
      ctx.fillStyle = tank.accent; ctx.strokeStyle = "#090b10"; ctx.lineWidth = 2;
      for (const y of [-11, 0, 11]) { ctx.fillRect(2, y - 3, 30, 6); ctx.strokeRect(2, y - 3, 30, 6); }
      ctx.fillStyle = "#ffe9b8";
      for (const [x, y] of [[24, -10], [31, -4], [27, 5], [34, 11]]) { ctx.beginPath(); ctx.arc(x, y, 2.4, 0, TAU); ctx.fill(); }
    } else if (tank.power === "bounce") {
      ctx.fillStyle = "#c792ea"; ctx.strokeStyle = "#090b10"; ctx.lineWidth = 2;
      ctx.fillRect(0, -3, 44, 6); ctx.strokeRect(0, -3, 44, 6);
      ctx.strokeStyle = "#f2e7ff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(25, 0, 8, 0, TAU);
      ctx.moveTo(25, -9); ctx.lineTo(25, 9); ctx.moveTo(17, 0); ctx.lineTo(35, 0); ctx.stroke();
    } else if (tank.power === "laser") {
      ctx.fillStyle = "#ff4fd8"; ctx.strokeStyle = "#090b10"; ctx.lineWidth = 2;
      ctx.fillRect(0, -4, 36, 8); ctx.strokeRect(0, -4, 36, 8);
      ctx.fillStyle = "#fff2fb"; ctx.fillRect(27, -2, 9, 4);
      ctx.strokeStyle = "rgba(255, 79, 216, 0.75)"; ctx.beginPath(); ctx.arc(0, 0, 24, -0.45, 0.45); ctx.stroke();
    } else if (tank.power === "missile") {
      ctx.fillStyle = "#ff9f43"; ctx.strokeStyle = "#090b10"; ctx.lineWidth = 2;
      for (const y of [-12, 12]) {
        ctx.beginPath(); ctx.moveTo(2, y - 5); ctx.lineTo(28, y); ctx.lineTo(2, y + 5); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      ctx.fillStyle = "#ffd7a0"; ctx.fillRect(-14, -4, 12, 8);
    } else if (tank.power === "shield") {
      ctx.fillStyle = "rgba(86, 214, 201, 0.35)"; ctx.beginPath(); ctx.arc(0, 0, 19, 0, TAU); ctx.fill();
      ctx.fillStyle = "#56d6c9"; ctx.strokeStyle = "#090b10"; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(-1, -13); ctx.lineTo(11, -7); ctx.lineTo(8, 5);
      ctx.quadraticCurveTo(5, 13, -1, 16); ctx.quadraticCurveTo(-7, 13, -10, 5);
      ctx.lineTo(-13, -7); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  }

  function drawShells() {
    for (const shell of view.shells) {
      ctx.fillStyle = shell.type === "missile" ? "#ff9f43" : shell.type === "snipe" ? "#f2e7ff" : shell.owner === 0 ? "#b7dcff" : "#ffd0d0";
      ctx.beginPath(); ctx.arc(shell.x, shell.y, shell.radius, 0, TAU); ctx.fill();
      ctx.strokeStyle = shell.type === "snipe" ? "#c792ea" : "#090b10";
      ctx.lineWidth = 2; ctx.stroke();
      ctx.strokeStyle = shell.type === "snipe" ? "rgba(199, 146, 234, 0.55)" : "rgba(242, 201, 76, 0.25)";
      ctx.beginPath();
      ctx.moveTo(shell.x, shell.y);
      ctx.lineTo(shell.x - shell.vx * (shell.type === "snipe" ? 0.09 : 0.045), shell.y - shell.vy * (shell.type === "snipe" ? 0.09 : 0.045));
      ctx.stroke();
      if (shell.type === "missile") {
        ctx.strokeStyle = "rgba(255, 159, 67, 0.45)";
        ctx.beginPath(); ctx.moveTo(shell.x, shell.y); ctx.lineTo(shell.x - shell.vx * 0.07, shell.y - shell.vy * 0.07); ctx.stroke();
      }
    }
  }

  function drawBeams() {
    for (const beam of beams) {
      const alpha = Math.max(0, Math.min(1, beam.ttl / 0.16));
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.lineCap = "round";
      for (const segment of beam.path) {
        ctx.strokeStyle = "rgba(255, 79, 216, 0.28)"; ctx.lineWidth = 10;
        ctx.beginPath(); ctx.moveTo(segment.x1, segment.y1); ctx.lineTo(segment.x2, segment.y2); ctx.stroke();
        ctx.strokeStyle = "#fff2fb"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(segment.x1, segment.y1); ctx.lineTo(segment.x2, segment.y2); ctx.stroke();
      }
      ctx.restore();
    }
  }

  function drawExplosions() {
    for (const p of explosions) {
      const alpha = Math.max(0, Math.min(1, p.ttl / p.maxTtl));
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = p.color;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.radius * (1.2 - alpha * 0.2), 0, TAU); ctx.fill();
      ctx.restore();
    }
  }

  function drawPowerupIcon(power) {
    ctx.save();
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.fillStyle = "#101318"; ctx.strokeStyle = "#101318"; ctx.lineWidth = 2.2;
    if (power.type === "bounce") {
      ctx.beginPath(); ctx.arc(0, 0, 7.5, 0, TAU); ctx.stroke();
      ctx.beginPath(); ctx.arc(0, 0, 2.1, 0, TAU); ctx.fill();
      ctx.beginPath();
      ctx.moveTo(-12, 0); ctx.lineTo(-6.5, 0); ctx.moveTo(6.5, 0); ctx.lineTo(12, 0);
      ctx.moveTo(0, -12); ctx.lineTo(0, -6.5); ctx.moveTo(0, 6.5); ctx.lineTo(0, 12); ctx.stroke();
    } else if (power.type === "shield") {
      ctx.beginPath();
      ctx.moveTo(0, -11); ctx.lineTo(9, -6.5); ctx.lineTo(6.5, 3.5);
      ctx.quadraticCurveTo(4, 9, 0, 11.5); ctx.quadraticCurveTo(-4, 9, -6.5, 3.5);
      ctx.lineTo(-9, -6.5); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.beginPath(); ctx.moveTo(0, -7); ctx.lineTo(0, 7.5); ctx.stroke();
    } else if (power.type === "triple") {
      for (const [x, y, r] of [[-6.5, -5.5, 2.4], [0, -7.5, 2.1], [6.5, -4.5, 2.4], [-4.5, 2, 2.1], [3.5, 2.5, 2.5], [0, 7.5, 2.2]]) {
        ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
      }
    } else if (power.type === "missile") {
      ctx.beginPath();
      ctx.moveTo(-9.5, -4.5); ctx.lineTo(3.5, -4.5);
      ctx.quadraticCurveTo(8.5, -3.5, 11, 0); ctx.quadraticCurveTo(8.5, 3.5, 3.5, 4.5);
      ctx.lineTo(-9.5, 4.5); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.moveTo(-8.5, -4); ctx.lineTo(-13, -8); ctx.lineTo(-11.5, -1.5); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.moveTo(-8.5, 4); ctx.lineTo(-13, 8); ctx.lineTo(-11.5, 1.5); ctx.closePath(); ctx.fill();
    } else if (power.type === "rapid") {
      ctx.beginPath();
      ctx.moveTo(-2.5, -10.5); ctx.lineTo(7, -1.5); ctx.lineTo(1.5, -1.5);
      ctx.lineTo(5.5, 10.5); ctx.lineTo(-7.5, 0.5); ctx.lineTo(-1.5, 0.5); ctx.closePath(); ctx.fill();
    } else if (power.type === "laser") {
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(-10, 0); ctx.lineTo(9.5, 0); ctx.stroke();
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(4, -7); ctx.lineTo(10.5, 0); ctx.lineTo(4, 7); ctx.stroke();
      ctx.beginPath(); ctx.arc(-8, 0, 2.4, 0, TAU); ctx.fill();
    }
    ctx.restore();
  }

  function drawPowerups() {
    for (const power of view.powerups) {
      ctx.save();
      ctx.translate(power.x, power.y);
      ctx.fillStyle = power.color;
      ctx.strokeStyle = "#090b10";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(0, 0, power.radius, 0, TAU); ctx.fill(); ctx.stroke();
      drawPowerupIcon(power);
      ctx.restore();
    }
  }

  function drawOverlay() {
    if (view.scenario === "nav_powerup_poison" && view.nav) {
      const n = view.nav, p = view.poison;
      const status = p && p.atMinCircle
        ? `MIN CIRCLE — survive ${n.survivalAfterMinCircle.toFixed(1)} / 12s`
        : (p && p.active ? "poison ring shrinking…" : "poison: waiting");
      ctx.save();
      ctx.fillStyle = "rgba(8,10,16,0.72)";
      roundedRect(12, 12, 330, 50, 8); ctx.fill();
      ctx.fillStyle = p && p.atMinCircle ? "#7cfc84" : "#e6edf3";
      ctx.font = "700 15px system-ui, sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("Lesson 1 · " + status, 24, 30);
      ctx.fillStyle = "rgba(230,237,243,0.8)";
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillText(`pickups ${n.pickups}  ·  wallHits ${n.wallHits}  ·  poisonDmg ${n.poisonDamageTaken.toFixed(1)}`, 24, 48);
      ctx.restore();
    }
    if ((view.scenario === "nav_route_to_center" || view.scenario === "moba_poison_run") && view.nav) {
      const n = view.nav;
      const pd = Number.isFinite(n.pathDist) ? n.pathDist : "?";
      const bpd = Number.isFinite(n.bestPathDist) ? n.bestPathDist : "?";
      const isPoisonRun = view.scenario === "moba_poison_run";
      const label = isPoisonRun ? "Poison Run" : "Lesson 1b";
      const pz = isPoisonRun && view.poison && view.poison.active ? " · ☠ ring" : "";
      const status = n.enteredCenter ? `IN CENTRE — stay ${n.centerStayTime.toFixed(1)} / 2s` : `to centre · pathDist ${pd}${pz}`;
      ctx.save();
      ctx.fillStyle = "rgba(8,10,16,0.72)";
      roundedRect(12, 12, 360, 50, 8); ctx.fill();
      ctx.fillStyle = n.enteredCenter ? "#7cfc84" : (pz ? "#ffb86b" : "#e6edf3");
      ctx.font = "700 15px system-ui, sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(label + " · " + status, 24, 30);
      ctx.fillStyle = "rgba(230,237,243,0.8)";
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillText(`newCells ${n.newCells}  ·  wallHits ${n.wallHits}  ·  stuck ${n.stuckEvents}  ·  best ${bpd}`, 24, 48);
      ctx.restore();
    }
    if (view.scenario === "fixed_moba") {
      ctx.save();
      ctx.fillStyle = "rgba(8,10,16,0.72)";
      roundedRect(12, 12, 300, 50, 8); ctx.fill();
      ctx.fillStyle = "#e6edf3";
      ctx.font = "700 15px system-ui, sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("Fixed Duel · symmetric arena", 24, 30);
      ctx.fillStyle = "rgba(230,237,243,0.8)";
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillText("map: fixed symmetric · open centre", 24, 48);
      ctx.restore();
    }
    if (view.scenario === "moba1v1duel") {
      const p = view.poison;
      const status = p && p.atMinCircle ? "final ring" : (p && p.active ? "poison shrinking…" : "poison: waiting");
      ctx.save();
      ctx.fillStyle = "rgba(8,10,16,0.72)";
      roundedRect(12, 12, 330, 50, 8); ctx.fill();
      ctx.fillStyle = p && p.active ? "#ffb86b" : "#e6edf3";
      ctx.font = "700 15px system-ui, sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("1v1 Duel · " + status, 24, 30);
      ctx.fillStyle = "rgba(230,237,243,0.8)";
      ctx.font = "600 12px system-ui, sans-serif";
      ctx.fillText("poison + breathing regen · Rapid 5-shot 2x · slow turn", 24, 48);
      ctx.restore();
    }
    if (view.roundState && view.roundState.phase === "pending") {
      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.56)";
      roundedRect(312, 18, 336, 48, 8);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "800 18px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`Draw window: ${Math.max(0, view.roundState.timer).toFixed(1)}s`, W / 2, 42);
      ctx.restore();
    }
    if (app.roundMessageTimer > 0) {
      ctx.fillStyle = "rgba(0,0,0,0.32)";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#fff";
      ctx.font = "800 44px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(app.roundMessage, W / 2, H / 2);
    }
  }

  function updateHud() {
    p1ScoreEl.textContent = app.scores[0];
    p2ScoreEl.textContent = app.scores[1];
    const blueLabel = app.matchActive ? app.blueCtrl : app.watch.enabled ? "live" : app.blueController;
    const redLabel = app.matchActive ? app.redCtrl : !app.botEnabled ? "off" : app.botType === "neural" && neuralReady() ? app.botModel : app.botType;
    modeLabel.textContent = `B:${blueLabel} / R:${redLabel} / ${app.speedMultiplier}x`;
    const next = view.powerups[0];
    powerupLabel.textContent = view.constants && view.constants.spawnPowerups === false
      ? "off"
      : next ? next.label : `${Math.ceil(view.powerupTimer)}s`;
  }

  // ----------------------------- input -----------------------------
  window.addEventListener("keydown", (event) => {
    ensureAudio();
    const key = event.key === " " ? " " : event.key.length === 1 ? event.key.toLowerCase() : event.key;
    keys.add(key);
    if ([" ", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Enter"].includes(event.key)) event.preventDefault();
    if (app.record.enabled) {   // record=human: R/N/E/G/B drive the recorder (WASD+Space still control blue)
      if (key === "r") { app.record.recording = !app.record.recording; updateHud(); return; }
      if (key === "n") { newRecordEpisode(); return; }
      if (key === "e") { exportJSONL(); return; }
      if (key === "g") { app.record.goodDemo = true; updateHud(); return; }
      if (key === "b") { app.record.goodDemo = false; updateHud(); return; }
    }
    if (key === "b") { app.botEnabled = !app.botEnabled; updateHud(); }
    if (key === "k") {
      const order = ["god", "blue", "red"];
      app.cameraMode = order[(order.indexOf(app.cameraMode) + 1) % order.length];
      updateHud();
    }
    if (key === "m") {
      const scripted = ["laika", "laika-aggressive-pro", "easy_laika", "laika-aggressive", "turret", "stationary"];
      const options = [...scripted, ...availableModels()];
      const current = app.botType === "neural" ? app.botModel : app.botType;
      const next = options[(options.indexOf(current) + 1) % options.length];
      if (scripted.includes(next)) app.botType = next;
      else { app.botType = "neural"; app.botModel = next; }
      updateHud();
    }
    if (key === "v") {
      const options = ["human", ...availableModels(), "laika", "laika-aggressive-pro", "laika-aggressive", "pathfind"];
      app.blueController = options[(options.indexOf(app.blueController) + 1) % options.length];
      updateHud();
    }
    if (key === "c") app.navGoal = null;
    if (key === "n") { core.reset(); view = core.getPublicState(); app.navGoal = null; arenaSeedEl.textContent = String(view.seed); resetRoundUi(); }
    if (key === "r") { core.reset(view.seed); view = core.getPublicState(); app.navGoal = null; resetRoundUi(); }
    if (key === "]" || key === "=") {
      const speeds = [0.5, 1, 2, 4, 8, 16];
      const index = speeds.indexOf(app.speedMultiplier);
      app.speedMultiplier = speeds[Math.min(speeds.length - 1, Math.max(0, index) + 1)];
      updateHud();
    }
    if (key === "[" || key === "-") {
      const speeds = [0.5, 1, 2, 4, 8, 16];
      const index = speeds.indexOf(app.speedMultiplier);
      app.speedMultiplier = speeds[Math.max(0, (index < 0 ? 1 : index) - 1)];
      updateHud();
    }
  });
  window.addEventListener("keyup", (event) => {
    const key = event.key === " " ? " " : event.key.length === 1 ? event.key.toLowerCase() : event.key;
    keys.delete(key);
  });
  canvas.addEventListener("pointerdown", (event) => {
    ensureAudio();
    canvas.focus();
    if (app.blueController === "pathfind") {
      const rect = canvas.getBoundingClientRect();
      app.navGoal = {
        x: (event.clientX - rect.left) * (canvas.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.height / rect.height)
      };
    }
  });

  function resetRoundUi() {
    app.pendingReset = 0;
    app.roundMessage = "";
    app.roundMessageTimer = 0;
    explosions.length = 0;
    beams.length = 0;
    updateHud();
  }

  // --------------------------- watch mode --------------------------
  function initWatchMode() {
    const params = new URLSearchParams(window.location.search);
    const w = params.get("watch");
    const playParam = params.get("play");            // play=1/agent -> YOU battle the agent live
    if (!w && !playParam) return;
    app.watch.enabled = true;                         // reuses the RL-cadence step + real-time accumulator + policy loader
    if (w && w !== "1" && w !== "true") app.watch.runDir = w;
    else if (playParam) app.watch.runDir = "runs/auto_live";   // load the deployed policies so neural agents are pickable
    else app.watch.runDir = null;   // ?watch=1 with no dir -> spectate the deployed in-memory agent (no live-training poll)
    app.play.enabled = !!playParam;
    if (app.play.enabled) {
      app.play.humanSide = params.get("side") === "blue" ? 0 : 1;   // default: YOU = red (agent = blue, full strength)
      app.speedMultiplier = 1;                                       // real-time (no fast-forward when a human is playing)
    }
    const speedParam = parseFloat(params.get("speed"));              // watch-mode preset speed, e.g. ?speed=4
    if (!app.play.enabled && [0.5, 1, 2, 4, 8, 16].includes(speedParam)) app.speedMultiplier = speedParam;
    app.botEnabled = !app.play.enabled;               // play mode: red is YOU (a human), not a script
    // Default to the v1.5 NEURAL clone when its weights are loaded (the random-spawn agent), else the pure-aggression
    // script champion. Both are 105-d-obs compatible; old 101-d neural models are not, so they are not offered.
    app.liveModel = (window.RICOCHET_POLICIES && window.RICOCHET_POLICIES["v15clone"]) ? "v15clone" : "aggro";
    const red = params.get("red");
    app.botType = red === "laika" ? "laika"
      : red === "laika-aggressive-pro" || red === "pro" ? "laika-aggressive-pro"
      : red === "laika-aggressive" || red === "aggressive" ? "laika-aggressive"
      : red === "easy_laika" ? "easy_laika"
      : red === "stationary" || red === "idle" ? "stationary"
      : red === "none" ? "none"
      : red === "turret" || red === "slow-turret" || red === "turret-slow" ? "turret"
      : "laika";

    // ===== a fixed TOP BAR: 地图 Map / 蓝方 Blue / 红方 Red — each side = 玩家(player) / 智能体(agent) / 脚本(rule_based) =====
    app.matchActive = true;
    const BEST = (window.RICOCHET_POLICIES && window.RICOCHET_POLICIES["v15clone"]) ? "v15clone" : "aggro";
    if (app.play.enabled) {                              // legacy ?play link -> the human drives one side, the agent the other
      app.blueCtrl = app.play.humanSide === 0 ? "player" : BEST;
      app.redCtrl = app.play.humanSide === 1 ? "player" : BEST;
    } else if (app.watch.runDir) {                       // legacy live-training watch (index.html dev): blue = the polled "live" policy
      app.blueCtrl = "live"; app.redCtrl = app.botType;
    } else {                                             // static watch: agent (blue) vs the ?red script (default laika)
      app.blueCtrl = params.get("blue") || BEST; app.redCtrl = app.botType;
    }
    const mkSel = (label, optList, val, on) => {
      const wrap = document.createElement("label");
      wrap.style.cssText = "display:flex;gap:6px;align-items:center;white-space:nowrap;";
      wrap.append(label + ":");
      const s = document.createElement("select");
      s.style.cssText = "background:#10131c;color:#cfe;border:1px solid rgba(255,255,255,0.25);border-radius:5px;padding:3px 6px;font:500 12px system-ui;max-width:210px;";
      let matched = false;
      optList.forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; if (v === val) { o.selected = true; matched = true; } s.appendChild(o); });
      if (!matched && val != null) {                       // never mis-display: show the REAL controller even if it isn't a listed option (live / none / turret / aggro)
        const o = document.createElement("option");
        o.value = val; o.textContent = val === "live" ? "实时训练 live" : val === "aggro" ? "laika-aggressive 脚本" : String(val);
        o.selected = true; s.insertBefore(o, s.firstChild);
      }
      s.onchange = () => on(s.value);
      wrap.appendChild(s); return wrap;
    };
    const resetRound = () => { app.scores = [0, 0]; app.router = null; core.reset(); view = core.getPublicState(); app.pendingReset = 0; app.roundMessage = ""; app.watchAccum = 0; arenaSeedEl.textContent = String(view.seed); updateHud(); };
    // controller options shared by 蓝方/红方: 玩家(player) + 智能体(agents) + 脚本(rule_based), each suffixed 脚本)
    // The neural 智能体 are trained ONLY as BLUE (seat 0); their obs include absolute board position, so on the RED
    // seat (right side) they run out-of-distribution and play erratically — NOT a faithful demo. So Red offers only
    // 玩家 + 脚本 (scripts are computed live and seat-symmetric); Blue (the showcase side) keeps the agents.
    const SCRIPTS = [["laika", "laika"], ["laika-aggressive", "laika-aggressive"], ["laika-aggressive-pro", "pro"],
      ["easy_laika", "easy_laika"], ["stationary", "stationary"], ["ace", "ace"], ["charger", "charger"]]
      .map(([v, t]) => [v, t + " 脚本"]);
    const AGENTS = [];
    if (window.RICOCHET_POLICIES) {
      if (window.RICOCHET_POLICIES["v15clone"]) AGENTS.push(["v15clone", "★ v1.5 随机出生专用 agent"]);
      if (window.RICOCHET_POLICIES["v3champ"]) AGENTS.push(["v3champ", "★ 固定出生点冠军 agent"]);
    }
    const ctrlOptsBlue = [["player", "玩家 Player"], ...AGENTS, ...SCRIPTS];
    const ctrlOptsRed = [["player", "玩家 Player"], ...SCRIPTS];   // no 智能体 on red (seat-1 OOD)
    const mapOpts = [["tri_fixed", "1v1 duel · 3points_randomSpawn"], ["half_random", "1v1 duel · Random Spawn(half)"], ["fixed", "1v1 duel · Fixed spawn"], ["full_random", "1v1 duel · Random spawn (full)"]];

    const bar = document.createElement("div");
    bar.id = "matchBar";
    bar.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:9999;display:flex;flex-wrap:wrap;row-gap:6px;" +
      "column-gap:16px;align-items:center;justify-content:center;background:rgba(10,12,18,0.96);color:#cfe;" +
      "padding:8px 14px;border-bottom:1px solid rgba(255,255,255,0.14);font:600 12px system-ui,sans-serif;";
    document.body.style.paddingTop = "46px";            // keep page content clear of the fixed bar
    const onPlayerPick = () => { if (app.blueCtrl === "player" || app.redCtrl === "player") app.speedMultiplier = 1; };
    // nav lessons (red removed) aren't duels: skip Map (it would navigate out) + Red (inert). Match the core's
    // spawn default (fixed for survival_v1, half_random for survival_v2) so the Map label reflects what actually runs.
    if (!isNavScenario) bar.appendChild(mkSel("地图 Map", mapOpts, (params.get("spawn") || (ruleset === "survival_v2" ? "half_random" : "fixed")),
      (v) => { const p = new URLSearchParams(location.search); p.set("scenario", "moba1v1duel"); p.set("spawn", v); location.search = p.toString(); }));
    bar.appendChild(mkSel("蓝方 Blue", ctrlOptsBlue, app.blueCtrl, (v) => { app.blueCtrl = v; onPlayerPick(); resetRound(); }));
    if (!isNavScenario) bar.appendChild(mkSel("红方 Red", ctrlOptsRed, app.redCtrl, (v) => { app.redCtrl = v; onPlayerPick(); resetRound(); }));
    const rb = document.createElement("button");
    rb.textContent = "restart";
    rb.style.cssText = "background:#10131c;color:#cfe;border:1px solid rgba(255,255,255,0.25);border-radius:5px;padding:4px 10px;font:600 12px system-ui;cursor:pointer;";
    rb.onclick = resetRound;
    bar.appendChild(rb);
    const wr = document.createElement("span"); wr.style.cssText = "color:#9fe;min-width:140px;"; bar.appendChild(wr);
    const note = document.createElement("span"); note.style.cssText = "color:#7f93a8;font-weight:400;";
    note.textContent = "K 切换视角 · [ ] 调速度 · 玩家先点场地捕获按键 (WASD=蓝 · 方向键+Enter=红)"; bar.appendChild(note);
    const banner = document.createElement("span"); banner.id = "watchBanner"; banner.style.cssText = "color:#7f93a8;font-weight:400;"; bar.appendChild(banner);
    document.body.appendChild(bar);
    setInterval(() => {                                  // live win-rate readout (blue:red)
      const b = app.scores[0], r = app.scores[1], t = b + r;
      wr.textContent = `blue ${b} : ${r} red` + (t ? `  (${Math.round(100 * b / t)}% blue)` : "");
    }, 400);
    // strip redundant clutter for a clean watch view: the Maps link list + the keyboard Controls block
    document.querySelectorAll(".panel-block").forEach((blk) => {
      const h = blk.querySelector("h2");
      if (h && (h.textContent === "Maps" || h.textContent === "Controls")) blk.style.display = "none";
    });

    const loadLiveScript = (name) => new Promise((resolve) => {
      const script = document.createElement("script");
      script.src = `${app.watch.runDir}/${name}?t=${Date.now()}`;
      script.onload = () => { script.remove(); resolve(true); };
      script.onerror = () => { script.remove(); resolve(false); };
      document.head.appendChild(script);
    });

    const poll = async () => {
      const bust = "?t=" + Date.now();
      await loadLiveScript("live_policy.js");
      await loadLiveScript("live_status.js");
      const pol = await fetch(app.watch.runDir + "/live_policy.json" + bust).then((r) => (r.ok ? r.json() : null)).catch(() => null);
      const st = await fetch(app.watch.runDir + "/live_status.json" + bust).then((r) => (r.ok ? r.json() : null)).catch(() => null);
      if (pol && pol.layers) {
        window.RICOCHET_POLICIES = window.RICOCHET_POLICIES || {};
        window.RICOCHET_POLICIES.live = pol;
      }
      const liveStatus = window.RICOCHET_LIVE_STATUS || st;
      if (liveStatus && !app.play.enabled) {   // play mode keeps its own "You vs Agent" banner
        const rew = liveStatus.ep_rew_mean == null ? "--" : liveStatus.ep_rew_mean.toFixed(2);
        const len = liveStatus.ep_len_mean == null ? "--" : liveStatus.ep_len_mean.toFixed(0);
        const phase = liveStatus.phase ? `  |  ${liveStatus.phase}` : "";
        banner.textContent =
          `watch ${app.watch.runDir}${phase}  |  steps ${liveStatus.timesteps}  |  ep_rew ${rew}` +
          `  |  ep_len ${len}  |  vs ${liveStatus.opponent}  |  speed ${app.speedMultiplier}x` +
          `  |  score B${app.scores[0]}-R${app.scores[1]}`;
      }
    };
    if (app.watch.runDir) { poll(); setInterval(poll, app.watch.pollMs); }   // live-training watch only; a static deployed-agent watch needs no poll
  }

  // ----------------------------- boot ------------------------------
  core.reset(1307);
  view = core.getPublicState();
  arenaSeedEl.textContent = String(view.seed);

  const models = availableModels();
  if (models.length) {
    app.botModel = models.includes("defensive") ? "defensive" : models[0];
    app.botType = "neural";
  }
  initWatchMode();
  if (urlParams.get("red") === "laika") app.botType = "laika";
  if (urlParams.get("red") === "laika-aggressive-pro") app.botType = "laika-aggressive-pro";
  if (urlParams.get("red") === "easy_laika") app.botType = "easy_laika";
  if (urlParams.get("red") === "idle" || urlParams.get("red") === "stationary") app.botType = "stationary";
  if (urlParams.get("red") === "none") app.botType = "none";
  if (urlParams.get("red") === "heuristic") app.botType = "laika";
  if (["turret", "slow-turret", "turret-slow"].includes(urlParams.get("red"))) app.botType = "turret";
  if (urlParams.get("blue") === "aggro") app.blueController = "laika-aggressive";
  if (urlParams.get("nav")) { app.blueController = "pathfind"; app.botType = "stationary"; }
  if (isNavScenario) app.botType = "none";   // nav lessons: red is removed
  if (urlParams.get("record") === "human") {   // human demonstration recording mode
    app.record.enabled = true;
    app.watch.enabled = false;
    app.blueController = "human";     // blue is YOU
    app.botEnabled = true;
    app.botType = "easy_laika";       // record against the stage-1 opponent
    app.speedMultiplier = 1;
    window.RICOCHET_RECORD = {
      toggle: () => { app.record.recording = !app.record.recording; updateHud(); },
      newEpisode: newRecordEpisode, export: exportJSONL,
      getState: () => ({ enabled: app.record.enabled, recording: app.record.recording, episode: app.record.episode,
        step: app.record.step, lastActionId: app.record.lastActionId, lastResult: app.record.lastResult,
        transitions: app.record.lines.filter((l) => l.type === "transition").length, lines: app.record.lines.length }),
    };
    const bar = document.createElement("div");
    bar.style.cssText = "position:fixed;bottom:10px;left:50%;transform:translateX(-50%);z-index:9999;display:flex;gap:8px;align-items:center;" +
      "background:rgba(10,12,18,0.92);padding:8px 12px;border-radius:10px;border:1px solid rgba(255,255,255,0.15);font:600 12px system-ui,sans-serif;color:#cfe;";
    const mkBtn = (label, fn) => { const b = document.createElement("button"); b.textContent = label;
      b.style.cssText = "cursor:pointer;padding:5px 10px;border-radius:6px;border:1px solid #2a3344;background:#1b2230;color:#cfe;font:600 12px system-ui;";
      b.onmousedown = (e) => e.preventDefault(); b.onclick = () => { fn(); canvas.focus(); }; return b; };
    bar.appendChild(document.createTextNode("REC: WASD+Space · "));
    bar.appendChild(mkBtn("● rec/pause (R)", () => { app.record.recording = !app.record.recording; updateHud(); }));
    bar.appendChild(mkBtn("next ep (N)", newRecordEpisode));
    bar.appendChild(mkBtn("export JSONL (E)", exportJSONL));
    document.body.appendChild(bar);
  }
  app.cameraMode = normalizeCamera(cameraParam) || (app.watch.enabled ? "god" : "blue");

  updateHud();
  canvas.focus();
  requestAnimationFrame(frame);
})();
