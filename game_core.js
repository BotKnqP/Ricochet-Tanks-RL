// game_core.js — headless-capable Ricochet Tanks engine (single source of truth).
//
// Contains ONLY simulation: maze, tanks, movement, shells/laser/missile, walls,
// collisions, power-ups, health/round resolution, scripted bots, the 105-d neural
// observation, and the RL reward. NO canvas, DOM, audio, input, or RAF — those
// live in game_render.js. The browser and the Python/Node training bridge both
// load THIS file, so there is exactly one physics implementation.
//
// UMD: in Node `require("./game_core.js")` -> { RicochetCore, ... };
//      in the browser `<script>` -> window.RicochetCore / window.RicochetCoreModule.

(function (root, factory) {
  "use strict";
  const mod = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = mod;
  if (typeof root !== "undefined") {
    root.RicochetCore = mod.RicochetCore;
    root.RicochetCoreModule = mod;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // --- Arena / physics constants (W/H are literals; no canvas dependency) ---
  const W = 960;
  const H = 640;
  const WALL = 12;
  const COLS = 8;
  const ROWS = 6;
  const CELL_W = W / COLS;
  const CELL_H = H / ROWS;
  const TAU = Math.PI * 2;
  const TANK_RADIUS = 13.6;
  const TANK_TURN_SPEED = 3.96;
  // moba1v1duel gameplay tweaks (scoped to that scenario only; every other mode is
  // unaffected): breathing regen, a 5-shot 2x-speed burst on the Rapid (lightning)
  // skill ONLY (all other power-ups keep their vanilla 2 shots), and -30% turn rate.
  const MOBA_DUEL = { regenDelay: 5.0, regenRate: 0.5, rapidShots: 5, turnScale: 0.7 };
  // easy_laika: stage-1 combat-curriculum opponent. Full laika brain (poison escape,
  // wall-aware pathing, shell dodging) but heavily dampened — slow move/turn, rare
  // imprecise fire, no hard chase — so a poison-run model can ease into combat.
  const EASY_LAIKA = {
    moveScale: 0.45, turnScale: 0.55, aggression: 0.25, fireCooldownMult: 3.0,
    fireAngle: 0.35, preferredDistance: 260, retreatHealth: 1.0, poisonEscapePriority: true
  };
  // laika-aggressive-pro: the strongest scripted expert for moba1v1duel — the behaviour-
  // cloning mother-distribution. Distance bands, safe-fire aiming (direct + bounce),
  // dangerous-shell dodging, poison-zone control, low-health retreat + shield-seeking,
  // and value powerup pickup. Built on the existing laika helpers; it never self-hits
  // because it only fires through laikaCanFireNow's muzzle/self-bounce safety check.
  const LAIKA_PRO = {
    preferLo: 220, preferHi: 320, tooClose: 160, tooFar: 420,
    fireAngle: 0.16, fireAngleBounce: 0.12, dodgeUrgency: 0.55, retreatHealth: 1.0,
    puGrabDist: 300, shieldGrabDist: 340
  };
  const TANK_MOVE_SPEED = 146.4;
  const MAX_HEALTH = 3;
  const BASE_SHELL_SPEED = TANK_MOVE_SPEED * 1.32;
  const SHELL_SPEED = BASE_SHELL_SPEED * 1.2;   // normal shells +20%
  const TRIPLE_SPEED = BASE_SHELL_SPEED * 2;    // special weapons keep their original speeds
  const MISSILE_SPEED = 339.57;
  const SNIPE_SPEED = BASE_SHELL_SPEED * 4;
  const SHELL_TTL = 18;
  const SHELL_BOOST = 2.0;        // experimental: shells launch at 2x speed...
  const SHELL_DECAY_DURATION = 4.0;  // ...then decelerate linearly to 0 over this many seconds and vanish
  const FIRE_DELAY = 0.24375;
  const RAPID_FIRE_DELAY = 0.0975;
  const LASER_FIRE_DELAY = 0.442;
  const LASER_DAMAGE = 1.5;          // laser is the high-skill aimed beam: 1.5x a normal hit (others deal 1)
  const AMMO_EXHAUSTED_COOLDOWN_MULT = 1.5;
  const POWER_DURATION = 36;        // x2 (Stage 3 power-up duration doubling)
  const SHIELD_DURATION = 54;       // x2
  const POWERUP_FIELD_TTL = 36;     // x2 (field lifetime)

  const DRAW_WINDOW = 3.0;
  const RAY_MAX = 260.0;
  const MAX_SHELL_FEATURES = 3;
  const ARENA_DIAG = Math.hypot(W, H);
  const POWER_TYPES = ["rapid", "triple", "shield", "bounce", "laser", "missile"];
  const WALL_RAY_COUNT = 32;
  const RAY_OFFSETS = Array.from(
    { length: WALL_RAY_COUNT },
    (_, i) => -Math.PI + i * TAU / WALL_RAY_COUNT
  );
  const OBS_SIZE = 73 + RAY_OFFSETS.length;  // 73 scalar features (incl. 11-wide poison + 4 motion/velocity) + wall rays

  // Level 1B open-arena spawn jitter ranges (uniform ±value), applied only when
  // arenaMode === "open" && spawnJitter === true, drawn from the seeded RNG so
  // episodes stay reproducible. Default-off; reported in info/constants.
  const OPEN_JITTER = { blueY: 60, redY: 120, blueAngle: 0.5 };

  // Shooting-lab: when randomTurret is set on an open arena, the turret (red) is dropped at
  // a uniform-random position in this region each episode (seeded RNG -> reproducible), so
  // the expert demonstrates aiming from many distances/angles. Blue starts left-centre
  // facing the turret. Region kept clear of the boundary walls.
  const OPEN_TURRET = { xMin: W * 0.40, xMax: W * 0.92, yMin: H * 0.12, yMax: H * 0.88 };

  // Level 1C low-frequency turret opponent: never moves, rotates to face the enemy,
  // and fires only with line-of-sight + rough alignment, rate-limited by a cooldown.
  // Fully deterministic (no Math.random); cadence driven by the sim clock.
  const TURRET = { cooldown: 2.0, initialDelay: 1.0, fireAngle: 0.18, turnGain: 1.2 };

  // Survival (big-map) geometry. open/maze keep the W×H "view" size; the survival
  // world is 2× each dim (4× area) on a 2× cell grid, with a maze-free central
  // safe zone (~1/4 area) where the poison ring eventually stops.
  const VIEW_W = W, VIEW_H = H;
  const WORLD_W = W * 2, WORLD_H = H * 2;
  const SCOLS = COLS * 2, SROWS = ROWS * 2;
  const WORLD_CENTER_CLEAR = { x: WORLD_W / 4, y: WORLD_H / 4, w: WORLD_W / 2, h: WORLD_H / 2 };

  // Survival poison ring: after startTime the safe rect shrinks inward CONTINUOUSLY all the way to
  // the map centre, so the poison eventually covers the WHOLE map -> no permanent safe zone, no
  // stalemate. Tanks outside the safe rect take dps damage (fractional health); once the ring fully
  // closes both tanks take damage and the lower-health one dies first ("first to die loses").
  // Width/height each close to ~0 over ~75s (1920px / (2*12.8px/s)) — a gentle, gradual ring; a duel
  // still always resolves (combat ends most rounds; the closing ring backstops any stalemate).
  const POISON = {
    startTime: 16.0,
    dps: 0.35,
    shrinkSpeedX: WORLD_W / 150,
    shrinkSpeedY: WORLD_H / 150
  };

  // Powerup display metadata (label/color) is kept here so getPublicState and the
  // pickup events carry it; the renderer never re-derives it.
  const POWERUP_META = {
    rapid: { label: "Rapid", color: "#f2c94c" },
    triple: { label: "Triple", color: "#9bdb7b" },
    shield: { label: "Shield", color: "#56d6c9" },
    bounce: { label: "Snipe", color: "#c792ea" },
    laser: { label: "Laser", color: "#ff4fd8" },
    missile: { label: "Missile", color: "#ff9f43" }
  };

  // throttle, turn, fire — 18-way discrete table (idle + 3x3x2 minus idle dup).
  const ACTION_TABLE = (function makeActionTable() {
    const table = [[0.0, 0.0, false]];
    for (const throttle of [-1.0, 0.0, 1.0]) {
      for (const turn of [-1.0, 0.0, 1.0]) {
        for (const fire of [false, true]) {
          if (throttle === 0 && turn === 0 && !fire) continue;
          table.push([throttle, turn, fire]);
        }
      }
    }
    return table;
  })();

  function actionToControl(action) {
    const [throttle, turn, fire] = ACTION_TABLE[action] || ACTION_TABLE[0];
    return { throttle, turn, fire };
  }

  // Map a {w,a,s,d,space} keyboard snapshot to a {throttle,turn,fire} control. Pure;
  // opposite keys cancel (W+S -> throttle 0, A+D -> turn 0). Used by the browser human
  // demonstration recorder, then funnelled through controlToAction() so a human action
  // becomes the SAME Discrete(18) id the PPO agent uses (never raw key events).
  function keysToControl(keys) {
    keys = keys || {};
    const throttle = keys.w && !keys.s ? 1 : keys.s && !keys.w ? -1 : 0;
    const turn = keys.a && !keys.d ? -1 : keys.d && !keys.a ? 1 : 0;
    const fire = !!keys.space;
    return { throttle, turn, fire };
  }

  function controlToAction(throttle, turn, fire) {
    const t = throttle < -0.33 ? -1.0 : throttle > 0.33 ? 1.0 : 0.0;
    const r = turn < -0.33 ? -1.0 : turn > 0.33 ? 1.0 : 0.0;
    const want = Boolean(fire);
    for (let i = 0; i < ACTION_TABLE.length; i++) {
      const [at, ar, af] = ACTION_TABLE[i];
      if (at === t && ar === r && af === want) return i;
    }
    return 0;
  }

  // --- Pure helpers (no state) ---
  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function angleTo(dx, dy) { return Math.atan2(dy, dx); }
  function angleDiff(a, b) {
    let d = (b - a + Math.PI) % TAU - Math.PI;
    if (d < -Math.PI) d += TAU;
    return d;
  }
  function distSq(a, b) { const dx = a.x - b.x, dy = a.y - b.y; return dx * dx + dy * dy; }
  function circleRectHit(cx, cy, radius, rect) {
    const nx = clamp(cx, rect.x, rect.x + rect.w);
    const ny = clamp(cy, rect.y, rect.y + rect.h);
    const dx = cx - nx, dy = cy - ny;
    return dx * dx + dy * dy <= radius * radius;
  }
  function pointInRect(x, y, rect) {
    return x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h;
  }
  function distanceToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const lenSq = dx * dx + dy * dy || 1;
    const t = clamp(((px - x1) * dx + (py - y1) * dy) / lenSq, 0, 1);
    const x = x1 + dx * t, y = y1 + dy * t;
    return Math.hypot(px - x, py - y);
  }
  function cellCenter(x, y) { return { x: x * CELL_W + CELL_W / 2, y: y * CELL_H + CELL_H / 2 }; }
  function cellOf(px, py) {
    return {
      cx: Math.min(COLS - 1, Math.max(0, Math.floor(px / CELL_W))),
      cy: Math.min(ROWS - 1, Math.max(0, Math.floor(py / CELL_H)))
    };
  }
  function powerOneHot(power) { return POWER_TYPES.map((item) => (power === item ? 1.0 : 0.0)); }
  function isWeaponPower(type) {
    return type === "rapid" || type === "triple" || type === "bounce" || type === "laser" || type === "missile";
  }

  // Deterministic LCG. The whole point of the headless core: same seed => same game.
  class RNG {
    constructor(seed) { this.seed = seed >>> 0; }
    next() { this.seed = (this.seed * 1664525 + 1013904223) >>> 0; return this.seed / 4294967296; }
    int(max) { return Math.floor(this.next() * max); }
    pick(items) { return items[this.int(items.length)]; }
  }

  function makeMaze(seed) {
    const rng = new RNG(seed);
    const cells = [];
    for (let y = 0; y < ROWS; y++) {
      const row = [];
      for (let x = 0; x < COLS; x++) {
        row.push({ x, y, visited: false, walls: { n: true, e: true, s: true, w: true } });
      }
      cells.push(row);
    }
    const stack = [cells[0][0]];
    cells[0][0].visited = true;
    while (stack.length) {
      const current = stack[stack.length - 1];
      const neighbors = [];
      const { x, y } = current;
      if (y > 0 && !cells[y - 1][x].visited) neighbors.push(["n", cells[y - 1][x], "s"]);
      if (x < COLS - 1 && !cells[y][x + 1].visited) neighbors.push(["e", cells[y][x + 1], "w"]);
      if (y < ROWS - 1 && !cells[y + 1][x].visited) neighbors.push(["s", cells[y + 1][x], "n"]);
      if (x > 0 && !cells[y][x - 1].visited) neighbors.push(["w", cells[y][x - 1], "e"]);
      if (!neighbors.length) { stack.pop(); continue; }
      const [dir, next, opposite] = rng.pick(neighbors);
      current.walls[dir] = false;
      next.walls[opposite] = false;
      next.visited = true;
      stack.push(next);
    }
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        if (rng.next() < 0.22) {
          const cell = cells[y][x];
          const options = [];
          if (x < COLS - 1 && cell.walls.e) options.push(["e", cells[y][x + 1], "w"]);
          if (y < ROWS - 1 && cell.walls.s) options.push(["s", cells[y + 1][x], "n"]);
          if (options.length) {
            const [dir, other, opposite] = rng.pick(options);
            cell.walls[dir] = false;
            other.walls[opposite] = false;
          }
        }
      }
    }
    return cells;
  }

  function mazeToRects(cells) {
    const walls = boundaryWalls();
    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const cell = cells[y][x];
        const px = x * CELL_W, py = y * CELL_H;
        if (cell.walls.e && x < COLS - 1) {
          walls.push({ x: px + CELL_W - WALL / 2, y: py - WALL / 2, w: WALL, h: CELL_H + WALL });
        }
        if (cell.walls.s && y < ROWS - 1) {
          walls.push({ x: px - WALL / 2, y: py + CELL_H - WALL / 2, w: CELL_W + WALL, h: WALL });
        }
      }
    }
    return walls;
  }

  function boundaryWalls(w = W, h = H) {
    return [
      { x: 0, y: 0, w: w, h: WALL },
      { x: 0, y: h - WALL, w: w, h: WALL },
      { x: 0, y: 0, w: WALL, h: h },
      { x: w - WALL, y: 0, w: WALL, h: h }
    ];
  }

  // Survival world: 2x grid maze with ~20% fewer internal walls than the base
  // maze; the central safe zone keeps ~50% of the edge wall density. Deterministic.
  function makeSurvivalWorld(seed, wallRemoveProb = 0.20, clearCenter = false) {
    const cols = SCOLS, rows = SROWS;
    const cellW = WORLD_W / cols, cellH = WORLD_H / rows;
    const rng = new RNG(seed);
    const cells = [];
    for (let y = 0; y < rows; y++) {
      const row = [];
      for (let x = 0; x < cols; x++) row.push({ x, y, visited: false, walls: { n: true, e: true, s: true, w: true } });
      cells.push(row);
    }
    const stack = [cells[0][0]];
    cells[0][0].visited = true;
    while (stack.length) {
      const current = stack[stack.length - 1];
      const neighbors = [];
      const { x, y } = current;
      if (y > 0 && !cells[y - 1][x].visited) neighbors.push(["n", cells[y - 1][x], "s"]);
      if (x < cols - 1 && !cells[y][x + 1].visited) neighbors.push(["e", cells[y][x + 1], "w"]);
      if (y < rows - 1 && !cells[y + 1][x].visited) neighbors.push(["s", cells[y + 1][x], "n"]);
      if (x > 0 && !cells[y][x - 1].visited) neighbors.push(["w", cells[y][x - 1], "e"]);
      if (!neighbors.length) { stack.pop(); continue; }
      const [dir, next, opposite] = rng.pick(neighbors);
      current.walls[dir] = false;
      next.walls[opposite] = false;
      next.visited = true;
      stack.push(next);
    }
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (rng.next() < 0.22) {
          const cell = cells[y][x];
          const options = [];
          if (x < cols - 1 && cell.walls.e) options.push(["e", cells[y][x + 1], "w"]);
          if (y < rows - 1 && cell.walls.s) options.push(["s", cells[y + 1][x], "n"]);
          if (options.length) {
            const [dir, other, opposite] = rng.pick(options);
            cell.walls[dir] = false;
            other.walls[opposite] = false;
          }
        }
      }
    }
    const cc = { x: WORLD_W / 4, y: WORLD_H / 4, w: WORLD_W / 2, h: WORLD_H / 2 };
    const insideCC = (rx, ry, rw, rh) => {
      const mx = rx + rw / 2, my = ry + rh / 2;
      return mx > cc.x && mx < cc.x + cc.w && my > cc.y && my < cc.y + cc.h;
    };
    // Density: edge walls keep (1 - wallRemoveProb); the central safe zone keeps
    // only CENTER_KEEP of that (sparser-but-not-empty middle). Mutates cell.walls
    // so pathfinding and the rendered/physical walls stay consistent. rng is
    // consumed identically across wallRemoveProb values for honest comparison.
    const CENTER_KEEP = 0.5;
    const edgeKeep = 1 - wallRemoveProb;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const cell = cells[y][x];
        const px = x * cellW, py = y * cellH;
        if (x < cols - 1 && cell.walls.e) {
          const inCC = insideCC(px + cellW - WALL / 2, py - WALL / 2, WALL, cellH + WALL);
          // clearCenter (lesson 1): keep=-1 so the wall is always removed, but rng is still
          // consumed so the rest of the map is identical to the normal 50%-density world.
          const keep = clearCenter && inCC ? -1 : (inCC ? edgeKeep * CENTER_KEEP : edgeKeep);
          if (rng.next() >= keep) { cell.walls.e = false; cells[y][x + 1].walls.w = false; }
        }
        if (y < rows - 1 && cell.walls.s) {
          const inCC = insideCC(px - WALL / 2, py + cellH - WALL / 2, cellW + WALL, WALL);
          const keep = clearCenter && inCC ? -1 : (inCC ? edgeKeep * CENTER_KEEP : edgeKeep);
          if (rng.next() >= keep) { cell.walls.s = false; cells[y + 1][x].walls.n = false; }
        }
      }
    }
    const walls = boundaryWalls(WORLD_W, WORLD_H);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const cell = cells[y][x];
        const px = x * cellW, py = y * cellH;
        if (cell.walls.e && x < cols - 1) walls.push({ x: px + cellW - WALL / 2, y: py - WALL / 2, w: WALL, h: cellH + WALL });
        if (cell.walls.s && y < rows - 1) walls.push({ x: px - WALL / 2, y: py + cellH - WALL / 2, w: cellW + WALL, h: WALL });
      }
    }
    return { cells, walls, centerClear: cc };
  }

  // Fixed 4-fold-symmetric MOBA arena (scenario "fixed_moba"). Same WORLD_W×WORLD_H
  // world and SCOLS×SROWS grid as survival, but hand-laid instead of a random maze:
  // an open centre (== WORLD_CENTER_CLEAR), a handful of short cover walls per
  // quadrant (no deep dead-ends), and clear main lanes. Built on the SAME cell graph
  // the maze uses, so BFS pathfinding and the physical wall rects stay consistent.
  // The grid starts fully OPEN (no internal walls); cover is ADDED symmetrically, so
  // it is deterministic and ignores the seed entirely.
  function makeFixedMobaWorld() {
    const cols = SCOLS, rows = SROWS;
    const cellW = WORLD_W / cols, cellH = WORLD_H / rows;
    const cells = [];
    for (let y = 0; y < rows; y++) {
      const row = [];
      for (let x = 0; x < cols; x++) row.push({ x, y, visited: true, walls: { n: false, e: false, s: false, w: false } });
      cells.push(row);
    }
    // Edge primitives: a wall on a cell edge mutates BOTH cells so the graph the BFS
    // walks and the rects physics collides with describe the same barrier.
    const vWall = (c, r) => { if (c >= 0 && c < cols - 1 && r >= 0 && r < rows) { cells[r][c].walls.e = true; cells[r][c + 1].walls.w = true; } };
    const hWall = (c, r) => { if (r >= 0 && r < rows - 1 && c >= 0 && c < cols) { cells[r][c].walls.s = true; cells[r + 1][c].walls.n = true; } };
    // 4-fold mirror about the world centre (col (cols-1)/2, row (rows-1)/2): one base
    // piece in the top-left quadrant -> the same cover in all four quadrants.
    const addSymVWall = (c, r) => { vWall(c, r); vWall(cols - 2 - c, r); vWall(c, rows - 1 - r); vWall(cols - 2 - c, rows - 1 - r); };
    const addSymHWall = (c, r) => { hWall(c, r); hWall(cols - 1 - c, r); hWall(c, rows - 2 - r); hWall(cols - 1 - c, rows - 2 - r); };
    // --- cover layout (top-left base pieces; each auto-mirrored to all 4 quadrants) ---
    // Discrete, gap-rich cover scattered symmetrically so shells get plenty of ricochet
    // surfaces (the whole point of the game). Each base piece is given in the top-left
    // fundamental domain and auto-mirrored to all four quadrants; every segment is a
    // single cell edge, so the field stays "多缺口" (many gaps, no long corridors / no
    // deep dead-ends) while reaching ~60% of the random survival map's wall count.
    const vBase = [[1, 1], [1, 4], [3, 2], [3, 5], [5, 1], [5, 4], [6, 2]];  // east-edge verticals
    const hBase = [[0, 3], [2, 1], [2, 4], [4, 2], [4, 0], [6, 1], [6, 4]];  // south-edge horizontals
    for (const [c, r] of vBase) addSymVWall(c, r);
    for (const [c, r] of hBase) addSymHWall(c, r);
    // Physical wall rects derived from the cell graph (identical scheme to the maze).
    const walls = boundaryWalls(WORLD_W, WORLD_H);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const cell = cells[y][x];
        const px = x * cellW, py = y * cellH;
        if (cell.walls.e && x < cols - 1) walls.push({ x: px + cellW - WALL / 2, y: py - WALL / 2, w: WALL, h: cellH + WALL });
        if (cell.walls.s && y < rows - 1) walls.push({ x: px - WALL / 2, y: py + cellH - WALL / 2, w: cellW + WALL, h: WALL });
      }
    }
    return { cells, walls, centerClear: { x: WORLD_W / 4, y: WORLD_H / 4, w: WORLD_W / 2, h: WORLD_H / 2 } };
  }

  // Symmetric fixed power-up candidate points for fixed_moba (8 points, mirrored 4-ways
  // about the world centre so neither side is favoured). Deterministic pixel positions.
  const FIXED_MOBA_POWERUP_POINTS = (function () {
    const mirror4 = (x, y) => [{ x, y }, { x: WORLD_W - x, y }, { x, y: WORLD_H - y }, { x: WORLD_W - x, y: WORLD_H - y }];
    const base = [
      { x: 2 * CELL_W + CELL_W / 2, y: 2 * CELL_H + CELL_H / 2 },   // outer/lane point
      { x: 6 * CELL_W + CELL_W / 2, y: 4 * CELL_H + CELL_H / 2 }    // centre-shoulder point
    ];
    const out = [];
    for (const b of base) for (const p of mirror4(b.x, b.y)) out.push(p);
    return out;
  })();

  function createTank(id, x, y, angle, color, accent) {
    return {
      id, x, y, angle, radius: TANK_RADIUS, color, accent,
      vx: 0, vy: 0,   // per-second velocity (displacement/dt), updated each updateTank -> feeds the motion obs
      reload: 0, reloadMax: FIRE_DELAY, fireDelay: FIRE_DELAY, ammoLimit: 5,
      alive: true, health: MAX_HEALTH, maxHealth: MAX_HEALTH, shield: 0, shieldTimer: 0,
      power: null, powerTimer: 0, powerShots: 0,
      lastHitAt: 0   // sim-time of last damage taken (drives moba1v1duel breathing regen)
    };
  }

  // Pure neural forward pass (used by the browser watch view and exported policies).
  function policyForward(obs, policy) {
    let x = obs;
    for (const layer of policy.layers) {
      const out = new Array(layer.b.length);
      for (let o = 0; o < layer.b.length; o++) {
        let sum = layer.b[o];
        const row = layer.w[o];
        for (let i = 0; i < row.length; i++) sum += row[i] * x[i];
        out[o] = layer.act === "tanh" ? Math.tanh(sum) : sum;
      }
      x = out;
    }
    let best = 0;
    for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
    return best;
  }

  const REWARD_DEFAULTS = {
    win: 1.0, loss: -1.0, draw: 0.0, winBySelfHit: 0.15, timeoutPenalty: 0.7,
    hit: 0.35, hitShield: 0.18, selfHit: -0.16, selfHitShield: -0.12, selfDefeat: -0.12,
    powerup: 0.04, poisonHurt: 0.0, closeRangeHit: 0.0, cleanTrade: 0.0, timePenalty: 0.001,
    // Shaping (off by default; the lessons from the deleted env are wired but dormant).
    backwardPenalty: 0.0, approachCoef: 0.0, aimBonus: 0.0,
    // NON-potential point-blank pressure: per-step bonus for staying inside proximityRange of a VISIBLE enemy.
    // Unlike approachCoef (potential-based -> preserves the optimum), this CHANGES the optimum toward cornering
    // evasive dodgers (laika) at close range where they cannot escape. Off by default (v1 stays byte-identical).
    proximityBonus: 0.0, proximityRange: 200,
    // ESCAPE-ANGLE / cornering reward (Janosov 2010.08193 formation score, SciRep wall-pinning): per-step bonus
    // proportional to the fraction of the VISIBLE enemy's escape directions that are BLOCKED (wall within cornerRange,
    // OR pointing back toward the agent). Rewards trapping a reactive dodger in a corner -- the script's 0.63 pressure
    // as a reward, not "be close" but "cut off its escape". Off by default.
    cornerCoef: 0.0, cornerRange: 120
  };

  // Lesson 1 (nav_powerup_poison) reward profile — entirely separate from battle rewards.
  const NAV_REWARD_DEFAULTS = {
    timePenalty: 0.001,
    wallHitPenalty: 0.02,
    wallHitCooldown: 0.25,
    pickupWhenEmptyReward: 0.8,
    poisonDamagePenaltyCoef: 0.2,
    successReward: 5.0,
    deathPenalty: -2.0,
    timeoutPenalty: -1.0,
    successSeconds: 12.0
  };

  // Lesson 1b (nav_route_to_center) reward profile — pure path-finding into the centre.
  // v2: stronger anti-stuck / anti-no-progress shaping to push held-out success past 80%.
  const NAV_ROUTE_REWARD_DEFAULTS = {
    timePenalty: 0.001,
    wallHitPenalty: 0.02,
    wallHitCooldown: 0.25,
    newCellReward: 0.04,        // first visit to a grid cell (v2: 0.02 -> 0.04)
    pathProgressCoef: 0.02,     // paid on BFS-distance improvement vs bestPathDist (not euclidean)
    stuckPenalty: 0.02,         // little movement over the window (v2: 0.005 -> 0.02)
    stuckWindow: 2.0,
    stuckDistance: 20,
    noProgressWindow: 4.0,      // v2: bestPathDist hasn't improved for this long...
    noProgressPenalty: 0.03,    // ...-> penalize (moving but not getting closer)
    enterCenterReward: 1.0,
    successReward: 5.0,
    timeoutPenalty: -1.0,
    deathPenalty: -2.0
  };

  // ============================ The engine ============================
  class RicochetCore {
    constructor(config = {}) {
      const arenaMode = config.arenaMode === "open" ? "open" : config.arenaMode === "survival" ? "survival" : "maze";
      const cfg = {
        seed: 1307, arenaMode, spawnPowerups: arenaMode === "open" ? false : true, maxSteps: 1000,
        actionRepeat: 2, stepDt: 1 / 30, spawnJitter: false, randomTurret: false,
        ...config
      };
      cfg.arenaMode = cfg.arenaMode === "open" ? "open" : cfg.arenaMode === "survival" ? "survival" : "maze";
      if (config.spawnPowerups === undefined) cfg.spawnPowerups = cfg.arenaMode === "open" ? false : true;
      cfg.spawnJitter = Boolean(cfg.spawnJitter);
      cfg.randomTurret = Boolean(cfg.randomTurret);   // shooting-lab: random turret spawn (open arena only)
      cfg.shellDecay = cfg.shellDecay !== false;   // default on; experimental shell-speed decay
      cfg.scenario = (config.scenario === "nav_powerup_poison" || config.scenario === "nav_route_to_center" || config.scenario === "fixed_moba" || config.scenario === "moba1v1duel" || config.scenario === "moba_poison_run")
        ? config.scenario : "battle";
      const isPoisonNav = cfg.scenario === "nav_powerup_poison";
      const isRoute = cfg.scenario === "nav_route_to_center";
      const isFixedMoba = cfg.scenario === "fixed_moba";   // fixed symmetric COMBAT map, poison OFF (red is a live opponent)
      const isMobaDuel = cfg.scenario === "moba1v1duel";   // same map + poison ring + duel mechanics (regen / 5-shot 2x / slow turn)
      const isPoisonRun = cfg.scenario === "moba_poison_run";  // NAV lesson: run the poison to the centre on the moba map
      const usesFixedMap = isFixedMoba || isMobaDuel || isPoisonRun;  // all render the fixed symmetric arena
      const isNav = isPoisonNav || isRoute || isPoisonRun;   // nav behaviour: red removed, fire inert
      const usesRouteReward = isRoute || isPoisonRun;        // share the BFS-to-centre reward + success + eval
      if (usesFixedMap) cfg.arenaMode = "survival";  // fixed maps always use the 2x survival world
      if (usesRouteReward) cfg.spawnPowerups = false;  // route / poison-run have no power-ups
      // Poison ring: on for survival battle, poison-nav, moba1v1duel, and moba_poison_run;
      // off for route and fixed_moba. Fully overridable via config.poisonEnabled.
      cfg.poisonEnabled = config.poisonEnabled !== undefined
        ? Boolean(config.poisonEnabled)
        : (cfg.arenaMode === "survival" && !isRoute && !isFixedMoba);
      // --- survival_v2 / combat_v2 long-form-battle rules (non-destructive: all knobs DEFAULT to the old
      // survival_v1 values, so every existing scenario/model is byte-identical unless v2 is opted into). The
      // `ruleset:"survival_v2"` preset just flips the defaults; individual knobs always override. ---
      const isV2 = config.ruleset === "survival_v2";
      cfg.ruleset = isV2 ? "survival_v2" : "survival_v1";
      cfg.tankMaxHp = Number(config.tankMaxHp) > 0 ? Number(config.tankMaxHp) : (isV2 ? MAX_HEALTH * 2 : MAX_HEALTH);
      cfg.regenDelay = Number(config.regenCooldownTicks) > 0
        ? Number(config.regenCooldownTicks) / 30
        : (isV2 ? MOBA_DUEL.regenDelay * 1.5 : MOBA_DUEL.regenDelay);
      cfg.spawnMode = (config.spawnMode === "half_random" || config.spawnMode === "full_random" || config.spawnMode === "fixed" || config.spawnMode === "tri_fixed")
        ? config.spawnMode : (isV2 ? "half_random" : "fixed");
      const reward = { ...REWARD_DEFAULTS, ...(config.reward || {}) };
      const navReward = { ...NAV_REWARD_DEFAULTS, ...(config.navReward || {}) };
      const navRouteReward = { ...NAV_ROUTE_REWARD_DEFAULTS, ...(config.navRouteReward || {}) };
      const navActive = usesRouteReward ? navRouteReward : navReward;   // shared-field accessor for the active nav lesson
      this.cfg = cfg;
      this.reward = reward;
      this.navReward = navReward;
      this.navRouteReward = navRouteReward;
      // Active world bounds: survival uses the 2x world; open/maze keep W x H.
      const worldW = cfg.arenaMode === "survival" ? WORLD_W : W;
      const worldH = cfg.arenaMode === "survival" ? WORLD_H : H;
      const worldDiag = Math.hypot(worldW, worldH);  // obs normalization uses the real world size (== ARENA_DIAG in open/maze)
      // Active maze grid: survival is a 2x cell grid; open/maze use the base grid.
      // (Cell size is identical, only the count differs.) Pathfinding MUST use this.
      const gridCols = cfg.arenaMode === "survival" ? SCOLS : COLS;
      const gridRows = cfg.arenaMode === "survival" ? SROWS : ROWS;
      function cellOfGrid(px, py) {
        return {
          cx: Math.min(gridCols - 1, Math.max(0, Math.floor(px / CELL_W))),
          cy: Math.min(gridRows - 1, Math.max(0, Math.floor(py / CELL_H)))
        };
      }

      // Episode-deterministic RNG: seedCounter drives no-arg resets; gameRng drives
      // within-episode randomness (power-up spawns / timers), reseeded each reset.
      let seedCounter = cfg.seed >>> 0;
      const pickSeed = () => {
        seedCounter = (seedCounter * 1664525 + 1013904223) >>> 0;
        return seedCounter;
      };
      let gameRng = new RNG(cfg.seed >>> 0);

      const state = {
        arenaSeed: cfg.seed >>> 0,
        walls: [], maze: null, tanks: [], shells: [], powerups: [],
        powerupTimer: 0, controls: [null, null],
        roundState: null,
        centerClear: null,
        safeRect: null, poisonActive: false, poisonAtMinCircle: false, nextShrinkTime: 0,
        nav: null,
        turretNextFireAt: TURRET.initialDelay, easyFireAt: 0, combat: null,
        elapsed: 0, stepCount: 0,
        rewardDelta: 0, done: false, truncated: false, result: "running",
        events: []
      };
      this.state = state;

      const emit = (type, data) => { state.events.push(data ? { type, ...data } : { type }); };
      const rewardForPlayer = (playerId, amount) => {
        state.rewardDelta += playerId === 0 ? amount : -amount;
      };

      // ---------------- geometry / world queries ----------------
      function collidesTank(x, y, tank) {
        if (x < tank.radius + WALL || x > worldW - tank.radius - WALL) return true;
        if (y < tank.radius + WALL || y > worldH - tank.radius - WALL) return true;
        if (state.walls.some((wall) => circleRectHit(x, y, tank.radius, wall))) return true;
        // Solid tank-vs-tank hitboxes: the two tanks can't drive through / overlap each other
        // (stops the endless "both tanks circling overlapped" stalemate). Axis-separated movement
        // in updateTank means a blocked tank can still slide along the contact, not freeze.
        for (const other of state.tanks) {
          if (other === tank || !other.alive) continue;
          const dx = x - other.x, dy = y - other.y, rr = tank.radius + other.radius;
          if (dx * dx + dy * dy < rr * rr) return true;
        }
        return false;
      }
      function canSee(a, b) {
        const steps = Math.ceil(Math.sqrt(distSq(a, b)) / 14);
        for (let i = 1; i < steps; i++) {
          const t = i / steps;
          const x = a.x + (b.x - a.x) * t;
          const y = a.y + (b.y - a.y) * t;
          if (state.walls.some((wall) => pointInRect(x, y, wall))) return false;
        }
        return true;
      }
      function tankShellCount(tank) {
        return state.shells.filter((shell) => shell.owner === tank.id).length;
      }
      function raycast(x, y, angle, maxDistance = RAY_MAX) {
        const step = 8.0;
        let distance = 0.0;
        while (distance < maxDistance) {
          distance += step;
          const px = x + Math.cos(angle) * distance;
          const py = y + Math.sin(angle) * distance;
          if (px <= WALL || px >= worldW - WALL || py <= WALL || py >= worldH - WALL) return distance;
          if (state.walls.some((wall) => pointInRect(px, py, wall))) return distance;
        }
        return maxDistance;
      }

      // ---------------- firing / shells ----------------
      function setReload(tank, duration) {
        tank.reload = duration;
        tank.reloadMax = Math.max(duration, 0.001);
      }
      function consumeWeaponPower(tank) {
        if (!isWeaponPower(tank.power)) return;
        tank.powerShots -= 1;
        if (tank.powerShots <= 0) { tank.power = null; tank.powerTimer = 0; tank.powerShots = 0; }
      }
      function applyPower(tank, power) {
        if (power.type === "shield") {
          tank.shield = 2; tank.shieldTimer = SHIELD_DURATION; emit("power", { x: tank.x, y: tank.y });
          return;
        }
        tank.power = power.type;
        tank.powerTimer = POWER_DURATION;
        tank.powerShots = isWeaponPower(power.type) ? ((isMobaDuel && power.type === "rapid") ? MOBA_DUEL.rapidShots : 2) : 0;
        emit("power", { x: tank.x, y: tank.y });
      }

      function fire(tank) {
        if (!tank.alive || tank.reload > 0) return;
        if (tank.power === "laser") {
          fireLaser(tank);
          setReload(tank, LASER_FIRE_DELAY);
          if (state.combat && tank.id === 0) state.combat.shotsFired += 1;
          emit("fire", { weapon: "laser" });
          consumeWeaponPower(tank);
          return;
        }
        let activeShells = tankShellCount(tank);
        if (activeShells >= tank.ammoLimit) return;
        const spread = tank.power === "triple" || tank.power === "bounce" ? [-0.14, 0, 0.14] : [0];
        let fired = false;
        for (const offset of spread) {
          if (activeShells >= tank.ammoLimit) break;
          const angle = tank.angle + offset;
          const nose = tank.radius + 9;
          const isMissile = tank.power === "missile";
          const isSnipe = tank.power === "bounce";
          const isTriple = tank.power === "triple";
          const isRapidDuel = isMobaDuel && tank.power === "rapid";   // lightning skill in the duel: 2x-speed bullets
          const baseSpeed = isSnipe ? SNIPE_SPEED : isMissile ? MISSILE_SPEED : isTriple ? TRIPLE_SPEED : SHELL_SPEED;
          const speed = (cfg.shellDecay ? baseSpeed * SHELL_BOOST : baseSpeed) * (isRapidDuel ? 2 : 1);
          // When decaying, the lifetime IS the decay window: speed hits 0 exactly when
          // ttl hits 0, so the shell visibly slows to a stop and then vanishes.
          const shellTtl = cfg.shellDecay ? SHELL_DECAY_DURATION : (isMissile ? 7 : isSnipe ? 8 : SHELL_TTL);
          const shell = {
            x: tank.x + Math.cos(angle) * nose,
            y: tank.y + Math.sin(angle) * nose,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            owner: tank.id,
            radius: isMissile ? 7 : isSnipe ? 4 : 5,
            ttl: shellTtl,
            ttl0: shellTtl,
            speed0: speed,
            age: 0,
            // decaying shells get a big bounce budget so the bounce limit doesn't
            // despawn them early -- they should live until they decelerate to a stop.
            bounces: cfg.shellDecay ? (isSnipe ? 0 : isMissile ? 1 : 60) : (isMissile ? 1 : isSnipe ? 0 : 5),
            hitIds: new Set(),
            type: isMissile ? "missile" : isSnipe ? "snipe" : "shell"
          };
          fired = true;
          if (!isSnipe && resolveMuzzleWallBounce(tank, shell)) {
            if (!tank.alive) break;
            continue;
          }
          state.shells.push(shell);
          activeShells += 1;
        }
        if (!fired || !tank.alive) return;
        if (state.combat && tank.id === 0) state.combat.shotsFired += 1;
        const baseReload = tank.power === "rapid" ? RAPID_FIRE_DELAY : tank.fireDelay;
        setReload(tank, activeShells >= tank.ammoLimit ? baseReload * AMMO_EXHAUSTED_COOLDOWN_MULT : baseReload);
        emit("fire", { weapon: tank.power === "missile" ? "missile" : tank.power === "bounce" ? "snipe" : "shell" });
        consumeWeaponPower(tank);
      }

      function resolveMuzzleWallBounce(tank, shell) {
        const overlappingWalls = state.walls.filter((wall) => circleRectHit(shell.x, shell.y, shell.radius, wall));
        if (!overlappingWalls.length) return false;
        let flipX = false, flipY = false;
        for (const wall of overlappingWalls) {
          if (wall.w <= wall.h) flipX = true;
          if (wall.h <= wall.w) flipY = true;
        }
        if (!flipX && !flipY) {
          if (Math.abs(shell.vx) > Math.abs(shell.vy)) flipX = true; else flipY = true;
        }
        if (flipX) shell.vx *= -1;
        if (flipY) shell.vy *= -1;
        shell.bounces -= 1;
        emit("bounce");
        const speed = Math.hypot(shell.vx, shell.vy) || 1;
        const ux = shell.vx / speed, uy = shell.vy / speed;
        for (let i = 0; i < 24 && state.walls.some((wall) => circleRectHit(shell.x, shell.y, shell.radius, wall)); i++) {
          shell.x += ux * 2; shell.y += uy * 2;
        }
        const hitDistance = tank.radius + shell.radius + 18;
        const hx = shell.x + ux * hitDistance, hy = shell.y + uy * hitDistance;
        const headingTowardOwner = (tank.x - shell.x) * ux + (tank.y - shell.y) * uy > 0;
        const ownerHit = headingTowardOwner &&
          distanceToSegment(tank.x, tank.y, shell.x, shell.y, hx, hy) <= tank.radius + shell.radius;
        if (!ownerHit) return false;
        damageTank(tank, tank.id, "shell");
        return true;
      }

      function fireLaser(tank) {
        const path = traceLaser(tank);
        emit("laser", { owner: tank.id, path });
        for (const target of state.tanks) {
          if (!target.alive || target.id === tank.id) continue;
          if (path.some((s) => distanceToSegment(target.x, target.y, s.x1, s.y1, s.x2, s.y2) <= target.radius)) {
            damageTank(target, tank.id, "laser");
            return;
          }
        }
      }

      function traceLaser(tank, maxLength = 2700) {
        const path = [];
        let x = tank.x + Math.cos(tank.angle) * (tank.radius + 12);
        let y = tank.y + Math.sin(tank.angle) * (tank.radius + 12);
        let vx = Math.cos(tank.angle), vy = Math.sin(tank.angle);
        let start = { x, y };
        let remaining = maxLength, bounces = 4;
        while (remaining > 0 && bounces >= 0) {
          const step = Math.min(6, remaining);
          const nx = x + vx * step, ny = y + vy * step;
          const hitX = state.walls.some((wall) => pointInRect(nx, y, wall));
          const hitY = state.walls.some((wall) => pointInRect(x, ny, wall));
          if (hitX || hitY) {
            path.push({ x1: start.x, y1: start.y, x2: x, y2: y });
            if (hitX) vx *= -1;
            if (hitY) vy *= -1;
            x += vx; y += vy; start = { x, y };
            bounces -= 1;
          } else { x = nx; y = ny; }
          remaining -= step;
        }
        if (Math.hypot(x - start.x, y - start.y) > 1) {
          path.push({ x1: start.x, y1: start.y, x2: x, y2: y });
        }
        return path;
      }

      function updateShell(shell, dt) {
        shell.ttl -= dt; shell.age += dt;
        if (cfg.shellDecay && shell.speed0 && shell.ttl0) {
          // Linear decay: 2x speed at birth -> 0 at end of life; direction preserved.
          const target = shell.speed0 * Math.max(0, 1 - shell.age / shell.ttl0);
          const mag = Math.hypot(shell.vx, shell.vy) || 1;
          const k = target / mag;
          shell.vx *= k; shell.vy *= k;
        }
        if (shell.type === "missile") steerMissile(shell, dt);
        const steps = Math.max(1, Math.ceil(Math.hypot(shell.vx * dt, shell.vy * dt) / 7));
        const stepDt = dt / steps;
        for (let i = 0; i < steps; i++) {
          let nextX = shell.x + shell.vx * stepDt;
          let nextY = shell.y + shell.vy * stepDt;
          if (shell.type !== "snipe" && state.walls.some((wall) => circleRectHit(nextX, shell.y, shell.radius, wall))) {
            shell.vx *= -1; shell.bounces -= 1; emit("bounce");
            nextX = shell.x + shell.vx * stepDt;
          }
          if (shell.type !== "snipe" && state.walls.some((wall) => circleRectHit(shell.x, nextY, shell.radius, wall))) {
            shell.vy *= -1; shell.bounces -= 1; emit("bounce");
            nextY = shell.y + shell.vy * stepDt;
          }
          shell.x = nextX; shell.y = nextY;
          for (const tank of state.tanks) {
            if (!tank.alive) continue;
            if (shell.type === "snipe" && shell.hitIds && shell.hitIds.has(tank.id)) continue;
            if (distSq(shell, tank) <= (shell.radius + tank.radius) ** 2) {
              if (shell.type === "snipe") {
                shell.hitIds.add(tank.id);
                damageTank(tank, shell.owner, "snipe");
                continue;
              }
              damageTank(tank, shell.owner, shell.type);
              shell.ttl = 0;
              return;
            }
          }
        }
      }

      function steerMissile(shell, dt) {
        const target = state.tanks.find((tank) => tank.id !== shell.owner && tank.alive);
        if (!target) return;
        const current = angleTo(shell.vx, shell.vy);
        const desired = angleTo(target.x - shell.x, target.y - shell.y);
        const next = current + clamp(angleDiff(current, desired), -2.2 * dt, 2.2 * dt);
        const speed = Math.hypot(shell.vx, shell.vy);
        shell.vx = Math.cos(next) * speed;
        shell.vy = Math.sin(next) * speed;
      }

      // ---------------- damage / round resolution (with reward) ----------------
      function damageTank(tank, scorerId, hitType) {
        if (!tank.alive) return false;
        tank.lastHitAt = state.elapsed;   // any hit (shield or health) interrupts breathing regen
        const isSelf = scorerId === tank.id;
        if (tank.shield > 0) {
          tank.shield -= 1;
          if (tank.shield <= 0) tank.shieldTimer = 0;
          emit("shield", { x: tank.x, y: tank.y });
          if (!isNav) rewardForPlayer(scorerId, isSelf ? reward.selfHitShield : reward.hitShield);
          return false;
        }
        tank.health = Math.max(0, tank.health - (hitType === "laser" ? LASER_DAMAGE : 1));
        if (state.combat) {
          if (tank.id === 0) {
            state.combat.hitsTaken += 1;
            if (isSelf) state.combat.selfHits += 1;
            else {                                    // learner took an ENEMY hit
              state._learnerLastHurt = state.elapsed;
              if (reward.closeRangeHit) {             // reckless-floor shaping: penalize blood-trading at close range
                const enemy = state.tanks[1];         // (teaches the evasive 1-shot-counter vs rushers; identity-blind)
                if (enemy && enemy.alive && Math.hypot(tank.x - enemy.x, tank.y - enemy.y) < 200) {
                  state.rewardDelta -= reward.closeRangeHit;
                }
              }
            }
          } else if (scorerId === 0) {
            state.combat.hitsDealt += 1;
            if (reward.cleanTrade && state.elapsed - (state._learnerLastHurt != null ? state._learnerLastHurt : -99) > 0.5) {
              state.rewardDelta += reward.cleanTrade; // landed a hit without being hit in the last 0.5s -> clean, not a trade
            }
          }
        }
        if (!isNav) rewardForPlayer(scorerId, isSelf ? reward.selfHit : reward.hit);
        if (tank.health <= 0) {
          defeatTank(tank, scorerId, isSelf ? "self_hit" : hitType);
          return true;
        }
        emit("hit", { x: tank.x, y: tank.y, color: tank.color });
        return false;
      }

      function defeatTank(tank, scorerId, reason) {
        if (!tank.alive) return;
        tank.alive = false;
        if (state.combat && state.combat.deathCause === null) { state.combat.deathCause = reason; state.combat.loserId = tank.id; }
        const winnerId = scorerId === tank.id ? 1 - tank.id : scorerId;
        emit("defeat", { x: tank.x, y: tank.y, color: tank.color });
        if (isNav) {
          // lesson 1: only blue matters; any blue death ends the episode as nav_death.
          if (tank.id === 0) navFinish("death");
          return;
        }
        if (cfg.arenaMode === "survival") {
          // No 3s draw window: first death ends the round. finishRound is
          // idempotent, so a same-tick double death resolves to the first one.
          if (reason === "self_hit") rewardForPlayer(tank.id, reward.selfDefeat);
          finishRound(winnerId, false, reason);
          return;
        }
        if (state.roundState && state.roundState.phase === "pending") {
          finishRound(null, true, reason);
          return;
        }
        if (state.roundState && state.roundState.phase === "settled") return;
        state.roundState = { phase: "pending", timer: DRAW_WINDOW, winnerId, reason };
        if (reason === "self_hit") rewardForPlayer(tank.id, reward.selfDefeat);
      }

      // ---------------- survival poison ring ----------------
      function inSafeRect(p) {
        const sr = state.safeRect;
        if (!sr) return true;
        return p.x >= sr.x && p.x <= sr.x + sr.w && p.y >= sr.y && p.y <= sr.y + sr.h;
      }
      // Laika priority: a tank caught in the poison heads for the safe-zone centre.
      // Deterministic; only fires in survival once the ring is active.
      function poisonEscape(bot) {
        if (cfg.arenaMode !== "survival" || !state.poisonActive || !state.safeRect) return null;
        if (inSafeRect(bot)) return null;
        const sr = state.safeRect;
        const desired = angleTo(sr.x + sr.w / 2 - bot.x, sr.y + sr.h / 2 - bot.y);
        return steerToward(bot, desired, 1.0);
      }
      function poisonTick(dt) {
        if (!cfg.poisonEnabled || !state.safeRect) return;  // gated by poisonEnabled (off for route + fixed_moba v1)
        if (state.elapsed >= POISON.startTime) {
          state.poisonActive = true;
          const sr = state.safeRect;
          const cx = worldW / 2, cy = worldH / 2;   // shrink all the way to the centre -> covers the whole map
          const dx = POISON.shrinkSpeedX * dt, dy = POISON.shrinkSpeedY * dt;
          const left = Math.min(sr.x + dx, cx);
          const top = Math.min(sr.y + dy, cy);
          const right = Math.max(sr.x + sr.w - dx, cx);
          const bottom = Math.max(sr.y + sr.h - dy, cy);
          state.safeRect = { x: left, y: top, w: right - left, h: bottom - top };
          // ring has fully closed onto the centre point -> the whole map is now poison
          if (state.safeRect.w <= 2 && state.safeRect.h <= 2) state.poisonAtMinCircle = true;
        }
        if (!state.poisonActive) return;
        const amount = POISON.dps * dt;
        for (const tank of state.tanks) {
          if (!tank.alive || inSafeRect(tank)) continue;
          tank.health = Math.max(0, tank.health - amount);
          tank.lastHitAt = state.elapsed;   // poison counts as damage -> interrupts breathing regen
          if (isNav && tank.id === 0) {
            state.nav.poisonDamageTaken += amount;
            state.rewardDelta -= amount * navReward.poisonDamagePenaltyCoef;
          } else if (state.combat && tank.id === 0) {
            state.combat.poisonDamageTaken += amount;
            state.rewardDelta -= amount * reward.poisonHurt;   // survival_v2 shaping: discourage sitting in poison
          }
          if (tank.health <= 0) defeatTank(tank, 1 - tank.id, "poison");
        }
      }

      // moba1v1duel breathing regen: a tank undamaged for regenDelay seconds heals
      // regenRate HP/s up to MAX_HEALTH. Any hit (shell/laser/poison) refreshes
      // tank.lastHitAt and thus interrupts it. Applies to both tanks (fair duel).
      function breathingRegenTick(dt) {
        if (!isMobaDuel) return;
        for (const tank of state.tanks) {
          if (!tank.alive || tank.health >= tank.maxHealth) continue;
          if (state.elapsed - tank.lastHitAt >= cfg.regenDelay) {   // survival_v2: slower regen cooldown
            tank.health = Math.min(tank.maxHealth, tank.health + MOBA_DUEL.regenRate * dt);
          }
        }
      }

      // Lesson 1 episode finisher (separate from battle finishRound).
      function navFinish(kind) {
        if (state.done) return;
        state.done = true;
        if (kind === "success") {
          state.rewardDelta += navActive.successReward;
          state.nav.navSuccess = true;
          state.result = usesRouteReward ? "route_success"
            : (state.nav.pickedAnyPowerup ? "nav_success_with_pickup" : "nav_success_no_pickup");
        } else if (kind === "death") {
          state.rewardDelta += navActive.deathPenalty;
          state.result = usesRouteReward ? "route_death" : "nav_death";
        } else if (kind === "timeout") {
          state.rewardDelta += navActive.timeoutPenalty;
          state.truncated = true;
          state.result = usesRouteReward ? "route_timeout" : "nav_timeout";
        }
      }

      function finishRound(winnerId, draw, reason) {
        if (state.done) return;
        state.done = true;
        state.roundState = { phase: "settled", timer: 0, draw: Boolean(draw), winnerId, reason };
        if (draw || winnerId === null || winnerId === undefined) {
          state.result = "draw";
          state.rewardDelta += reward.draw;
          return;
        }
        state.result = winnerId === 0 ? "win" : "loss";
        if (winnerId === 0) {
          state.rewardDelta += reason === "self_hit" ? reward.winBySelfHit : reward.win;
        } else {
          state.rewardDelta += reward.loss;
        }
      }

      // ---------------- power-up spawning (seeded) ----------------
      // fixed_moba: spawn onto a free symmetric candidate point (random type, seeded)
      // instead of a random pixel, so power-ups stay on the fixed, fair layout.
      function spawnPowerupFixed() {
        const free = FIXED_MOBA_POWERUP_POINTS.filter((pt) =>
          !state.powerups.some((pw) => distSq(pw, pt) < 30 * 30) &&
          !state.tanks.some((tk) => tk.alive && distSq(tk, pt) < 90 * 90) &&
          !collidesTank(pt.x, pt.y, { radius: 16 }));
        if (!free.length) return;
        const pt = free[gameRng.int(free.length)];
        const type = gameRng.pick(POWER_TYPES);
        const meta = POWERUP_META[type];
        state.powerups.push({ x: pt.x, y: pt.y, radius: 13, type, label: meta.label, color: meta.color, ttl: POWERUP_FIELD_TTL });
      }
      function spawnPowerup() {
        if (usesFixedMap) return spawnPowerupFixed();
        for (let attempts = 0; attempts < 60; attempts++) {
          const x = WALL + 34 + gameRng.next() * (worldW - WALL * 2 - 68);
          const y = WALL + 34 + gameRng.next() * (worldH - WALL * 2 - 68);
          if (!collidesTank(x, y, { radius: 16 })) {
            const nearTank = state.tanks.some((tank) => distSq(tank, { x, y }) < 90 * 90);
            if (!nearTank) {
              const type = gameRng.pick(POWER_TYPES);
              const meta = POWERUP_META[type];
              state.powerups.push({ x, y, radius: 13, type, label: meta.label, color: meta.color, ttl: POWERUP_FIELD_TTL });
              return;
            }
          }
        }
      }

      // Lesson 1: blue starts in the OUTER ring (outside the cleared centre), on a clear
      // cell-centre, chosen with the seeded gameRng. Falls back to the old corner spawn.
      function navSpawnBlue() {
        const cc = state.centerClear;
        const ccC0 = Math.floor(cc.x / CELL_W), ccC1 = Math.ceil((cc.x + cc.w) / CELL_W);
        const ccR0 = Math.floor(cc.y / CELL_H), ccR1 = Math.ceil((cc.y + cc.h) / CELL_H);
        for (let attempts = 0; attempts < 200; attempts++) {
          const cx = 1 + Math.floor(gameRng.next() * (SCOLS - 2));
          const cy = 1 + Math.floor(gameRng.next() * (SROWS - 2));
          if (cx >= ccC0 && cx < ccC1 && cy >= ccR0 && cy < ccR1) continue;  // inside centerClear
          const c = cellCenter(cx, cy);
          if (!collidesTank(c.x, c.y, { radius: TANK_RADIUS })) return c;
        }
        return cellCenter(1, SROWS - 2);
      }

      // Lesson 1: guarantee >=1 reachable powerup at episode start, biased onto the
      // blue -> centre line so there is an early learnable target. Seeded.
      function navSpawnInitialPowerup() {
        const blue = state.tanks[0], cc = state.centerClear;
        const ccx = cc.x + cc.w / 2, ccy = cc.y + cc.h / 2;
        for (let attempts = 0; attempts < 120; attempts++) {
          const t = 0.3 + gameRng.next() * 0.5;
          const x = blue.x + (ccx - blue.x) * t + (gameRng.next() * 2 - 1) * 120;
          const y = blue.y + (ccy - blue.y) * t + (gameRng.next() * 2 - 1) * 120;
          if (x < WALL + 30 || x > worldW - WALL - 30 || y < WALL + 30 || y > worldH - WALL - 30) continue;
          if (collidesTank(x, y, { radius: 16 })) continue;
          if (state.tanks.some((tk) => distSq(tk, { x, y }) < 90 * 90)) continue;
          const type = gameRng.pick(POWER_TYPES);
          const meta = POWERUP_META[type];
          state.powerups.push({ x, y, radius: 13, type, label: meta.label, color: meta.color, ttl: POWERUP_FIELD_TTL });
          return true;
        }
        spawnPowerup();
        return state.powerups.length > 0;
      }

      // Lesson 1b: BFS cell-distance field from the cleared centre over the maze graph.
      // Walls are static in the route lesson (no poison), so it is computed once per reset.
      function computeCenterDistField() {
        const dist = [];
        for (let y = 0; y < gridRows; y++) dist.push(new Array(gridCols).fill(Infinity));
        const cc = state.centerClear;
        const queue = [];
        for (let y = 0; y < gridRows; y++) {
          for (let x = 0; x < gridCols; x++) {
            const c = cellCenter(x, y);
            if (c.x > cc.x && c.x < cc.x + cc.w && c.y > cc.y && c.y < cc.y + cc.h) { dist[y][x] = 0; queue.push([x, y]); }
          }
        }
        for (let head = 0; head < queue.length; head++) {
          const [x, y] = queue[head];
          const d = dist[y][x], cell = state.maze[y][x];
          if (y > 0 && !cell.walls.n && dist[y - 1][x] > d + 1) { dist[y - 1][x] = d + 1; queue.push([x, y - 1]); }
          if (x < gridCols - 1 && !cell.walls.e && dist[y][x + 1] > d + 1) { dist[y][x + 1] = d + 1; queue.push([x + 1, y]); }
          if (y < gridRows - 1 && !cell.walls.s && dist[y + 1][x] > d + 1) { dist[y + 1][x] = d + 1; queue.push([x, y + 1]); }
          if (x > 0 && !cell.walls.w && dist[y][x - 1] > d + 1) { dist[y][x - 1] = d + 1; queue.push([x - 1, y]); }
        }
        return dist;
      }

      // Lesson 1b per-RL-step shaping: new-cell exploration, BFS path-distance progress
      // (paid only when bestPathDist improves -> no back-and-forth farming), stuck detection.
      function routeProgress() {
        const blue = state.tanks[0];
        const inCentre = pointInRect(blue.x, blue.y, state.centerClear);  // waiting to win -> no stuck/no-progress penalty
        const { cx, cy } = cellOfGrid(blue.x, blue.y);
        const key = cy * gridCols + cx;
        if (!state.nav.visited.has(key)) {
          state.nav.visited.add(key);
          state.nav.newCells += 1;
          state.rewardDelta += navRouteReward.newCellReward;
        }
        const pd = (state.routeDistField && state.routeDistField[cy]) ? state.routeDistField[cy][cx] : Infinity;
        state.nav.pathDist = pd;
        if (Number.isFinite(pd) && pd < state.nav.bestPathDist) {
          if (Number.isFinite(state.nav.bestPathDist)) state.rewardDelta += navRouteReward.pathProgressCoef * (state.nav.bestPathDist - pd);
          state.nav.bestPathDist = pd;
          state.nav.lastProgressAt = state.elapsed;   // real path progress refreshes the no-progress clock
        }
        // no-progress: moving but bestPathDist stalled for noProgressWindow (only before the centre, once per window)
        if (!inCentre && state.elapsed - state.nav.lastProgressAt >= navRouteReward.noProgressWindow) {
          state.rewardDelta -= navRouteReward.noProgressPenalty;
          state.nav.lastProgressAt = state.elapsed;
          state.nav.noProgressEvents += 1;
        }
        // stuck: little movement over stuckWindow (once per window; skipped inside the centre)
        if (!inCentre && state.elapsed - state.nav.stuckAnchorElapsed >= navRouteReward.stuckWindow) {
          const movedSq = (blue.x - state.nav.stuckAnchor.x) ** 2 + (blue.y - state.nav.stuckAnchor.y) ** 2;
          if (movedSq < navRouteReward.stuckDistance ** 2) { state.rewardDelta -= navRouteReward.stuckPenalty; state.nav.stuckEvents += 1; }
          state.nav.stuckAnchor = { x: blue.x, y: blue.y };
          state.nav.stuckAnchorElapsed = state.elapsed;
        }
      }

      // ---------------- per-tank update ----------------
      function updateTank(tank, dt) {
        if (!tank.alive) return;
        const input = state.controls[tank.id] || { throttle: 0, turn: 0, fire: false };
        if (tank.id === 0 && input.throttle < -0.1) state.rewardDelta -= reward.backwardPenalty;

        tank.angle += input.turn * (isMobaDuel ? TANK_TURN_SPEED * MOBA_DUEL.turnScale : TANK_TURN_SPEED) * dt;
        tank.angle = (tank.angle + TAU) % TAU;
        const distance = input.throttle * TANK_MOVE_SPEED * dt;
        const nx = tank.x + Math.cos(tank.angle) * distance;
        const ny = tank.y + Math.sin(tank.angle) * distance;
        const sx0 = tank.x, sy0 = tank.y;
        const movedX = !collidesTank(nx, tank.y, tank); if (movedX) tank.x = nx;
        const movedY = !collidesTank(tank.x, ny, tank); if (movedY) tank.y = ny;
        // actual per-second velocity AFTER wall/tank blocking (a blocked rusher reads ~0 -> the obs sees it stall)
        tank.vx = dt > 0 ? (tank.x - sx0) / dt : 0;
        tank.vy = dt > 0 ? (tank.y - sy0) / dt : 0;
        // Lesson 1 wall-collision event: penalize a blocked move at most once per cooldown
        // (so hugging a wall doesn't bleed reward every physics frame).
        if (isNav && tank.id === 0 && Math.abs(input.throttle) > 0.1 && (!movedX || !movedY)
            && state.elapsed - state.nav.lastWallPenaltyAt >= navActive.wallHitCooldown) {
          state.rewardDelta -= navActive.wallHitPenalty;
          state.nav.lastWallPenaltyAt = state.elapsed;
          state.nav.wallHits += 1;
        }

        if (input.fire && !isNav) fire(tank);   // lesson 1 is navigation-only: the fire action is inert (no self-suicide)
        tank.reload = Math.max(0, tank.reload - dt);
        if (tank.powerTimer > 0) {
          tank.powerTimer -= dt;
          if (tank.powerTimer <= 0) { tank.power = null; tank.powerTimer = 0; tank.powerShots = 0; }
        }
        if (tank.shieldTimer > 0) {
          tank.shieldTimer -= dt;
          if (tank.shieldTimer <= 0) { tank.shield = 0; tank.shieldTimer = 0; }
        }
        for (let i = state.powerups.length - 1; i >= 0; i--) {
          const power = state.powerups[i];
          if (distSq(tank, power) < (tank.radius + power.radius) ** 2) {
            const wasEmpty = !tank.power && tank.shield <= 0;
            applyPower(tank, power);
            state.powerups.splice(i, 1);
            if (isNav) {
              if (tank.id === 0) {
                state.nav.pickups += 1;
                state.nav.pickedAnyPowerup = true;
                if (wasEmpty) { state.rewardDelta += navReward.pickupWhenEmptyReward; state.nav.pickupsWhenEmpty += 1; }
              }
            } else {
              rewardForPlayer(tank.id, reward.powerup);
              if (state.combat && tank.id === 0) state.combat.powerups += 1;
              else if (state.combat && tank.id === 1) state.combat.enemyPowerups += 1;
            }
          }
        }
      }

      // ---------------- the simulation tick ----------------
      function update(dt) {
        state.elapsed += dt;
        if (state.done) return;

        if (state.roundState && state.roundState.phase === "pending") {
          state.roundState.timer -= dt;
          if (state.roundState.timer <= 0) {
            finishRound(state.roundState.winnerId, false, state.roundState.reason);
            return;
          }
        }

        for (const tank of state.tanks) updateTank(tank, dt);

        if (state.combat) {   // contact/stuck-duration metric: physics ticks the two tanks are touching
          const t0 = state.tanks[0], t1 = state.tanks[1];
          if (t0 && t1 && t0.alive && t1.alive) {
            const rr = t0.radius + t1.radius + 4;
            if ((t0.x - t1.x) ** 2 + (t0.y - t1.y) ** 2 < rr * rr) state.combat.contactSteps += 1;
          }
        }

        for (let i = state.shells.length - 1; i >= 0; i--) {
          const shell = state.shells[i];
          updateShell(shell, dt);
          if (shell.ttl <= 0 || shell.bounces < 0) {
            if (shell.type === "missile") emit("missileExplode", { x: shell.x, y: shell.y });
            state.shells.splice(i, 1);
          }
        }

        poisonTick(dt);
        breathingRegenTick(dt);

        // Lesson 1 (poison) success: survive successSeconds after the ring reaches the min circle.
        if (isPoisonNav && !state.done && state.poisonAtMinCircle && state.tanks[0].alive) {
          state.nav.survivalAfterMinCircle += dt;
          if (state.nav.survivalAfterMinCircle >= navReward.successSeconds) navFinish("success");
        }

        // Lesson 1b (route) success: stay inside the centre continuously for 2s.
        if (usesRouteReward && !state.done) {
          const blue = state.tanks[0];
          if (pointInRect(blue.x, blue.y, state.centerClear)) {
            if (!state.nav.enteredCenter) { state.nav.enteredCenter = true; state.rewardDelta += navRouteReward.enterCenterReward; }
            state.nav.centerStayTime += dt;
            if (state.nav.centerStayTime >= 2.0) navFinish("success");
          } else {
            state.nav.centerStayTime = 0;
          }
        }

        if (cfg.spawnPowerups) {
          state.powerupTimer -= dt;
          if (state.powerupTimer <= 0 && state.powerups.length < 3) {
            spawnPowerup();
            state.powerupTimer = 5 + gameRng.next() * 4;
          }
        }
        for (let i = state.powerups.length - 1; i >= 0; i--) {
          state.powerups[i].ttl -= dt;
          if (state.powerups[i].ttl <= 0) state.powerups.splice(i, 1);
        }
      }

      // ---------------- dense (per-step) reward ----------------
      function denseReward() {
        if (isNav) return -navActive.timePenalty;   // nav lessons: per-step time cost only; events handled elsewhere
        const me = state.tanks[0], opp = state.tanks[1];
        if (!me.alive) return -reward.timePenalty;
        let r = -reward.timePenalty;
        const visible = opp.alive && canSee(me, opp);
        if (reward.approachCoef && visible) {
          const dx = opp.x - me.x, dy = opp.y - me.y;
          const dist = Math.hypot(dx, dy) / ARENA_DIAG;
          r += reward.approachCoef * (state._prevDist - dist);
          state._prevDist = dist;
        }
        if (reward.aimBonus && visible) {
          const desired = angleTo(opp.x - me.x, opp.y - me.y);
          const alignment = 1.0 - Math.min(Math.abs(angleDiff(me.angle, desired)) / Math.PI, 1.0);
          r += reward.aimBonus * alignment;
        }
        if (reward.proximityBonus && visible) {                       // point-blank pressure vs evasive dodgers (NON-potential)
          const dx = opp.x - me.x, dy = opp.y - me.y;
          if (Math.hypot(dx, dy) < (reward.proximityRange || 200)) r += reward.proximityBonus;
        }
        if (reward.cornerCoef && visible) {                           // cut off the enemy's escape angles (cornering)
          const N = 16, rng = reward.cornerRange || 120, bearMe = angleTo(me.x - opp.x, me.y - opp.y);
          let blocked = 0;
          for (let i = 0; i < N; i++) {
            const ang = -Math.PI + i * TAU / N;                       // a direction is BLOCKED if a wall is near OR it heads toward the agent
            if (raycast(opp.x, opp.y, ang, rng) < rng - 1 || Math.cos(ang - bearMe) > 0.4) blocked++;
          }
          r += reward.cornerCoef * (blocked / N);
        }
        return r;
      }

      // ---------------- scripted controllers (opponents/teachers) ----------------
      // (Verbatim Laika/heuristic/pathfinder logic; performance.now() oscillators
      //  are replaced by the deterministic sim clock state.elapsed.)
      function steerToward(tank, desired, pace = 1) {
        const diff = angleDiff(tank.angle, desired);
        const abs = Math.abs(diff);
        return {
          throttle: abs < 0.72 ? pace : abs < 1.45 ? pace * 0.45 : pace * 0.12,
          turn: diff > 0.06 ? 1 : diff < -0.06 ? -1 : 0,
          fire: false
        };
      }
      function evadeToward(tank, desired, pace = 1) {
        const diff = angleDiff(tank.angle, desired);
        if (Math.abs(diff) > Math.PI * 0.62) {
          const rearDiff = angleDiff(tank.angle + Math.PI, desired);
          return { throttle: -pace, turn: rearDiff > 0.08 ? -1 : rearDiff < -0.08 ? 1 : 0, fire: false };
        }
        return steerToward(tank, desired, pace);
      }
      function blendAngles(primary, secondary, weight) {
        const w = clamp(weight, 0, 1);
        const x = Math.cos(primary) * (1 - w) + Math.cos(secondary) * w;
        const y = Math.sin(primary) * (1 - w) + Math.sin(secondary) * w;
        return angleTo(x, y);
      }
      function findPathCells(start, goal) {
        if (!state.maze) return null;
        const key = (x, y) => y * gridCols + x;
        const startK = key(start.cx, start.cy);
        const goalK = key(goal.cx, goal.cy);
        const prev = new Map();
        const visited = new Set([startK]);
        const queue = [[start.cx, start.cy]];
        let head = 0;
        while (head < queue.length) {
          const [x, y] = queue[head++];
          if (x === goal.cx && y === goal.cy) break;
          const w = state.maze[y][x].walls;
          const nbrs = [];
          if (!w.n && y > 0) nbrs.push([x, y - 1]);
          if (!w.e && x < gridCols - 1) nbrs.push([x + 1, y]);
          if (!w.s && y < gridRows - 1) nbrs.push([x, y + 1]);
          if (!w.w && x > 0) nbrs.push([x - 1, y]);
          for (const [nx, ny] of nbrs) {
            const k = key(nx, ny);
            if (!visited.has(k)) { visited.add(k); prev.set(k, key(x, y)); queue.push([nx, ny]); }
          }
        }
        if (!visited.has(goalK)) return null;
        const path = [];
        let cur = goalK;
        while (cur !== undefined) {
          const cx = cur % gridCols, cy = Math.floor(cur / gridCols);
          path.push({ x: cx * CELL_W + CELL_W / 2, y: cy * CELL_H + CELL_H / 2 });
          if (cur === startK) break;
          cur = prev.get(cur);
        }
        path.reverse();
        return path;
      }
      function pathWaypointFor(tank, goal) {
        const start = cellOfGrid(tank.x, tank.y);
        const gcell = cellOfGrid(goal.x, goal.y);
        if (start.cx === gcell.cx && start.cy === gcell.cy) return goal;
        const path = findPathCells(start, gcell);
        if (!path || path.length < 2) return goal;
        let waypoint = path[1];
        if (path.length > 2 && Math.hypot(waypoint.x - tank.x, waypoint.y - tank.y) < CELL_W * 0.36) {
          waypoint = path[2];
        }
        return waypoint;
      }
      function muzzleClear(tank, angle, shellRadius = 5) {
        const ux = Math.cos(angle), uy = Math.sin(angle);
        const start = Math.max(0, tank.radius - shellRadius - 2);
        const end = tank.radius + 12;
        for (let d = start; d <= end; d += 2) {
          const sx = tank.x + ux * d, sy = tank.y + uy * d;
          if (state.walls.some((wall) => circleRectHit(sx, sy, shellRadius + 0.75, wall))) return false;
        }
        return true;
      }
      function immediateSelfBounceRisk(tank, angle, shellRadius = 5) {
        let x = tank.x + Math.cos(angle) * (tank.radius + 9);
        let y = tank.y + Math.sin(angle) * (tank.radius + 9);
        let vx = Math.cos(angle), vy = Math.sin(angle);
        let bounced = false;
        for (let i = 0; i < 24; i++) {
          const step = 4;
          let nx = x + vx * step, ny = y + vy * step;
          const hitX = state.walls.some((wall) => circleRectHit(nx, y, shellRadius, wall));
          const hitY = state.walls.some((wall) => circleRectHit(x, ny, shellRadius, wall));
          if (hitX) { vx *= -1; bounced = true; nx = x + vx * step; }
          if (hitY) { vy *= -1; bounced = true; ny = y + vy * step; }
          if (bounced && distanceToSegment(tank.x, tank.y, x, y, nx, ny) <= tank.radius + shellRadius) return true;
          x = nx; y = ny;
        }
        return false;
      }
      function safeFireAngle(tank, angle, shellRadius = 5) {
        return muzzleClear(tank, angle, shellRadius) && !immediateSelfBounceRisk(tank, angle, shellRadius);
      }
      function laikaCanFireNow(tank) {
        if (tank.reload > 0.02) return false;
        if (tank.power === "laser") return true;
        if (tankShellCount(tank) >= tank.ammoLimit) return false;
        if (tank.power === "bounce") return true;
        const shellRadius = tank.power === "missile" ? 7 : 5;
        const spread = tank.power === "triple" ? [-0.14, 0, 0.14] : [0];
        return spread.every((offset) => safeFireAngle(tank, tank.angle + offset, shellRadius));
      }
      function traceCandidateShot(shooter, target, angle, maxLength = 1350) {
        if (!safeFireAngle(shooter, angle)) return null;
        let x = shooter.x + Math.cos(angle) * (shooter.radius + 11);
        let y = shooter.y + Math.sin(angle) * (shooter.radius + 11);
        let vx = Math.cos(angle), vy = Math.sin(angle);
        let travelled = 0, bounces = 0, safeFromSelf = false;
        while (travelled < maxLength && bounces <= 5) {
          const step = 7;
          const nx = x + vx * step, ny = y + vy * step;
          if (safeFromSelf && distanceToSegment(shooter.x, shooter.y, x, y, nx, ny) <= shooter.radius + 5) return null;
          if (distanceToSegment(target.x, target.y, x, y, nx, ny) <= target.radius + 5) return { angle, length: travelled, bounces };
          const hitX = state.walls.some((wall) => circleRectHit(nx, y, 5, wall));
          const hitY = state.walls.some((wall) => circleRectHit(x, ny, 5, wall));
          if (hitX || hitY) {
            if (hitX) vx *= -1;
            if (hitY) vy *= -1;
            bounces += 1; x += vx * step; y += vy * step;
          } else { x = nx; y = ny; }
          travelled += step;
          if (travelled > shooter.radius * 2.8) safeFromSelf = true;
        }
        return null;
      }
      function findLaikaShot(bot, target) {
        const direct = angleTo(target.x - bot.x, target.y - bot.y);
        const directHit = canSee(bot, target) ? traceCandidateShot(bot, target, direct, 760) : null;
        if (directHit) return { ...directHit, direct: true };
        const candidates = [];
        for (let i = 0; i < 48; i++) candidates.push((i / 48) * TAU);
        for (const offset of [-0.95, -0.7, -0.46, -0.28, 0.28, 0.46, 0.7, 0.95]) candidates.push(direct + offset);
        let best = null;
        for (const angle of candidates) {
          const shot = traceCandidateShot(bot, target, (angle + TAU) % TAU);
          if (!shot) continue;
          const turnCost = Math.abs(angleDiff(bot.angle, shot.angle)) * 130;
          const score = shot.length + shot.bounces * 95 + turnCost;
          if (!best || score < best.score) best = { ...shot, score, direct: false };
        }
        return best;
      }
      function shellDangerFor(tank, shell) {
        let x = shell.x, y = shell.y, vx = shell.vx, vy = shell.vy;
        let minDistance = Infinity;
        let closest = { x, y }, closestTime = 0, closestVelocity = { x: vx, y: vy };
        let bounces = shell.bounces;
        for (let i = 0; i < 72 && bounces >= 0; i++) {
          const dt = 1 / 30;
          let nx = x + vx * dt, ny = y + vy * dt;
          if (shell.type !== "snipe" && state.walls.some((wall) => circleRectHit(nx, y, shell.radius, wall))) {
            vx *= -1; bounces -= 1; nx = x + vx * dt;
          }
          if (shell.type !== "snipe" && state.walls.some((wall) => circleRectHit(x, ny, shell.radius, wall))) {
            vy *= -1; bounces -= 1; ny = y + vy * dt;
          }
          const d = distanceToSegment(tank.x, tank.y, x, y, nx, ny);
          if (d < minDistance) { minDistance = d; closest = { x: nx, y: ny }; closestTime = i * dt; closestVelocity = { x: vx, y: vy }; }
          x = nx; y = ny;
        }
        const dangerRadius = tank.radius + shell.radius + 16;
        const speed = Math.hypot(closestVelocity.x, closestVelocity.y) || 1;
        const headingTowardTank =
          (tank.x - closest.x) * (closestVelocity.x / speed) + (tank.y - closest.y) * (closestVelocity.y / speed) > -6;
        const distanceUrgency = clamp((dangerRadius * 3.8 - minDistance) / (dangerRadius * 3.8), 0, 1);
        const timeUrgency = clamp((2.2 - closestTime) / 2.2, 0, 1);
        const hitRisk = minDistance <= tank.radius + shell.radius + 7 ? 0.35 : 0;
        const urgency = clamp(distanceUrgency * (0.35 + timeUrgency * 0.65) + hitRisk + (headingTowardTank ? 0.08 : 0), 0, 1);
        return { shell, closest, closestTime, closestVelocity, minDistance, urgency, lethal: minDistance <= tank.radius + shell.radius + 7 };
      }
      function laikaEvadeAngle(tank, threats) {
        let best = null;
        const preferredClearance = tank.radius + 34;
        for (let i = 0; i < 32; i++) {
          const angle = (i / 32) * TAU;
          const x = tank.x + Math.cos(angle) * 94, y = tank.y + Math.sin(angle) * 94;
          if (collidesTank(x, y, tank)) continue;
          const wallClearance = raycast(tank.x, tank.y, angle, 150);
          let score = Math.min(wallClearance, preferredClearance) * 0.9;
          score -= Math.abs(angleDiff(tank.angle, angle)) * 10;
          for (const threat of threats) {
            const lead = clamp(threat.closestTime, 0.18, 1.0);
            const futureX = threat.closest.x + threat.closestVelocity.x * lead * 0.25;
            const futureY = threat.closest.y + threat.closestVelocity.y * lead * 0.25;
            const away = Math.hypot(x - futureX, y - futureY);
            const shellHeading = angleTo(threat.closestVelocity.x, threat.closestVelocity.y);
            const crossing = Math.abs(Math.sin(angleDiff(angle, shellHeading)));
            const movingWithShell = Math.cos(angleDiff(angle, shellHeading));
            score += away * threat.urgency * (threat.lethal ? 1.6 : 1.0);
            score += crossing * 55 * threat.urgency;
            score -= Math.max(0, movingWithShell) * 42 * threat.urgency;
          }
          if (!best || score > best.score) best = { angle, score };
        }
        if (best) return best.angle;
        const worst = threats[0];
        return angleTo(tank.x - worst.closest.x, tank.y - worst.closest.y);
      }
      function laikaGoal(bot, target) {
        let goal = target, bestScore = -Infinity;
        for (const power of state.powerups) {
          const dBot = Math.hypot(power.x - bot.x, power.y - bot.y);
          const dTarget = Math.hypot(power.x - target.x, power.y - target.y);
          const value = power.type === "shield" ? 95 : power.type === "bounce" ? 100 : power.type === "laser" || power.type === "missile" ? 80 : 55;
          const score = value - dBot * 0.28 + dTarget * 0.08;
          if (dBot < 290 && score > bestScore) { bestScore = score; goal = power; }
        }
        return goal;
      }
      function aggressiveLaikaGoal(bot, target) {
        let goal = target, bestScore = -Infinity;
        for (const power of state.powerups) {
          const dBot = Math.hypot(power.x - bot.x, power.y - bot.y);
          const dTarget = Math.hypot(power.x - target.x, power.y - target.y);
          const value = power.type === "bounce" ? 135 : power.type === "laser" || power.type === "missile" ? 98 : power.type === "rapid" || power.type === "triple" ? 72 : 18;
          const score = value - dBot * 0.34 + dTarget * 0.03;
          if (dBot < 210 && score > bestScore) { bestScore = score; goal = power; }
        }
        return goal;
      }
      function laikaInput(bot, target) {
        if (!target.alive) return { throttle: 0, turn: 0, fire: false };
        const escape = poisonEscape(bot);
        if (escape) return escape;
        const threats = state.shells
          .filter((shell) => shell.owner !== bot.id)
          .map((shell) => shellDangerFor(bot, shell))
          .filter((threat) => threat.urgency > 0.12)
          .sort((a, b) => b.urgency - a.urgency);
        if (threats.length && threats[0].urgency > 0.32) {
          const desired = laikaEvadeAngle(bot, threats.slice(0, 3));
          const input = evadeToward(bot, desired, threats[0].lethal ? 1 : 0.88);
          input.fire = false;
          return input;
        }
        const attackDistance = Math.hypot(target.x - bot.x, target.y - bot.y);
        const direct = angleTo(target.x - bot.x, target.y - bot.y);
        if (bot.power === "bounce") {
          const aimError = Math.abs(angleDiff(bot.angle, direct));
          const input = steerToward(bot, direct, attackDistance > 150 ? 0.8 : 0.25);
          input.fire = aimError < 0.13 && laikaCanFireNow(bot);
          return input;
        }
        const shot = findLaikaShot(bot, target);
        if (shot && attackDistance < 760) {
          const aimError = Math.abs(angleDiff(bot.angle, shot.angle));
          const input = steerToward(bot, shot.angle, shot.direct ? 0.38 : 0.2);
          const clearToFire = laikaCanFireNow(bot);
          if (!clearToFire && aimError < 0.22) input.throttle = -0.7;
          input.fire = aimError < (shot.direct ? 0.12 : 0.08) && clearToFire;
          return input;
        }
        const goal = laikaGoal(bot, target);
        const waypoint = pathWaypointFor(bot, goal);
        let desired = angleTo(waypoint.x - bot.x, waypoint.y - bot.y);
        if (goal === target && attackDistance < 230 && canSee(bot, target)) {
          const strafe = Math.sin(state.elapsed * 2.0833 + bot.id * 1.7) > 0 ? 1 : -1;
          desired = angleTo(target.x - bot.x, target.y - bot.y) + strafe * Math.PI / 2;
        }
        const input = steerToward(bot, desired, goal === target ? 0.92 : 1);
        if (canSee(bot, target)) {
          const clearToFire = laikaCanFireNow(bot);
          const aimError = Math.abs(angleDiff(bot.angle, direct));
          if (!clearToFire && aimError < 0.22) input.throttle = -0.7;
          input.fire = aimError < 0.15 && clearToFire;
        }
        return input;
      }
      function aggressiveLaikaInput(bot, target) {
        if (!target.alive) return { throttle: 0, turn: 0, fire: false };
        const escape = poisonEscape(bot);
        if (escape) return escape;
        const threats = state.shells
          .filter((shell) => shell.owner !== bot.id)
          .map((shell) => shellDangerFor(bot, shell))
          .filter((threat) => threat.urgency > 0.34)
          .sort((a, b) => b.urgency - a.urgency);
        const pressureThreat = threats[0];
        const pressureEvade = pressureThreat ? laikaEvadeAngle(bot, threats.slice(0, 3)) : null;
        if (pressureThreat && (pressureThreat.urgency > 0.78 || (pressureThreat.lethal && pressureThreat.urgency > 0.62))) {
          const input = evadeToward(bot, pressureEvade, 1);
          input.fire = false;
          return input;
        }
        const shot = findLaikaShot(bot, target);
        const attackDistance = Math.hypot(target.x - bot.x, target.y - bot.y);
        const direct = angleTo(target.x - bot.x, target.y - bot.y);
        const clearToFire = laikaCanFireNow(bot);
        if (bot.power === "bounce") {
          const aimError = Math.abs(angleDiff(bot.angle, direct));
          const input = steerToward(bot, direct, attackDistance > 65 ? 1 : 0.35);
          input.fire = aimError < 0.2 && clearToFire;
          return input;
        }
        if (shot && attackDistance < 900 && (shot.direct || attackDistance < 260)) {
          const aimError = Math.abs(angleDiff(bot.angle, shot.angle));
          const desiredPace = attackDistance > 78 ? 1 : 0.35;
          let moveAngle = shot.direct && attackDistance > 70 ? direct : shot.angle;
          if (pressureThreat && pressureEvade !== null) {
            moveAngle = blendAngles(moveAngle, pressureEvade, pressureThreat.lethal ? 0.55 : 0.32);
          }
          const input = steerToward(bot, moveAngle, desiredPace);
          if (!clearToFire && aimError < 0.2) input.throttle = attackDistance > 58 ? 0.55 : -0.25;
          input.fire = aimError < (shot.direct ? 0.2 : 0.14) && clearToFire;
          return input;
        }
        const goal = aggressiveLaikaGoal(bot, target);
        const waypoint = pathWaypointFor(bot, goal);
        let desired = angleTo(waypoint.x - bot.x, waypoint.y - bot.y);
        if (goal === target && canSee(bot, target)) {
          if (attackDistance > 70) {
            desired = direct;
          } else {
            const strafe = Math.sin(state.elapsed * 3.8462 + bot.id * 2.1) > 0 ? 1 : -1;
            desired = direct + strafe * Math.PI / 3;
          }
        }
        if (pressureThreat && pressureEvade !== null) {
          desired = blendAngles(desired, pressureEvade, pressureThreat.lethal ? 0.5 : 0.28);
        }
        const pace = goal === target ? (attackDistance > 58 ? 1 : 0.42) : 1;
        const input = steerToward(bot, desired, pace);
        if (canSee(bot, target)) {
          const aimError = Math.abs(angleDiff(bot.angle, direct));
          if (!clearToFire && aimError < 0.2) input.throttle = attackDistance > 58 ? 0.55 : -0.25;
          input.fire = aimError < 0.22 && clearToFire;
        }
        return input;
      }
      function nearbyPowerup(bot, types, maxDist) {
        let best = null, bestD = Infinity;
        for (const pu of state.powerups) {
          if (types && types.indexOf(pu.type) < 0) continue;
          const d = Math.hypot(pu.x - bot.x, pu.y - bot.y);
          if (d <= maxDist && d < bestD) { bestD = d; best = pu; }
        }
        return best;
      }
      // The strongest scripted expert (see LAIKA_PRO). Reuses laika's shot-finding,
      // dodging, poison-escape and pathfinding; adds distance bands, low-health retreat,
      // shield/weapon pickup, and disciplined safe firing. Output is the standard 18-way
      // control, so controlToAction() turns it into a Discrete(18) id for BC.
      function laikaAggressiveProInput(bot, target) {
        if (!target || !target.alive) {
          const esc = poisonEscape(bot);
          return esc || { throttle: 0, turn: 0, fire: false };
        }
        // (1) poison priority: get into the safe zone first.
        const escape = poisonEscape(bot);
        if (escape) return escape;

        const dist = Math.hypot(target.x - bot.x, target.y - bot.y);
        const direct = angleTo(target.x - bot.x, target.y - bot.y);
        const clearToFire = laikaCanFireNow(bot);
        const seeT = canSee(bot, target);

        // (5) dodge the MOST dangerous incoming shell (not merely the nearest).
        const threats = state.shells
          .filter((s) => s.owner !== bot.id)
          .map((s) => shellDangerFor(bot, s))
          .filter((t) => t.urgency > 0.3)
          .sort((a, b) => b.urgency - a.urgency);
        const top = threats[0];
        if (top && (top.lethal || top.urgency > LAIKA_PRO.dodgeUrgency)) {
          const evade = laikaEvadeAngle(bot, threats.slice(0, 3));
          const inp = evadeToward(bot, evade, 1);
          inp.fire = !top.lethal && seeT && clearToFire && Math.abs(angleDiff(bot.angle, direct)) < 0.08;
          return inp;
        }

        // (7) low health: grab a near shield if any, else fall back toward the safe centre.
        // retreatHealth is given at the 3-HP scale; scale by maxHealth/MAX_HEALTH so the bot retreats at the
        // same HP FRACTION under survival_v2 (HP x2) instead of waiting until a much lower fraction.
        if (bot.health <= LAIKA_PRO.retreatHealth * bot.maxHealth / MAX_HEALTH) {
          const shield = nearbyPowerup(bot, ["shield"], LAIKA_PRO.shieldGrabDist);
          if (shield) {
            const wp = pathWaypointFor(bot, shield);
            const inp = steerToward(bot, angleTo(wp.x - bot.x, wp.y - bot.y), 1);
            inp.fire = false;
            return inp;
          }
          const sr = state.safeRect;
          const cx = sr ? sr.x + sr.w / 2 : worldW / 2, cy = sr ? sr.y + sr.h / 2 : worldH / 2;
          const away = dist < LAIKA_PRO.preferLo ? angleTo(bot.x - target.x, bot.y - target.y)
                                                 : angleTo(cx - bot.x, cy - bot.y);
          const inp = steerToward(bot, away, 0.95);
          const shotR = findLaikaShot(bot, target);   // chip damage if a clean shot lines up
          inp.fire = !!shotR && shotR.direct && seeT && clearToFire && Math.abs(angleDiff(bot.angle, shotR.angle)) < 0.12;
          return inp;
        }

        const shot = findLaikaShot(bot, target);

        // (2) distance bands -> base movement intent.
        let moveAngle, pace;
        if (dist > LAIKA_PRO.tooFar) {
          moveAngle = direct; pace = 1;                                    // close in
        } else if (dist < LAIKA_PRO.tooClose) {
          const s = Math.sin(state.elapsed * 2.7 + bot.id * 1.3) > 0 ? 1 : -1;
          moveAngle = direct + Math.PI + s * 0.5; pace = 0.85;             // peel out (no faceplant)
        } else {
          const s = Math.sin(state.elapsed * 2.2 + bot.id * 1.7) > 0 ? 1 : -1;
          moveAngle = direct + s * (Math.PI / 2) * 0.7;                    // strafe in the firing band
          pace = dist > LAIKA_PRO.preferHi ? 0.7 : 0.45;
        }

        // (6) divert for a valuable, nearby powerup only when no shot is available.
        const goal = aggressiveLaikaGoal(bot, target);                    // target, or a close valued powerup
        const wantPU = (goal !== target && Math.hypot(goal.x - bot.x, goal.y - bot.y) <= LAIKA_PRO.puGrabDist) ? goal : null;

        // (3,4) aim + fire: align to the shot, fire only when safe + aligned.
        let desired = moveAngle, fire = false;
        if (shot && dist < 900) {
          const aimErr = Math.abs(angleDiff(bot.angle, shot.angle));
          const thr = shot.direct ? LAIKA_PRO.fireAngle : LAIKA_PRO.fireAngleBounce;
          if (shot.direct || aimErr < 0.6) desired = shot.angle;          // turn onto the shot
          fire = aimErr < thr && clearToFire && (shot.direct ? seeT : true);
          if (aimErr < 0.3) pace = Math.min(pace, dist < LAIKA_PRO.tooClose ? 0.3 : 0.42);
          if (!clearToFire && aimErr < 0.2) pace = Math.min(pace, 0.25);  // settle while reloading
        } else if (wantPU) {
          const wp = pathWaypointFor(bot, wantPU);
          desired = angleTo(wp.x - bot.x, wp.y - bot.y); pace = 1;
        } else if (!seeT) {
          const wp = pathWaypointFor(bot, target);                        // no LOS: path to the enemy
          desired = angleTo(wp.x - bot.x, wp.y - bot.y); pace = 1;
        }

        const inp = steerToward(bot, desired, pace);
        inp.fire = fire;
        return inp;
      }
      function botInput(bot, target) {
        if (!target.alive) return { throttle: 0, turn: 0, fire: false };
        const direct = angleTo(target.x - bot.x, target.y - bot.y);
        let desired = direct, fire = false;
        const nearestShell = state.shells
          .filter((shell) => shell.owner !== bot.id)
          .map((shell) => ({ shell, d: distSq(shell, bot) }))
          .sort((a, b) => a.d - b.d)[0];
        if (nearestShell && nearestShell.d < 130 * 130) {
          const incoming = angleTo(nearestShell.shell.vx, nearestShell.shell.vy);
          desired = incoming + Math.PI / 2;
          if (Math.sin(state.elapsed * 2.9412 + bot.id) < 0) desired += Math.PI;
        } else if (canSee(bot, target)) {
          fire = Math.abs(angleDiff(bot.angle, direct)) < 0.18;
        } else {
          const dx = target.x - bot.x, dy = target.y - bot.y;
          if (Math.abs(dx) > Math.abs(dy)) desired = dx > 0 ? 0 : Math.PI;
          else desired = dy > 0 ? Math.PI / 2 : -Math.PI / 2;
        }
        const diff = angleDiff(bot.angle, desired);
        return { throttle: Math.abs(diff) < 1.2 ? 0.85 : 0.15, turn: diff > 0.08 ? 1 : diff < -0.08 ? -1 : 0, fire };
      }

      function pathfindControl(me, opp, goal) {
        const chasingEnemy = !goal;
        const target = goal || { x: opp.x, y: opp.y };
        const start = cellOfGrid(me.x, me.y);
        const gcell = cellOfGrid(target.x, target.y);
        let aim;
        if (start.cx === gcell.cx && start.cy === gcell.cy) {
          aim = target;
        } else {
          const path = findPathCells(start, gcell);
          let wp = path && path.length > 1 ? path[1] : target;
          // Only advance to the next waypoint once nearly centred on this one, so the
          // tank passes through the corridor opening instead of cutting the corner.
          if (path && path.length > 2 && Math.hypot(wp.x - me.x, wp.y - me.y) < CELL_W * 0.22) wp = path[2];
          aim = wp;
        }
        const desired = angleTo(aim.x - me.x, aim.y - me.y);
        const diff = angleDiff(me.angle, desired);
        const turn = diff > 0.06 ? 1 : diff < -0.06 ? -1 : 0;
        // Crawl through turns; only commit to full speed when well aligned -> no wall clipping.
        const absd = Math.abs(diff);
        const throttle = absd < 0.3 ? 1 : absd < 0.8 ? 0.45 : 0.12;
        let fire = false;
        if (chasingEnemy && opp.alive && canSee(me, opp)) {
          const aimErr = Math.abs(angleDiff(me.angle, angleTo(opp.x - me.x, opp.y - me.y)));
          fire = aimErr < 0.12;
        }
        return { throttle, turn, fire };
      }

      // Plan a pixel-space waypoint list for the nav-path debug overlay.
      function planPath(fromX, fromY, toX, toY) {
        const start = cellOfGrid(fromX, fromY);
        const gcell = cellOfGrid(toX, toY);
        if (start.cx === gcell.cx && start.cy === gcell.cy) return [{ x: fromX, y: fromY }, { x: toX, y: toY }];
        return findPathCells(start, gcell);
      }

      function turretControl(me, opp) {
        if (!opp.alive) return { throttle: 0, turn: 0, fire: false };
        const desired = angleTo(opp.x - me.x, opp.y - me.y);
        const diff = angleDiff(me.angle, desired);
        const turn = clamp(diff * TURRET.turnGain, -1, 1);
        const visible = canSee(me, opp);
        const aligned = Math.abs(diff) <= TURRET.fireAngle;
        const cooldownReady = state.elapsed >= state.turretNextFireAt;
        let fire = false;
        if (visible && aligned && cooldownReady && me.reload <= 0) {
          fire = true;
          state.turretNextFireAt = state.elapsed + TURRET.cooldown;
        }
        return { throttle: 0, turn, fire };
      }

      let heuristicWarned = false;
      function heuristicFallback(me, opp) {
        if (!heuristicWarned) {
          heuristicWarned = true;
          if (typeof console !== "undefined" && console.warn) {
            console.warn('[ricochet] opponent "heuristic" was removed; falling back to "laika".');
          }
        }
        return laikaInput(me, opp);
      }

      // Stage-1 combat-curriculum opponent: full laika behaviour, heavily dampened.
      function easyLaikaInput(bot, target) {
        if (!target || !target.alive) {
          const e = poisonEscape(bot);
          return e ? { throttle: e.throttle, turn: e.turn * EASY_LAIKA.turnScale, fire: false } : { throttle: 0, turn: 0, fire: false };
        }
        // (1) poison priority: escape at full pace so it reliably survives the ring.
        const esc = poisonEscape(bot);
        if (esc) return { throttle: esc.throttle, turn: esc.turn * EASY_LAIKA.turnScale, fire: false };
        const dist = Math.hypot(target.x - bot.x, target.y - bot.y);
        const direct = angleTo(target.x - bot.x, target.y - bot.y);
        // (2) low health: ease back toward the safe-zone centre, hold fire. (retreatHealth scaled to HP fraction)
        if (bot.health <= EASY_LAIKA.retreatHealth * bot.maxHealth / MAX_HEALTH) {
          const sr = state.safeRect;
          const cx = sr ? sr.x + sr.w / 2 : worldW / 2, cy = sr ? sr.y + sr.h / 2 : worldH / 2;
          const s = steerToward(bot, angleTo(cx - bot.x, cy - bot.y), 0.9);
          return { throttle: s.throttle * EASY_LAIKA.moveScale, turn: s.turn * EASY_LAIKA.turnScale, fire: false };
        }
        // (3-6) reuse laika for wall-aware steering + shell dodging, then dampen it.
        const base = laikaInput(bot, target);
        let throttle = base.throttle * EASY_LAIKA.moveScale;
        let turn = base.turn * EASY_LAIKA.turnScale;
        if (throttle > 0 && dist > EASY_LAIKA.preferredDistance) throttle *= (0.5 + EASY_LAIKA.aggression);  // half-hearted chase
        else if (throttle > 0 && dist < EASY_LAIKA.preferredDistance) throttle = -0.45 * EASY_LAIKA.moveScale;  // don't faceplant
        // low-frequency, rough-aim fire (fireAngle + 3x cooldown), self-bounce-safe via laikaCanFireNow
        let fire = false;
        if (canSee(bot, target) && Math.abs(angleDiff(bot.angle, direct)) < EASY_LAIKA.fireAngle &&
            laikaCanFireNow(bot) && state.elapsed - state.easyFireAt >= bot.fireDelay * EASY_LAIKA.fireCooldownMult) {
          fire = true; state.easyFireAt = state.elapsed;
        }
        return { throttle, turn, fire };
      }

      // Shooting-gallery target: strafe VERTICALLY at a controlled speed, never fire, bounce off
      // top/bottom. Transverse velocity = moverSpeed * TANK_MOVE_SPEED for a side-on shooter, so a
      // sweep of cfg.moverSpeed isolates "can the shooter lead a moving target" from full-moba noise.
      function moverInput(me) {
        const speed = clamp(cfg.moverSpeed === undefined ? 1.0 : cfg.moverSpeed, 0, 1);
        if (me._moverSign === undefined) me._moverSign = -1;     // -1 = up (-y), +1 = down (+y)
        if (me.y <= 40) me._moverSign = 1;
        else if (me.y >= H - 40) me._moverSign = -1;
        const desired = me._moverSign < 0 ? -Math.PI / 2 : Math.PI / 2;
        const err = angleDiff(me.angle, desired);
        const turn = clamp(err * 3.0, -1, 1);
        const throttle = Math.abs(err) < 0.3 ? speed : 0;        // drive only once aligned -> clean vertical strafe
        return { throttle, turn, fire: false };
      }

      // ---------------- parameterized "weird" opponents (generalization training/eval) ----------------
      // One knob-driven bot composing the existing aim/dodge/escape primitives, so a self-play league can
      // face a BEHAVIOURALLY DIVERSE pool instead of 4 fixed scripts -> forces general skill over per-script
      // counters. The policy never sees an opponent id; it must read behaviour from the obs.
      const PARAM_PRESETS = {
        "p-rusher":    { aggression: 0.8, preferredDist: 60,  fireTol: 0.16, fireProb: 0.9, dodge: 0.3,  retreatHealth: 0,   powerupPriority: 1.0, powerupReach: 900, kite: false, wallBounce: false, strafe: false },
        "p-kiter":     { aggression: 0.25,preferredDist: 430, fireTol: 0.12, fireProb: 0.8, dodge: 0.85, retreatHealth: 0,   powerupPriority: 0.2, powerupReach: 300, kite: true,  wallBounce: false, strafe: true  },
        "wall-sniper": { aggression: 0.2, preferredDist: 520, fireTol: 0.09, fireProb: 0.7, dodge: 0.6,  retreatHealth: 0,   powerupPriority: 0.1, powerupReach: 250, kite: false, wallBounce: true,  strafe: false },
        "charger":     { aggression: 1.0, preferredDist: 0,   fireTol: 0.3,  fireProb: 1.0, dodge: 0.1,  retreatHealth: 0,   powerupPriority: 0,   powerupReach: 0,   kite: false, wallBounce: false, strafe: false },
        "precision":   { aggression: 0.4, preferredDist: 320, fireTol: 0.06, fireProb: 0.6, dodge: 0.5,  retreatHealth: 0,   powerupPriority: 0.2, powerupReach: 300, kite: false, wallBounce: false, strafe: false },
        // "ace" = the HUMAN-STYLE threat that actually pressures the agent: AGGRESSIVE point-blank brawler that also
        // CONTESTS POWERUPS (rapid/triple/laser = the human's "shotgun/machine-gun"), strafes to dodge, only retreats
        // near-dead. Aggression beats evasion under v2, so the tough opponent is aggressive+resourceful, not a kiter.
        "ace":         { aggression: 0.85,preferredDist: 90,  fireTol: 0.12, fireProb: 0.95,dodge: 0.6,  retreatHealth: 1,   powerupPriority: 0.9, powerupReach: 760, kite: false, wallBounce: false, strafe: true  },
        "spammer":     { aggression: 0.6, preferredDist: 180, fireTol: 0.4,  fireProb: 1.0, dodge: 0.3,  retreatHealth: 0,   powerupPriority: 0.1, powerupReach: 250, kite: false, wallBounce: false, strafe: true  },
        "counter":     { aggression: 0.5, preferredDist: 250, fireTol: 0.12, fireProb: 0.8, dodge: 0.6,  retreatHealth: 2.0, powerupPriority: 0.3, powerupReach: 350, kite: false, wallBounce: false, strafe: false },
        "baiter":      { aggression: 0.2, preferredDist: 450, fireTol: 0.1,  fireProb: 0.5, dodge: 0.7,  retreatHealth: 0,   powerupPriority: 0.3, powerupReach: 400, kite: true,  wallBounce: false, strafe: false },
        "turtle":      { aggression: 0.15,preferredDist: 400, fireTol: 0.1,  fireProb: 0.6, dodge: 0.9,  retreatHealth: 1.5, powerupPriority: 0.2, powerupReach: 300, kite: false, wallBounce: false, strafe: false }
      };
      function sampleRandomParams(me) {
        let s = (state.arenaSeed ^ (me.id * 0x9e3779b9)) >>> 0;        // own LCG, seeded by the arena -> deterministic, no gameRng side-effects
        s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) ^ (s >>> 13)) >>> 0;  // avalanche-mix the seed so SEQUENTIAL arena seeds don't draw correlated archetypes
        const r = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
        // Draw an ARCHETYPE so the family FREQUENTLY presents the hard styles the policy generalises worst to
        // (spammer/baiter/wall-sniper) rather than only hitting those extremes in the tails of a uniform draw.
        const arch = r();
        let P;
        if (arch < 0.22) {          // spammer-like: high-fire volume, close-mid
          P = { aggression: 0.5 + r() * 0.5, preferredDist: 120 + r() * 260, fireTol: 0.22 + r() * 0.22, fireProb: 0.85 + r() * 0.15, dodge: r() * 0.5 };
        } else if (arch < 0.40) {   // baiter-like: passive, evasive, picky fire, keeps distance
          P = { aggression: 0.15 + r() * 0.25, preferredDist: 360 + r() * 200, fireTol: 0.05 + r() * 0.12, fireProb: 0.35 + r() * 0.35, dodge: 0.6 + r() * 0.4, kite: r() < 0.6 };
        } else if (arch < 0.56) {   // wall-sniper-like: bounce shots, mid-long range, precise
          P = { aggression: 0.2 + r() * 0.4, preferredDist: 280 + r() * 240, fireTol: 0.05 + r() * 0.13, fireProb: 0.55 + r() * 0.35, dodge: r() * 0.6, wallBounce: true };
        } else {                    // broad uniform (the original wide draw) -> covers everything in between
          P = { aggression: 0.15 + r() * 0.85, preferredDist: r() * 520, fireTol: 0.05 + r() * 0.38, fireProb: 0.35 + r() * 0.65, dodge: r() };
        }
        P.retreatHealth = r() < 0.4 ? 1 + r() * 1.2 : 0;       // shared knobs (fill any the archetype left unset)
        P.powerupPriority = r();
        P.powerupReach = 150 + r() * 750;
        if (P.kite === undefined) P.kite = r() < 0.4;
        if (P.wallBounce === undefined) P.wallBounce = r() < 0.3;
        P.strafe = r() < 0.5;
        P.ignorePoison = r() < 0.15;
        return P;
      }
      function botRand(me, salt) {  // deterministic pseudo-random in [0,1) from sim time + bot phase (fire cadence/strafe)
        const x = Math.sin(state.elapsed * 12.9898 + me.id * 78.233 + (salt || 0) * 37.71) * 43758.5453;
        return x - Math.floor(x);
      }
      function paramBot(me, opp, P) {
        if (!opp.alive) return { throttle: 0, turn: 0, fire: false };
        if (!P.ignorePoison) { const esc = poisonEscape(me); if (esc) return esc; }
        const dist = Math.hypot(opp.x - me.x, opp.y - me.y);
        const direct = angleTo(opp.x - me.x, opp.y - me.y);
        if (P.powerupPriority > 0 && state.powerups.length) {                 // contest a nearby powerup
          let best = null, bd = Infinity;
          for (const pu of state.powerups) { const d = Math.hypot(pu.x - me.x, pu.y - me.y); if (d < bd) { bd = d; best = pu; } }
          if (best && bd < P.powerupReach && P.powerupPriority > botRand(me, 1)) {
            const inp = steerToward(me, angleTo(best.x - me.x, best.y - me.y), 1); inp.fire = false; return inp;
          }
        }
        const threats = state.shells.filter((s) => s.owner !== me.id).map((s) => shellDangerFor(me, s))
          .filter((t) => t.urgency > 0.14).sort((a, b) => b.urgency - a.urgency);
        if (threats.length && threats[0].urgency > 0.85 - P.dodge * 0.6) {     // dodge (sensitivity = P.dodge)
          const inp = evadeToward(me, laikaEvadeAngle(me, threats.slice(0, 3)), threats[0].lethal ? 1 : 0.9); inp.fire = false; return inp;
        }
        if (me.health <= P.retreatHealth * me.maxHealth / MAX_HEALTH && dist < P.preferredDist + 120) {    // retreat-then-counter (HP-fraction scaled)
          const inp = steerToward(me, direct, -0.85);
          inp.fire = laikaCanFireNow(me) && Math.abs(angleDiff(me.angle, direct)) < P.fireTol; return inp;
        }
        const shot = findLaikaShot(me, opp);   // LEAD-shooting solution (predicts motion, CLEAR line / valid bounce)
        const canShoot = shot && shot.angle != null && (shot.direct || P.wallBounce);   // only fire viable shots -> no wall self-kill
        const aimAngle = canShoot ? shot.angle : direct;
        const aimError = Math.abs(angleDiff(me.angle, aimAngle));
        let throttle = dist > P.preferredDist + 40 ? P.aggression : (P.kite ? -0.6 : 0.08);   // distance control
        const inp = steerToward(me, aimAngle, throttle);
        if (P.strafe && dist < 320) inp.turn += (botRand(me, 2) > 0.5 ? 1 : -1) * 0.35;
        inp.fire = canShoot && aimError < P.fireTol && laikaCanFireNow(me) && (P.fireProb >= 1 || botRand(me, 3) < P.fireProb);
        return inp;
      }

      function scriptedControl(playerId, kind, opts) {
        const me = state.tanks[playerId];
        const opp = state.tanks[1 - playerId];
        if (!me || !opp) return { throttle: 0, turn: 0, fire: false };
        switch (kind) {
          case "none":         // navigation mode: red is inert and does not fight
          case "stationary": return { throttle: 0, turn: 0, fire: false };
          case "mover": return moverInput(me);
          case "turret":
          case "slow-turret":
          case "turret-slow": return turretControl(me, opp);
          case "heuristic": return heuristicFallback(me, opp);
          case "easy_laika": return easyLaikaInput(me, opp);
          case "laika-aggressive-pro":
          case "pro": return laikaAggressiveProInput(me, opp);
          case "laika": return laikaInput(me, opp);
          case "laika-aggressive":
          case "aggressive": return aggressiveLaikaInput(me, opp);
          case "p-rusher": case "p-kiter": case "wall-sniper": case "charger":
          case "precision": case "spammer": case "counter": case "baiter": case "turtle": case "ace":
            return paramBot(me, opp, PARAM_PRESETS[kind]);
          case "randomized":
            if (!me._randP) me._randP = sampleRandomParams(me);
            return paramBot(me, opp, me._randP);
          case "pathfind": return pathfindControl(me, opp, opts && opts.goal);
          default: return laikaInput(me, opp);
        }
      }

      // ---------------- observation ----------------
      function buildObservation(playerId) {
        const me = state.tanks[playerId];
        const other = state.tanks[1 - playerId];
        const dx = other.x - me.x, dy = other.y - me.y;
        const bearing = angleDiff(me.angle, angleTo(dx, dy));
        const heading = angleDiff(me.angle, other.angle);
        const values = [
          clamp((me.x / worldW) * 2.0 - 1.0, -1.0, 1.0),
          clamp((me.y / worldH) * 2.0 - 1.0, -1.0, 1.0),
          Math.sin(me.angle), Math.cos(me.angle),
          clamp(me.reload / FIRE_DELAY, 0.0, 1.0),
          clamp(me.health / me.maxHealth, 0.0, 1.0),   // fraction of max HP -> obs stays [0,1] under survival_v2 HP x2
          clamp(me.shield / 2.0, 0.0, 1.0),
          clamp(Math.max(me.powerTimer, me.shieldTimer) / SHIELD_DURATION, 0.0, 1.0),
          clamp(me.powerShots / 2.0, 0.0, 1.0)
        ];
        values.push(...powerOneHot(me.power));
        if (isNav) {
          values.push(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);  // lesson 1: no opponent -> zero all 10 enemy features
        } else {
          values.push(
            clamp(dx / worldW, -1.0, 1.0), clamp(dy / worldH, -1.0, 1.0),
            clamp(Math.hypot(dx, dy) / worldDiag, 0.0, 1.0),
            Math.sin(bearing), Math.cos(bearing), Math.sin(heading), Math.cos(heading),
            clamp(other.reload / FIRE_DELAY, 0.0, 1.0),
            clamp(other.health / other.maxHealth, 0.0, 1.0),   // enemy HP as a fraction -> [0,1] under survival_v2
            clamp(other.shield / 2.0, 0.0, 1.0)
          );
        }
        for (const offset of RAY_OFFSETS) {
          values.push(clamp(raycast(me.x, me.y, me.angle + offset) / RAY_MAX, 0.0, 1.0));
        }
        const shells = state.shells
          .map((shell) => ({ shell, d: (shell.x - me.x) ** 2 + (shell.y - me.y) ** 2 }))
          .sort((a, b) => a.d - b.d)
          .slice(0, MAX_SHELL_FEATURES);
        for (const { shell } of shells) {
          values.push(
            1.0, shell.owner === me.id ? 1.0 : -1.0,
            clamp((shell.x - me.x) / worldW, -1.0, 1.0), clamp((shell.y - me.y) / worldH, -1.0, 1.0),
            clamp(shell.vx / SHELL_SPEED, -1.0, 1.0), clamp(shell.vy / SHELL_SPEED, -1.0, 1.0),
            clamp(shell.ttl / SHELL_TTL, 0.0, 1.0)
          );
        }
        for (let i = 0; i < MAX_SHELL_FEATURES - shells.length; i++) values.push(0, 0, 0, 0, 0, 0, 0);
        let closest = null, closestD = Infinity;
        for (const power of state.powerups) {
          const d = (power.x - me.x) ** 2 + (power.y - me.y) ** 2;
          if (d < closestD) { closestD = d; closest = power; }
        }
        if (closest === null) {
          values.push(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        } else {
          values.push(
            1.0, clamp((closest.x - me.x) / worldW, -1.0, 1.0), clamp((closest.y - me.y) / worldH, -1.0, 1.0),
            clamp(closest.ttl / POWERUP_FIELD_TTL, 0.0, 1.0)
          );
          values.push(...powerOneHot(closest.type));
        }
        if (state.roundState && state.roundState.phase === "pending") {
          values.push(clamp(state.roundState.timer / DRAW_WINDOW, 0.0, 1.0), state.roundState.winnerId === me.id ? 1.0 : -1.0);
        } else {
          values.push(0, 0);
        }
        // --- motion (4 dims): self + enemy velocity in the agent's HEADING frame [forward, lateral], normalized.
        // The signal the campaign was missing ("needs motion obs"): lets the policy ANTICIPATE where an evasive/juking
        // target is going (lead the shot) and read a rusher's closing rate, not just a static snapshot. Ego-relative,
        // identity-blind. Kept BEFORE the poison block so the survival poison/safe block stays the trailing 11 dims. ---
        {
          const ego = (vx, vy) => [vx * Math.cos(me.angle) + vy * Math.sin(me.angle), -vx * Math.sin(me.angle) + vy * Math.cos(me.angle)];
          const [mvf, mvl] = ego(me.vx || 0, me.vy || 0);
          const [ovf, ovl] = isNav ? [0, 0] : ego(other.vx || 0, other.vy || 0);
          values.push(
            clamp(mvf / TANK_MOVE_SPEED, -1.0, 1.0), clamp(mvl / TANK_MOVE_SPEED, -1.0, 1.0),
            clamp(ovf / TANK_MOVE_SPEED, -1.0, 1.0), clamp(ovl / TANK_MOVE_SPEED, -1.0, 1.0)
          );
        }
        // --- survival poison / safe-zone awareness (11 dims; all zero in open/maze) ---
        // order: [ poisonActive, insideSafeRect, safeCenterDx, safeCenterDy, safeCenterDistance,
        //          sinSafeBearing, cosSafeBearing, safeRectWidth, safeRectHeight, safeEdgeMargin, timeToPoisonStart ]
        if (cfg.arenaMode === "survival" && state.safeRect) {
          const sr = state.safeRect;
          const scx = sr.x + sr.w / 2, scy = sr.y + sr.h / 2;
          const sdx = scx - me.x, sdy = scy - me.y;
          const safeBearing = angleDiff(me.angle, angleTo(sdx, sdy));
          const inside = inSafeRect(me);
          // edge margin: + = distance to nearest safe edge while inside, - = how far into the poison while outside
          const insideMargin = Math.min(me.x - sr.x, sr.x + sr.w - me.x, me.y - sr.y, sr.y + sr.h - me.y);
          const ox = Math.max(sr.x - me.x, me.x - (sr.x + sr.w), 0);
          const oy = Math.max(sr.y - me.y, me.y - (sr.y + sr.h), 0);
          const edgeMargin = inside ? insideMargin : -Math.hypot(ox, oy);
          const timeToPoison = (state.poisonActive || !cfg.poisonEnabled) ? 0.0 : clamp((POISON.startTime - state.elapsed) / POISON.startTime, 0.0, 1.0);
          values.push(
            state.poisonActive ? 1.0 : 0.0,
            inside ? 1.0 : -1.0,
            clamp(sdx / worldW, -1.0, 1.0), clamp(sdy / worldH, -1.0, 1.0),
            clamp(Math.hypot(sdx, sdy) / worldDiag, 0.0, 1.0),
            Math.sin(safeBearing), Math.cos(safeBearing),
            clamp(sr.w / worldW, 0.0, 1.0), clamp(sr.h / worldH, 0.0, 1.0),
            clamp(edgeMargin / (CELL_W * 3), -1.0, 1.0),
            timeToPoison
          );
        } else {
          values.push(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }
        for (let i = 0; i < values.length; i++) values[i] = clamp(values[i], -1.0, 1.0);
        return values;
      }

      function getPublicState() {
        return {
          constants: {
            width: W, height: H, wall: WALL, cols: COLS, rows: ROWS,
            cellW: CELL_W, cellH: CELL_H, tankRadius: TANK_RADIUS, maxHealth: cfg.tankMaxHp,
            obsSize: OBS_SIZE, actionSize: ACTION_TABLE.length, wallRayCount: WALL_RAY_COUNT, drawWindow: DRAW_WINDOW,
            arenaMode: cfg.arenaMode, spawnPowerups: cfg.spawnPowerups,
            spawnJitter: cfg.spawnJitter, openJitter: { ...OPEN_JITTER },
            worldW, worldH, viewW: W, viewH: H,
            poisonEnabled: cfg.poisonEnabled, poisonStartTime: POISON.startTime,
            poisonShrinkSpeedX: POISON.shrinkSpeedX, poisonShrinkSpeedY: POISON.shrinkSpeedY,
            poisonDamagePerSecond: POISON.dps,
            // time for the ring to finish shrinking onto centerClear (continuous shrink, the slower axis wins)
            poisonEstimatedMinCircleTime: POISON.startTime
              + Math.max(((WORLD_W - WORLD_CENTER_CLEAR.w) / 2) / POISON.shrinkSpeedX,
                         ((WORLD_H - WORLD_CENTER_CLEAR.h) / 2) / POISON.shrinkSpeedY),
            poisonSuccessExtraSeconds: 12,
            shellDecay: cfg.shellDecay, shellBoost: SHELL_BOOST, shellDecayDuration: SHELL_DECAY_DURATION
          },
          arenaMode: cfg.arenaMode,
          scenario: cfg.scenario,
          spawnPowerups: cfg.spawnPowerups,
          spawnJitter: cfg.spawnJitter,
          worldW, worldH, viewW: W, viewH: H,
          centerClear: state.centerClear ? { ...state.centerClear } : null,
          poison: cfg.poisonEnabled
            ? { active: state.poisonActive, atMinCircle: state.poisonAtMinCircle, safeRect: state.safeRect ? { ...state.safeRect } : null, nextShrinkTime: state.nextShrinkTime, dps: POISON.dps }
            : null,
          nav: isNav && state.nav ? { ...state.nav } : null,
          seed: state.arenaSeed,
          elapsed: state.elapsed, stepCount: state.stepCount,
          done: state.done, result: state.result,
          roundState: state.roundState ? { ...state.roundState } : null,
          walls: state.walls.map((wall) => ({ ...wall })),
          maze: state.maze ? state.maze.map((row) => row.map((cell) => ({ x: cell.x, y: cell.y, walls: { ...cell.walls } }))) : null,
          tanks: state.tanks.map((tank) => ({ ...tank })),
          shells: state.shells.map((shell) => ({
            x: shell.x, y: shell.y, vx: shell.vx, vy: shell.vy, owner: shell.owner,
            radius: shell.radius, ttl: shell.ttl, bounces: shell.bounces, type: shell.type, age: shell.age
          })),
          powerups: state.powerups.map((power) => ({ ...power })),
          powerupTimer: state.powerupTimer
        };
      }

      function normalizeControl(input, playerId) {
        if (input === null || input === undefined) return { throttle: 0, turn: 0, fire: false };
        if (typeof input === "number") return actionToControl(input);
        if (typeof input === "string") return scriptedControl(playerId, input);
        return { throttle: input.throttle || 0, turn: input.turn || 0, fire: Boolean(input.fire) };
      }

      function applyControls(input0, input1) {
        state.controls[0] = normalizeControl(input0, 0);
        state.controls[1] = normalizeControl(input1, 1);
      }

      function info() {
        const out = {
          seed: state.arenaSeed, arenaMode: cfg.arenaMode, scenario: cfg.scenario, spawnJitter: cfg.spawnJitter,
          spawnPowerups: cfg.spawnPowerups, result: state.result, steps: state.stepCount,
          obsSize: OBS_SIZE, actionSize: ACTION_TABLE.length,
          elapsed: Math.round(state.elapsed * 1000) / 1000,
          done: state.done, truncated: state.truncated,
          learnerAlive: state.tanks[0] ? state.tanks[0].alive : false,
          opponentAlive: state.tanks[1] ? state.tanks[1].alive : false
        };
        if (usesRouteReward && state.nav) {
          out.centerStayTime = Math.round(state.nav.centerStayTime * 1000) / 1000;
          out.enteredCenter = state.nav.enteredCenter;
          out.wallHits = state.nav.wallHits;
          out.newCells = state.nav.newCells;
          out.stuckEvents = state.nav.stuckEvents;
          out.noProgressEvents = state.nav.noProgressEvents;
          out.pathDist = Number.isFinite(state.nav.pathDist) ? state.nav.pathDist : -1;
          out.bestPathDist = Number.isFinite(state.nav.bestPathDist) ? state.nav.bestPathDist : -1;
          out.routeSuccess = state.nav.navSuccess;
          out.routeTimeout = state.result === "route_timeout";
          if (cfg.poisonEnabled) {   // moba_poison_run: surface poison status as well
            out.poisonActive = state.poisonActive;
            out.poisonDamageTaken = Math.round((state.nav.poisonDamageTaken || 0) * 1000) / 1000;
          }
        } else if (isPoisonNav && state.nav) {
          out.poisonActive = state.poisonActive;
          out.poisonAtMinCircle = state.poisonAtMinCircle;
          out.survivalAfterMinCircle = Math.round(state.nav.survivalAfterMinCircle * 1000) / 1000;
          out.wallHits = state.nav.wallHits;
          out.pickups = state.nav.pickups;
          out.pickupsWhenEmpty = state.nav.pickupsWhenEmpty;
          out.poisonDamageTaken = Math.round(state.nav.poisonDamageTaken * 1000) / 1000;
          out.pickedAnyPowerup = state.nav.pickedAnyPowerup;
          out.navSuccess = state.nav.navSuccess;
        } else if (state.combat) {
          out.hitsDealt = state.combat.hitsDealt;
          out.hitsTaken = state.combat.hitsTaken;
          out.selfHits = state.combat.selfHits;
          out.powerups = state.combat.powerups;
          out.poisonDamageTaken = Math.round(state.combat.poisonDamageTaken * 1000) / 1000;
          out.learnerHealth = state.tanks[0] ? Math.round(state.tanks[0].health * 1000) / 1000 : 0;
          out.opponentHealth = state.tanks[1] ? Math.round(state.tanks[1].health * 1000) / 1000 : 0;
          out.shotsFired = state.combat.shotsFired;
          out.enemyPowerups = state.combat.enemyPowerups;
          out.contactSteps = state.combat.contactSteps;
          out.deathCause = state.combat.deathCause;
          out.loserId = state.combat.loserId;
        }
        return out;
      }

      // ---------------- public API ----------------
      this.reset = (seed) => {
        state.arenaSeed = (seed !== undefined && seed !== null) ? (seed >>> 0) : pickSeed();
        gameRng = new RNG((state.arenaSeed * 2654435761) >>> 0);
        let p1, p2, p1Angle, p2Angle;
        state.centerClear = null;
        state.safeRect = null;
        state.poisonActive = false;
        state.nextShrinkTime = 0;
        if (cfg.arenaMode === "open") {
          state.maze = null;
          state.walls = boundaryWalls();
          if (cfg.randomTurret) {
            // Shooting-lab: drop the turret (red) at a uniform-random position each episode
            // (seeded RNG -> reproducible) so the expert aims from varied geometry. Blue
            // starts left-centre facing the turret; both kept clear of the boundary walls.
            const uni = (lo, hi) => lo + gameRng.next() * (hi - lo);
            p1 = { x: W * 0.15, y: H * 0.5 };
            p2 = { x: uni(OPEN_TURRET.xMin, OPEN_TURRET.xMax), y: uni(OPEN_TURRET.yMin, OPEN_TURRET.yMax) };
            // Blue starts at a RANDOM heading (not pre-aimed) so it must turn to acquire the
            // turret -> the demos contain aim-acquisition, not just fire-from-spawn.
            p1Angle = (gameRng.next() * 2 - 1) * Math.PI;
            p2Angle = angleTo(p1.x - p2.x, p1.y - p2.y);   // turret faces blue
          } else if (cfg.spawnJitter) {
            // Uniform jitter in [-range, range] from the seeded RNG (reproducible
            // per seed). Ranges keep spawns well clear of the boundary walls.
            const jit = (range) => (gameRng.next() * 2 - 1) * range;
            p1 = { x: W * 0.25, y: H * 0.5 + jit(OPEN_JITTER.blueY) };
            p2 = { x: W * 0.75, y: H * 0.5 + jit(OPEN_JITTER.redY) };
            p1Angle = jit(OPEN_JITTER.blueAngle);
            p2Angle = Math.PI;
          } else {
            p1 = { x: W * 0.25, y: H * 0.5 };
            p2 = { x: W * 0.75, y: H * 0.5 };
            p1Angle = 0;
            p2Angle = Math.PI;
          }
        } else if (cfg.arenaMode === "survival") {
          const world = usesFixedMap ? makeFixedMobaWorld()
            : makeSurvivalWorld(state.arenaSeed, 0.20, isNav);  // nav -> centerClear fully cleared
          state.maze = world.cells;
          state.walls = world.walls;
          state.centerClear = world.centerClear;
          state.safeRect = { x: 0, y: 0, w: worldW, h: worldH };
          state.nextShrinkTime = POISON.startTime;
          if (isFixedMoba || isMobaDuel) {
            if (cfg.spawnMode === "fixed") {
              // v1: mirror-symmetric bases near the left/right edges, facing each other.
              // x sums to WORLD_W so blue/red stay left-right symmetric.
              p1 = { x: WORLD_W * 0.12, y: WORLD_H * 0.5 };
              p2 = { x: WORLD_W * 0.88, y: WORLD_H * 0.5 };
              p1Angle = 0;
              p2Angle = Math.PI;
            } else if (cfg.spawnMode === "tri_fixed") {
              // 3 FIXED left points (top/mid/bottom, x=0.12W) x 3 mirror right points (x=0.88W). Blue and red
              // each pick one INDEPENDENTLY (seeded) -> 9 discrete openings, NOT always face-to-face. A small,
              // MEMORIZABLE opening set (unlike the continuous half_random that over-fits/caps laika ~0.3), so
              // per-opening tactics are learnable. Tanks face each other regardless of the lane mismatch.
              const ys = [WORLD_H * 0.25, WORLD_H * 0.5, WORLD_H * 0.75];
              const bi = Math.min(2, Math.floor(gameRng.next() * 3));
              const ri = Math.min(2, Math.floor(gameRng.next() * 3));
              p1 = { x: WORLD_W * 0.12, y: ys[bi] };
              p2 = { x: WORLD_W * 0.88, y: ys[ri] };
              p1Angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
              p2Angle = Math.atan2(p1.y - p2.y, p1.x - p2.x);
            } else {
              // survival_v2 RANDOM spawn (seeded). Legal = clear of boundary + walls (with clearance) and a
              // minimum separation between the two tanks. At spawn the poison ring is inactive (whole map safe)
              // and power-ups have not spawned yet (they spawn after the tanks and avoid them via collidesTank),
              // so wall + boundary + min-distance is the full legality test. half_random: blue LEFT / red RIGHT.
              const r = TANK_RADIUS, pad = 36, minSep = worldW * 0.32;
              const pickSpawn = (xLo, xHi, avoid) => {
                for (let a = 0; a < 240; a++) {
                  const x = xLo + gameRng.next() * (xHi - xLo);
                  const y = WALL + r + pad + gameRng.next() * (worldH - 2 * (WALL + r + pad));
                  if (x < r + WALL + pad || x > worldW - r - WALL - pad) continue;
                  if (state.walls.some((wall) => circleRectHit(x, y, r + 6, wall))) continue;   // wall + clearance
                  if (avoid && distSq({ x, y }, avoid) < minSep * minSep) continue;             // min spawn distance
                  return { x, y };
                }
                return { x: (xLo + xHi) / 2, y: worldH / 2 };   // safe-ish fallback (mid-lane)
              };
              const full = cfg.spawnMode === "full_random";
              p1 = pickSpawn(WALL + r + pad, full ? worldW - WALL - r - pad : worldW / 2 - pad, null);
              p2 = pickSpawn(full ? WALL + r + pad : worldW / 2 + pad, worldW - WALL - r - pad, p1);
              p1Angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);   // face each other
              p2Angle = Math.atan2(p1.y - p2.y, p1.x - p2.x);
            }
          } else {
            p1 = isNav ? navSpawnBlue() : cellCenter(1, SROWS - 2);  // nav: blue starts outside the centre (seeded)
            p2 = cellCenter(SCOLS - 2, 1);
            p1Angle = -Math.PI / 2;
            p2Angle = Math.PI / 2;
          }
        } else {
          state.maze = makeMaze(state.arenaSeed);
          state.walls = mazeToRects(state.maze);
          p1 = cellCenter(1, ROWS - 2);
          p2 = cellCenter(COLS - 2, 1);
          p1Angle = -Math.PI / 2;
          p2Angle = Math.PI / 2;
        }
        state.tanks = [
          createTank(0, p1.x, p1.y, p1Angle, "#58a6ff", "#b7dcff"),
          createTank(1, p2.x, p2.y, p2Angle, "#ff6b6b", "#ffd0d0")
        ];
        for (const t of state.tanks) { t.maxHealth = cfg.tankMaxHp; t.health = cfg.tankMaxHp; }  // survival_v2: HP pool
        state.shells = [];
        state.powerups = [];
        state.powerupTimer = 4.0;
        state.controls = [null, null];
        state.roundState = null;
        state.elapsed = 0;
        state.stepCount = 0;
        state.rewardDelta = 0;
        state.done = false;
        state.truncated = false;
        state.result = "running";
        state.events = [];
        state.turretNextFireAt = TURRET.initialDelay;
        state.easyFireAt = 0;
        state.combat = isNav ? null : { hitsDealt: 0, hitsTaken: 0, selfHits: 0, powerups: 0, poisonDamageTaken: 0,
          shotsFired: 0, enemyPowerups: 0, contactSteps: 0, deathCause: null, loserId: null };
        state.poisonAtMinCircle = false;
        if (isNav) state.tanks[1].alive = false;   // nav lessons: red is out of the fight
        state.routeDistField = null;
        if (usesRouteReward) {
          state.nav = {
            wallHits: 0, lastWallPenaltyAt: -1e9, navSuccess: false,
            centerStayTime: 0, enteredCenter: false, newCells: 0, stuckEvents: 0,
            noProgressEvents: 0, lastProgressAt: 0, poisonDamageTaken: 0,
            visited: new Set(), pathDist: Infinity, bestPathDist: Infinity,
            stuckAnchor: { x: p1.x, y: p1.y }, stuckAnchorElapsed: 0
          };
          state.routeDistField = computeCenterDistField();
        } else if (isPoisonNav) {
          state.nav = {
            survivalAfterMinCircle: 0, wallHits: 0, pickups: 0, pickupsWhenEmpty: 0,
            poisonDamageTaken: 0, pickedAnyPowerup: false, lastWallPenaltyAt: -1e9, navSuccess: false
          };
        } else {
          state.nav = null;
        }
        if (isPoisonNav && cfg.spawnPowerups) navSpawnInitialPowerup();
        const dxy = Math.hypot(p2.x - p1.x, p2.y - p1.y) / ARENA_DIAG;
        state._prevDist = dxy;
        return buildObservation(0);
      };

      // RL step: fixed dt, action_repeat substeps, returns reward/done.
      this.step = (input0, input1, opts) => {
        const dt = (opts && opts.dt) || cfg.stepDt;
        const repeat = (opts && opts.repeat) || cfg.actionRepeat;
        if (state.done) {
          return { obs0: buildObservation(0), obs1: buildObservation(1), reward: 0, done: true, truncated: state.truncated, info: info(), events: [] };
        }
        state.events = [];
        state.rewardDelta = 0;
        applyControls(input0, input1);
        for (let k = 0; k < repeat; k++) {
          update(dt);
          if (state.done) break;
        }
        if (usesRouteReward && !state.done) routeProgress();   // path-distance / exploration / stuck shaping, once per RL step
        state.stepCount += 1;
        if (!state.done && state.stepCount >= cfg.maxSteps) {
          state.done = true; state.truncated = true;
          if (isNav) { state.result = usesRouteReward ? "route_timeout" : "nav_timeout"; state.rewardDelta += navActive.timeoutPenalty; }
          else {
            // No combat "timeout": the closing poison ring always forces a death well before maxSteps.
            // If the cap is ever reached first, resolve by health (lower health = loss) so the outcome
            // is always a win/loss, never a timeout/draw.
            const a = state.tanks[0], b = state.tanks[1];
            const ha = a ? a.health : 0, hb = b ? b.health : 0;
            state.result = ha > hb ? "win" : hb > ha ? "loss" : "draw";
          }
        }
        let r = state.rewardDelta;
        if (!state.done) r += denseReward();
        return { obs0: buildObservation(0), obs1: buildObservation(1), reward: r, done: state.done, truncated: state.truncated, info: info(), events: state.events };
      };

      // Browser tick: single variable-dt update, returns public state + events.
      this.advance = (dt, input0, input1) => {
        state.events = [];
        applyControls(input0, input1);
        update(dt);
        return { publicState: getPublicState(), events: state.events, done: state.done, result: state.result };
      };

      this.getPublicState = getPublicState;
      this.buildObservation = buildObservation;
      this.observe = buildObservation;
      this.traceLaser = traceLaser;
      this.scriptedControl = scriptedControl;
      this.planPath = planPath;
      this.canSee = (a, b) => canSee(a, b);
      this.raycast = raycast;
      this.isDone = () => state.done;
      this.result = () => state.result;
      this.constants = {
        W, H, WALL, COLS, ROWS, CELL_W, CELL_H, TANK_RADIUS, TANK_SCALE: 0.8,
        MAX_HEALTH, FIRE_DELAY, ARENA_DIAG, TAU, OBS_SIZE, WALL_RAY_COUNT, RAY_MAX, DRAW_WINDOW, POWER_TYPES, POWERUP_META,
        arenaMode: cfg.arenaMode, spawnJitter: cfg.spawnJitter, openJitter: { ...OPEN_JITTER }, turret: { ...TURRET },
        worldW, worldH, viewW: W, viewH: H, WORLD_W, WORLD_H, SCOLS, SROWS, centerClear: WORLD_CENTER_CLEAR, poison: { ...POISON }
      };
    }
  }

  return {
    RicochetCore,
    ACTION_TABLE,
    OBS_SIZE,
    POWER_TYPES,
    actionToControl,
    controlToAction,
    keysToControl,
    policyForward,
    makeMaze,
    mazeToRects,
    makeSurvivalWorld,
    boundaryWalls,
    RAY_OFFSETS,
    constants: {
      W, H, WALL, COLS, ROWS, CELL_W, CELL_H, TANK_RADIUS, MAX_HEALTH, OBS_SIZE, WALL_RAY_COUNT, DRAW_WINDOW, RAY_MAX,
      VIEW_W, VIEW_H, WORLD_W, WORLD_H, SCOLS, SROWS, WORLD_CENTER_CLEAR, POISON,
      SHELL_SPEED, POWER_DURATION, SHIELD_DURATION, POWERUP_FIELD_TTL,
      SHELL_BOOST, SHELL_DECAY_DURATION
    }
  };
});
