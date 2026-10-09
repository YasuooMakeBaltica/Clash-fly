// JavaScript port of the Clash Royale simulator (flybrain/envs/clash/sim.py,
// strategy.py, env.py) and of the fly brain's decision pass (network.py,
// agent.py), so a trained brain can play in a browser. Keep in sync with the
// Python. Runs in browsers and in Node (for tests).
"use strict";

const ClashCore = (() => {
  // ------------------------------------------------------------ cards.py
  const card = (name, cost, kind, role, o = {}) => Object.assign(
    { name, cost, kind, role, count: 1, hp: 0, damage: 0, hit_speed: 1, range: 1, speed: 1, sight: 5.5,
      targets: "ground", radius: 0, tower_damage: 0, delay: 1 }, o);
  const DECK = [
    card("Knight", 3, "troop", "mini_tank", { hp: 1766, damage: 202, hit_speed: 1.2, range: 1.2, speed: 1.0 }),
    card("Archers", 3, "troop", "ranged", { count: 2, hp: 304, damage: 107, hit_speed: 0.9, range: 5.0, targets: "air_ground" }),
    card("Giant", 5, "troop", "tank", { hp: 4091, damage: 254, hit_speed: 1.5, range: 1.2, speed: 0.75, sight: 7.5, targets: "buildings" }),
    card("Musketeer", 4, "troop", "ranged", { hp: 720, damage: 218, hit_speed: 1.0, range: 6.0, targets: "air_ground" }),
    card("Mini P.E.K.K.A", 4, "troop", "tank_killer", { hp: 1361, damage: 720, hit_speed: 1.6, range: 0.8, speed: 1.5 }),
    card("Goblins", 2, "troop", "swarm", { count: 4, hp: 202, damage: 120, hit_speed: 1.1, range: 0.5, speed: 2.0 }),
    card("Fireball", 4, "spell", "spell", { damage: 689, radius: 2.5, tower_damage: 207, delay: 1.0 }),
    card("Arrows", 3, "spell", "spell", { damage: 366, radius: 4.0, tower_damage: 93, delay: 0.8 }),
  ];
  const PRINCESS_HP = 3052, PRINCESS_DAMAGE = 109, PRINCESS_HIT_SPEED = 0.8, PRINCESS_RANGE = 7.5;
  const KING_HP = 4824, KING_DAMAGE = 109, KING_HIT_SPEED = 1.0, KING_RANGE = 7.0;
  const TOWER_SIZE = 1.5, WIDTH = 18, HEIGHT = 32, RIVER_LO = 15, RIVER_HI = 17, RIVER_Y = 16;
  const LANE_X = [3.5, 14.5];
  const DEPLOY_TIME = 1.0, MATCH_TIME = 180, DOUBLE_ELIXIR_AT = 120, ELIXIR_RATE = 1 / 2.8;
  const START_ELIXIR = 5, MAX_ELIXIR = 10, DT = 0.1;

  // Seeded PRNG (mulberry32) and helpers
  function rng(seed) {
    let a = seed >>> 0;
    const next = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    return {
      random: next,
      int: (n) => Math.floor(next() * n),
      perm(n) { const p = [...Array(n).keys()]; for (let i = n - 1; i > 0; i--) { const j = Math.floor(next() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; },
      normal() {
        if (spare !== null) { const s = spare; spare = null; return s; }
        let u = 0, v = 0; while (u === 0) u = next(); v = next();
        const r = Math.sqrt(-2 * Math.log(u)); spare = r * Math.sin(2 * Math.PI * v); return r * Math.cos(2 * Math.PI * v);
      },
    };
  }
  const dist = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);
  const frameY = (player, y) => (player === 0 ? y : HEIGHT - y);
  const absY = frameY;
  const laneOf = (x) => (x < WIDTH / 2 ? 0 : 1);
  const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  // --------------------------------------------------------------- sim.py
  class Sim {
    constructor(seed = 1, shuffleUpdates = true) { this.rng = rng(seed); this.shuffleUpdates = shuffleUpdates; this.reset(); }
    order(n) { return this.shuffleUpdates ? this.rng.perm(n) : [...Array(n).keys()]; }
    reset() {
      this.time = 0; this.units = []; this.spells = []; this.events = []; this.uid = 0;
      this.players = [0, 1].map(() => {
        const order = this.rng.perm(DECK.length);
        return { elixir: START_ELIXIR, hand: order.slice(0, 4), queue: order.slice(4), crowns: 0, leaked: 0, spent: 0 };
      });
      this.towers = [];
      for (const owner of [0, 1]) {
        LANE_X.forEach((x, lane) => this.towers.push(
          { owner, kind: "princess", lane, x, y: absY(owner, 6.5), hp: PRINCESS_HP, max_hp: PRINCESS_HP, active: true, cooldown: 0, target: null, building: true }));
        this.towers.push({ owner, kind: "king", lane: null, x: WIDTH / 2, y: absY(owner, 3), hp: KING_HP, max_hp: KING_HP, active: false, cooldown: 0, target: null, building: true });
      }
      this.done = false; this.winner = null;
    }
    tower(owner, kind, lane = null) { return this.towers.find((t) => t.owner === owner && t.kind === kind && (kind === "king" || t.lane === lane)); }
    laneTarget(owner, lane) { const p = this.tower(1 - owner, "princess", lane); return p.hp > 0 ? p : this.tower(1 - owner, "king"); }
    get doubleElixir() { return this.time >= DOUBLE_ELIXIR_AT; }
    canPlay(player, c) { const p = this.players[player]; return !this.done && p.hand.includes(c) && p.elixir >= DECK[c].cost; }
    validPosition(player, c, x, y) {
      if (!(x >= 0.5 && x <= WIDTH - 0.5 && y >= 0.5 && y <= HEIGHT - 0.5)) return false;
      if (DECK[c].kind === "spell") return true;
      const fy = frameY(player, y);
      if (fy <= RIVER_LO - 0.5) return true;
      return fy <= 20 && !(this.tower(1 - player, "princess", laneOf(x)).hp > 0);
    }
    play(player, c, x, y) {
      if (!this.canPlay(player, c) || !this.validPosition(player, c, x, y)) return false;
      const cd = DECK[c], p = this.players[player];
      p.elixir -= cd.cost; p.spent += cd.cost;
      p.hand[p.hand.indexOf(c)] = p.queue.shift(); p.queue.push(c);
      this.events.push({ t: +this.time.toFixed(2), kind: "play", player, card: c, x, y });
      if (cd.kind === "spell") { this.spells.push({ owner: player, card: cd, idx: c, x, y, delay: cd.delay }); return true; }
      const offsets = { 1: [[0, 0]], 2: [[-0.6, 0], [0.6, 0]], 4: [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]] }[cd.count];
      for (const [dx, dy] of offsets) {
        this.units.push({ uid: ++this.uid, owner: player, card: cd, idx: c, x: clip(x + dx, 0.5, WIDTH - 0.5), y: y + dy,
          hp: cd.hp, deploy: DEPLOY_TIME, cooldown: 0, target: null, retarget: 0, building: false });
      }
      return true;
    }
    step(dt = DT) {
      if (this.done) return;
      this.time += dt;
      const rate = ELIXIR_RATE * (this.doubleElixir ? 2 : 1);
      for (const p of this.players) {
        const gain = rate * dt, room = MAX_ELIXIR - p.elixir;
        p.leaked += Math.max(0, gain - room); p.elixir = Math.min(MAX_ELIXIR, p.elixir + gain);
      }
      for (const s of this.spells) { s.delay -= dt; if (s.delay <= 0) this.resolveSpell(s); }
      this.spells = this.spells.filter((s) => s.delay > 0);
      for (const i of this.order(this.units.length)) { const u = this.units[i]; if (u.hp > 0) this.updateUnit(u, dt); }
      for (const i of this.order(this.towers.length)) { const t = this.towers[i]; if (t.hp > 0 && t.active) this.updateTower(t, dt); }
      this.units = this.units.filter((u) => u.hp > 0);
      if (this.time >= MATCH_TIME - 1e-9 && !this.done) this.finish();
    }
    resolveSpell(s) {
      for (const u of this.units) if (u.owner !== s.owner && u.hp > 0 && dist(u.x, u.y, s.x, s.y) <= s.card.radius) u.hp -= s.card.damage;
      for (const t of this.towers) if (t.owner !== s.owner && t.hp > 0 && dist(t.x, t.y, s.x, s.y) <= s.card.radius + TOWER_SIZE) this.damageTower(t, s.card.tower_damage, s.owner);
    }
    canTarget(u, o) { if (!(o.hp > 0) || o.owner === u.owner) return false; return u.card.targets === "buildings" ? o.building : true; }
    gap(u, o) { return dist(u.x, u.y, o.x, o.y) - (o.building ? TOWER_SIZE : 0); }
    updateUnit(u, dt) {
      if (u.deploy > 0) { u.deploy -= dt; return; }
      u.retarget -= dt;
      let t = u.target;
      if (t && (!(t.hp > 0) || (!t.building && this.gap(u, t) > u.card.sight + 1))) t = null;
      if (t === null || (u.retarget <= 0 && !(t && this.gap(u, t) <= u.card.range))) {
        u.retarget = 0.5;
        let best = null, bestD = u.card.sight;
        for (const o of this.units) if (this.canTarget(u, o)) { const d = this.gap(u, o); if (d <= bestD) { best = o; bestD = d; } }
        for (const o of this.towers) if (this.canTarget(u, o)) { const d = this.gap(u, o); if (d <= bestD) { best = o; bestD = d; } }
        t = best || this.laneTarget(u.owner, laneOf(u.x));
      }
      u.target = t;
      u.cooldown -= dt;
      if (this.gap(u, t) <= u.card.range) {
        if (u.cooldown <= 0) {
          u.cooldown = u.card.hit_speed;
          if (t.building) this.damageTower(t, u.card.damage, u.owner); else t.hp -= u.card.damage;
        }
        return;
      }
      u.cooldown = Math.max(u.cooldown, u.card.hit_speed * 0.5);
      let wx = t.x, wy = t.y;
      const sideU = u.y < RIVER_LO ? 0 : (u.y > RIVER_HI ? 1 : null);
      const sideT = t.y < RIVER_LO ? 0 : 1;
      if (sideU !== null && sideU !== sideT) {
        wx = Math.abs(LANE_X[0] - u.x) <= Math.abs(LANE_X[1] - u.x) ? LANE_X[0] : LANE_X[1]; wy = RIVER_Y;
      }
      const dx = wx - u.x, dy = wy - u.y, d = Math.hypot(dx, dy);
      if (d > 1e-6) { const s = Math.min(d, u.card.speed * dt); u.x += dx / d * s; u.y += dy / d * s; }
    }
    updateTower(t, dt) {
      const range = t.kind === "princess" ? PRINCESS_RANGE : KING_RANGE;
      if (t.target && (!(t.target.hp > 0) || dist(t.x, t.y, t.target.x, t.target.y) > range)) t.target = null;
      if (!t.target) {
        let best = null, bestD = range;
        for (const u of this.units) if (u.owner !== t.owner && u.hp > 0 && u.deploy <= 0) { const d = dist(t.x, t.y, u.x, u.y); if (d <= bestD) { best = u; bestD = d; } }
        t.target = best;
      }
      t.cooldown -= dt;
      if (t.target && t.cooldown <= 0) {
        t.cooldown = t.kind === "princess" ? PRINCESS_HIT_SPEED : KING_HIT_SPEED;
        t.target.hp -= t.kind === "princess" ? PRINCESS_DAMAGE : KING_DAMAGE;
      }
    }
    damageTower(t, amount, attacker) {
      if (!(t.hp > 0)) return;
      t.hp -= amount;
      if (t.kind === "king") t.active = true;
      if (t.hp <= 0) {
        t.hp = 0;
        this.events.push({ t: +this.time.toFixed(2), kind: "tower", player: attacker, tower: t.kind, lane: t.lane });
        if (t.kind === "king") { this.players[attacker].crowns = 3; this.finish(); }
        else { this.players[attacker].crowns += 1; this.tower(t.owner, "king").active = true; }
      }
    }
    finish() {
      this.done = true;
      const [c0, c1] = this.players.map((p) => p.crowns);
      this.winner = c0 > c1 ? 0 : c1 > c0 ? 1 : null;
    }
  }

  // ----------------------------------------------------------- strategy.py
  const THREAT_LINE = 20;
  class View {
    constructor(sim, me) { this.sim = sim; this.me = me; this.foe = 1 - me; this.p = sim.players[me]; }
    fy(y) { return frameY(this.me, y); }
    get elixir() { return this.p.elixir; }
    threats(lane) { return this.sim.units.filter((u) => u.owner === this.foe && laneOf(u.x) === lane && this.fy(u.y) < THREAT_LINE); }
    defenders(lane) { return this.sim.units.filter((u) => u.owner === this.me && laneOf(u.x) === lane && this.fy(u.y) < RIVER_LO + 1); }
    pushers(lane) { return this.sim.units.filter((u) => u.owner === this.me && laneOf(u.x) === lane && this.fy(u.y) >= 9); }
    enemyBackTank() { for (const u of this.sim.units) if (u.owner === this.foe && u.card.role === "tank" && this.fy(u.y) > 24) return laneOf(u.x); return null; }
    towerFrac(mine, lane) { const t = this.sim.tower(mine ? this.me : this.foe, "princess", lane); return t.hp / t.max_hp; }
  }
  function spellValue(view, cd, x, y) {
    let v = 0;
    for (const u of view.sim.units) if (u.owner === view.foe && dist(u.x, u.y, x, y) <= cd.radius) v += Math.min(u.hp, cd.damage) / u.card.hp * u.card.cost / u.card.count;
    return v;
  }
  function bestSpellSpot(view, cd, lane = null) {
    let best = [0, 0, 0];
    for (const u of view.sim.units) if (u.owner === view.foe && (lane === null || laneOf(u.x) === lane)) {
      const v = spellValue(view, cd, u.x, u.y); if (v > best[0]) best = [v, u.x, u.y];
    }
    return best;
  }
  function place(view, c, lane) {
    const cd = DECK[c], me = view.me, lx = LANE_X[lane], tc = lane === 0 ? 1 : -1;
    if (cd.kind === "spell") {
      const [v, x, y] = bestSpellSpot(view, cd, lane);
      if (v > 0) return [x, y];
      const t = view.sim.laneTarget(me, lane); return [t.x, t.y];
    }
    const threats = view.threats(lane);
    if (threats.length) {
      const lead = threats.reduce((a, b) => (view.fy(b.y) < view.fy(a.y) ? b : a));
      const leadFy = view.fy(lead.y);
      if (cd.role === "ranged") return [lx + 1.5 * tc, absY(me, 4.5)];
      if (leadFy > RIVER_LO) return [lx + tc, absY(me, 11)];
      if (cd.role === "swarm") return [lead.x, absY(me, Math.max(0.5, leadFy - 1))];
      return [clip(lead.x + tc, 0.5, WIDTH - 0.5), absY(me, Math.max(0.5, leadFy - 2.5))];
    }
    const pushers = view.pushers(lane);
    if (cd.role === "tank") {
      if (view.elixir >= 8 || view.sim.doubleElixir) return [lx, absY(me, 14)];
      return [WIDTH / 2 - tc * 1.5, absY(me, 1)];
    }
    if (pushers.length) {
      const lead = pushers.reduce((a, b) => (view.fy(b.y) > view.fy(a.y) ? b : a));
      return [clip(lead.x, 0.5, WIDTH - 0.5), absY(me, Math.min(14, view.fy(lead.y) - 2))];
    }
    if (cd.role === "ranged") return [lx, absY(me, 10)];
    return [lx, absY(me, 14)];
  }

  // ---------------------------------------------------------------- env.py
  const searchRight = (edges, v) => { let i = 0; while (i < edges.length && edges[i] <= v) i++; return i; };
  function channelNames() {
    const n = [[0, 2], [2, 3], [3, 4], [4, 5], [5, 7], [7, 9], [9, 10]].map(([a, b]) => `elixir ${a}-${b}`);
    for (const c of DECK) n.push(`${c.name} ready`);
    for (const c of DECK) n.push(`${c.name} in hand`);
    for (const s of ["left", "right"]) {
      n.push(`${s}: no threat`, `${s}: small threat`, `${s}: medium threat`, `${s}: big threat`);
      n.push(`${s}: enemy tank`, `${s}: enemy melee`, `${s}: enemy ranged`, `${s}: enemy swarm`);
      n.push(`${s}: threat at my tower`, `${s}: I'm defending`);
      n.push(`${s}: no push`, `${s}: small push`, `${s}: strong push`);
      n.push(`${s}: my tower low`, `${s}: my tower down`, `${s}: enemy tower low`, `${s}: enemy tower down`);
      n.push(`${s}: enemy tank building at back`);
    }
    n.push("double elixir", "fireball has value", "arrows has value");
    return n;
  }
  const CHANNELS = channelNames();
  function features(view) {
    const f = new Float32Array(CHANNELS.length);
    let i = 0;
    f[i + searchRight([2, 3, 4, 5, 7, 9], view.elixir)] = 1; i += 7;
    for (const c of view.p.hand) { f[i + c] = DECK[c].cost <= view.elixir ? 1 : 0; f[i + DECK.length + c] = 1; }
    i += 2 * DECK.length;
    for (const lane of [0, 1]) {
      const th = view.threats(lane);
      const hp = th.reduce((s, u) => s + u.hp, 0);
      f[i + searchRight([1, 700, 2000], hp)] = 1;
      const roles = new Set(th.map((u) => u.card.role));
      f[i + 4] = +roles.has("tank"); f[i + 5] = +(roles.has("tank_killer") || roles.has("mini_tank"));
      f[i + 6] = +roles.has("ranged"); f[i + 7] = +roles.has("swarm");
      f[i + 8] = +th.some((u) => view.fy(u.y) < 10);
      f[i + 9] = +(view.defenders(lane).length > 0);
      const push = view.pushers(lane).reduce((s, u) => s + u.hp, 0);
      f[i + 10 + searchRight([1, 1000], push)] = 1;
      const mine = view.towerFrac(true, lane), theirs = view.towerFrac(false, lane);
      f[i + 13] = +(mine > 0 && mine < 0.5); f[i + 14] = +(mine === 0);
      f[i + 15] = +(theirs > 0 && theirs < 0.5); f[i + 16] = +(theirs === 0);
      f[i + 17] = +(view.enemyBackTank() === lane);
      i += 18;
    }
    f[i] = +view.sim.doubleElixir;
    f[i + 1] = +(bestSpellSpot(view, DECK[6])[0] >= DECK[6].cost);
    f[i + 2] = +(bestSpellSpot(view, DECK[7])[0] >= DECK[7].cost);
    return f;
  }
  function cardMask(view) {
    const m = new Array(DECK.length + 1).fill(false); m[0] = true;
    for (const c of view.p.hand) m[c + 1] = DECK[c].cost <= view.elixir;
    return m;
  }

  // ------------------------------------------------- brain (network.py / agent.py)
  const b64 = (s, T) => { const bin = typeof atob === "function" ? atob(s) : Buffer.from(s, "base64").toString("binary");
    const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new T(u8.buffer); };
  class Brain {
    // data: exported by scripts/export_brain_js.py
    constructor(d, seed = 7) {
      this.d = d; this.rng = rng(seed);
      this.n = d.n_pn + d.n_kc + d.n_mbon;
      this.kc0 = d.n_pn; this.mb0 = d.n_pn + d.n_kc;
      this.syn_ptr = b64(d.syn_ptr, Int32Array);      // by presynaptic neuron (CSC)
      this.syn_post = b64(d.syn_post, Uint16Array);
      this.syn_w = b64(d.syn_w, Float32Array);
      this.chan_ptr = d.chan_ptr; this.chan_pn = d.chan_pn;
      this.card_group = d.card_group; this.lane_group = d.lane_group;   // per MBON, -1 if none
      this.card_n = d.card_n; this.lane_n = d.lane_n;                   // MBONs per group
      this.v = new Float32Array(this.n); this.isyn = new Float32Array(this.n);
      this.iext = new Float32Array(this.n); this.refr = new Int16Array(this.n);
      this.spk = new Uint8Array(this.n); this.counts = new Float32Array(this.n);
    }
    decide(feat, mask, epsilon = 0) {
      const d = this.d, n = this.n;
      this.v.fill(0); this.isyn.fill(0); this.refr.fill(0); this.spk.fill(0); this.counts.fill(0); this.iext.fill(0);
      for (let c = 0; c < feat.length; c++) if (feat[c]) for (let k = this.chan_ptr[c]; k < this.chan_ptr[c + 1]; k++) this.iext[this.chan_pn[k]] = d.pn_current;
      const rec = new Float32Array(n);
      for (let step = 0; step < d.decision_steps; step++) {
        rec.fill(0);
        let kcSpikes = 0;
        for (let i = 0; i < n; i++) if (this.spk[i]) {
          if (i >= this.kc0 && i < this.mb0) kcSpikes++;
          for (let k = this.syn_ptr[i]; k < this.syn_ptr[i + 1]; k++) rec[this.syn_post[k]] += this.syn_w[k];
        }
        const apl = d.apl_gain * kcSpikes / d.n_kc;
        for (let i = 0; i < n; i++) {
          let s = d.beta * this.isyn[i] + rec[i];
          if (i >= this.kc0 && i < this.mb0) s -= apl;
          this.isyn[i] = s;
          let itot = s + this.iext[i] + d.noise * this.rng.normal();
          if (i >= this.mb0) itot += d.mbon_noise * this.rng.normal();
          if (this.refr[i] <= 0) this.v[i] = d.alpha * this.v[i] + itot;
          const fire = this.v[i] >= 1;
          if (fire) { this.v[i] = 0; this.refr[i] = d.refractory; this.counts[i]++; } else this.refr[i]--;
          this.spk[i] = fire ? 1 : 0;
        }
      }
      const cardVotes = new Array(d.card_n.length).fill(0), laneVotes = new Array(d.lane_n.length).fill(0);
      for (let m = 0; m < d.n_mbon; m++) {
        const c = this.counts[this.mb0 + m];
        if (this.card_group[m] >= 0) cardVotes[this.card_group[m]] += c;
        if (this.lane_group[m] >= 0) laneVotes[this.lane_group[m]] += c;
      }
      for (let g = 0; g < cardVotes.length; g++) cardVotes[g] /= d.card_n[g];
      for (let g = 0; g < laneVotes.length; g++) laneVotes[g] /= d.lane_n[g];
      const pick = (votes, ok) => {
        const allowed = votes.map((_, i) => i).filter((i) => ok[i]);
        if (this.rng.random() < epsilon) return allowed[this.rng.int(allowed.length)];
        let best = allowed[0], bv = -Infinity;
        for (const i of allowed) { const v = votes[i] + 1e-3 * this.rng.random(); if (v > bv) { bv = v; best = i; } }
        return best;
      };
      let kcActive = 0;
      for (let i = this.kc0; i < this.mb0; i++) if (this.counts[i] > 0) kcActive++;
      return { card: pick(cardVotes, mask), lane: pick(laneVotes, [true, true]), cardVotes, laneVotes, kcActive: kcActive / d.n_kc };
    }
  }

  return { DECK, LANE_X, WIDTH, HEIGHT, RIVER_LO, RIVER_HI, MATCH_TIME, DOUBLE_ELIXIR_AT, MAX_ELIXIR, DT, CHANNELS,
    Sim, View, place, features, cardMask, Brain, frameY, absY, laneOf, rng };
})();

if (typeof module !== "undefined") module.exports = ClashCore;
