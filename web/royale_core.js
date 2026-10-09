// JavaScript port of the full simulator (flybrain/envs/royale/sim.py), the
// strategy helpers the fly needs (strategy.py: View, place, Coach.suggest,
// auto_ability) and its inputs (env.py: features, role_mask, pick_card), so
// the full-card-pool fly can play in a browser. Keep in sync with the Python;
// tests/test_web_royale.py checks they agree. Needs ClashCore.Brain and
// ClashCore.rng from clash_core.js.
"use strict";

const RoyaleCore = (() => {
  const WIDTH = 18, HEIGHT = 32, RIVER_LO = 15, RIVER_HI = 17, RIVER_Y = 16, LANE_X = [3.5, 14.5], DT = 0.1;
  const REGULATION = 180, OVERTIME_END = 300, DOUBLE_AT = 120, TRIPLE_AT = 240, ELIXIR_RATE = 1 / 2.8;
  const START_ELIXIR = 5, MAX_ELIXIR = 10, TUNNEL_SPEED = 10, CHARGE_AFTER = 2.5;
  const TOWER = { princess: { hp: 3052, damage: 109, hit_speed: 0.8, range: 7.5, radius: 1.5 },
                  king: { hp: 4824, damage: 109, hit_speed: 1.0, range: 7.0, radius: 2.0 } };
  const ROLES = ["win_condition", "tank_killer", "frontline", "splash_ground", "splash_air", "ranged", "swarm", "air",
                 "cycle", "building", "spawner", "small_spell", "big_spell", "utility", "champion"];
  const frameY = (p, y) => (p === 0 ? y : HEIGHT - y), absY = frameY;
  const laneOf = (x) => (x < WIDTH / 2 ? 0 : 1);
  const sideOf = (y) => (y < RIVER_LO ? 0 : y > RIVER_HI ? 1 : null);
  const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const dist = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);
  const minBy = (arr, f) => { let b = null, bv = Infinity; for (const a of arr) { const v = f(a); if (v < bv) { bv = v; b = a; } } return b; };
  const maxBy = (arr, f) => { let b = null, bv = -Infinity; for (const a of arr) { const v = f(a); if (v > bv) { bv = v; b = a; } } return b; };
  const searchRight = (edges, v) => { let i = 0; while (i < edges.length && edges[i] <= v) i++; return i; };

  // ------------------------------------------------------------------ db.py
  const ROLE_OF = {};
  const roleList = {
    win_condition: ["Giant", "Hog Rider", "Balloon", "Golem", "Royal Giant", "Lava Hound", "Miner", "Battle Ram", "Ram Rider",
      "Goblin Barrel", "Graveyard", "Wall Breakers", "Royal Hogs", "Goblin Giant", "Electro Giant", "Elixir Golem", "X-Bow",
      "Mortar", "Goblin Drill", "Skeleton Barrel", "Three Musketeers"],
    tank_killer: ["P.E.K.K.A", "Mini P.E.K.K.A", "Inferno Dragon", "Prince", "Lumberjack", "Hunter", "Elite Barbarians",
      "Sparky", "Night Witch", "Raging Prince"],
    frontline: ["Knight", "Ice Golem", "Giant Skeleton", "Bandit", "Fisherman", "Battle Healer", "Rascals", "Cannon Cart"],
    splash_ground: ["Valkyrie", "Bowler", "Bomber", "Dark Prince", "Mega Knight", "Royal Ghost"],
    splash_air: ["Wizard", "Baby Dragon", "Executioner", "Witch", "Electro Dragon", "Firecracker", "Ice Wizard", "Princess",
      "Magic Archer", "Skeleton Dragons", "Electro Wizard", "Zappies", "Mother Witch"],
    ranged: ["Musketeer", "Archers", "Dart Goblin", "Spear Goblins", "Flying Machine"],
    swarm: ["Goblins", "Goblin Gang", "Skeleton Army", "Barbarians", "Guards", "Royal Recruits", "Minion Horde"],
    air: ["Minions", "Mega Minion", "Bats", "Phoenix"],
    cycle: ["Skeletons", "Ice Spirit", "Fire Spirit", "Electro Spirit", "Heal Spirit"],
    building: ["Cannon", "Tesla", "Inferno Tower", "Bomb Tower", "Goblin Cage", "Tombstone"],
    spawner: ["Goblin Hut", "Furnace", "Barbarian Hut", "Elixir Collector", "Party Hut"],
    small_spell: ["Zap", "The Log", "Arrows", "Giant Snowball", "Barbarian Barrel", "Royal Delivery", "Tornado"],
    big_spell: ["Fireball", "Poison", "Lightning", "Rocket", "Earthquake"],
    utility: ["Freeze", "Rage", "Clone", "Mirror"],
    champion: ["Archer Queen", "Golden Knight", "Skeleton King", "Mighty Miner", "Monk"],
  };
  for (const [r, ns] of Object.entries(roleList)) for (const n of ns) ROLE_OF[n] = r;

  function buildDB(raw) {
    const db = { level: raw.level, buffs: {}, projectiles: {}, effects: {}, characters: {}, abilities: {}, cards: {} };
    for (const [k, v] of Object.entries(raw.buffs)) db.buffs[k] = Object.assign({ name: k }, v);
    for (const [k, v] of Object.entries(raw.projectiles)) db.projectiles[k] = Object.assign({ name: k }, v);
    for (const [k, v] of Object.entries(raw.effects)) db.effects[k] = Object.assign({ name: k }, v);
    for (const [k, v] of Object.entries(raw.characters)) db.characters[k] = Object.assign({ name: k }, v);
    for (const [k, v] of Object.entries(raw.abilities)) db.abilities[k] = Object.assign({ name: k }, v);
    const g = (m, k) => (k == null ? null : m[k] || null);
    for (const p of Object.values(db.projectiles)) { p.buff = g(db.buffs, p.buff); p.spawn_character = g(db.characters, p.spawn_character); p.spawn_projectile = g(db.projectiles, p.spawn_projectile); }
    for (const e of Object.values(db.effects)) { e.buff = g(db.buffs, e.buff); e.projectile = g(db.projectiles, e.projectile); e.spawn_character = g(db.characters, e.spawn_character); }
    for (const b of Object.values(db.buffs)) b.death_spawn = g(db.characters, b.death_spawn);
    for (const c of Object.values(db.characters)) {
      for (const k of ["spawn_character", "death_spawn", "death_spawn2"]) c[k] = g(db.characters, c[k]);
      for (const k of ["projectile", "death_projectile", "special_projectile"]) c[k] = g(db.projectiles, c[k]);
      c.spawn_effect = g(db.effects, c.spawn_effect); c.death_effect = g(db.effects, c.death_effect);
      c.buff_on_damage = g(db.buffs, c.buff_on_damage); c.ability = g(db.abilities, c.ability);
      c.vd = c.variable_damage || null;
    }
    for (const d of raw.cards) {
      const c = Object.assign({}, d);
      c.summons = d.summons.map(([n, k]) => [db.characters[n], k]);
      c.effect = g(db.effects, d.effect); c.projectile = g(db.projectiles, d.projectile);
      c.role = ROLE_OF[c.name] || null; c.champion = c.rarity === "Champion"; c.full_lane = !!d.full_lane;
      db.cards[c.name] = c;
    }
    db.pool = () => Object.values(db.cards).filter((c) => !c.event && c.role).map((c) => c.name);
    return db;
  }

  // ----------------------------------------------------------------- sim.py
  class Unit {
    constructor(sim, spec, owner, x, y, card = null, deploy = null, tower = null, lane = null) {
      sim._uid += 1;
      this.uid = sim._uid; this.sim = sim; this.spec = spec; this.owner = owner; this.x = x; this.y = y;
      this.card = card; this.tower = tower; this.lane = lane;
      if (tower) {
        const t = TOWER[tower];
        this.hp = this.max_hp = t.hp; this.radius = t.radius; this.building = true; this.flying = false;
        this.deploy_left = 0; this.active = tower === "princess"; this.timed = false;
      } else {
        this.hp = this.max_hp = spec.hp; this.radius = spec.radius; this.building = spec.building; this.flying = spec.flying;
        this.deploy_left = deploy == null ? spec.deploy_time : deploy; this.active = true;
        this.timed = spec.building && spec.hp <= 0;
      }
      this.shield = tower ? 0 : spec.shield;
      this.alive = true;
      this.life_left = spec && spec.life_time ? spec.life_time : null;
      this.cooldown = 0; this.target = null; this.retarget = 0; this.buffs = new Map();
      this.walked = 0; this.charging = false; this.dash_cd = 0; this.dashing = 0; this.lock_time = 0; this.combo = 0;
      this.idle = 0; this.parent = null; this.children = []; this.ability_cd = 0; this.ability_left = 0; this.souls = 0;
      this.hook_cd = 0; this.elixir_timer = 0;
      if (spec && spec.spawn_character && !spec.spawn_attach) { this.spawn_timer = spec.spawn_start || spec.spawn_pause || 1.0; this.spawned = 0; }
      else { this.spawn_timer = null; this.spawned = 0; }
    }
    factor(kind) {
      let f = 1;
      for (const [b] of this.buffs.values()) f *= b[kind];
      if (kind === "hit_speed" && this.ability_left > 0 && this.spec.ability && this.spec.ability.hit_speed) f *= this.spec.ability.hit_speed;
      return f;
    }
    get invisible() {
      for (const [b] of this.buffs.values()) if (b.invisible) return true;
      const s = this.spec;
      if (!s) return false;
      if (s.invisible_when_idle && this.idle >= 1.8) return true;
      if (s.ability && s.ability.invisible && this.ability_left > 0) return true;
      if (s.hides_when_idle && this.idle >= 0.8) return true;
      return false;
    }
  }

  class Projectile {
    constructor(sim, spec, owner, x, y, o = {}) {
      this.sim = sim; this.spec = spec; this.owner = owner; this.x = x; this.y = y;
      this.target = o.target || null;
      this.tx = o.tx != null ? o.tx : (this.target ? this.target.x : x);
      this.ty = o.ty != null ? o.ty : (this.target ? this.target.y : y);
      this.damage = o.damage != null ? o.damage : spec.damage;
      this.source = o.source || null; this.pierce = !!o.pierce; this.stop_on_hit = !!o.stop_on_hit;
      this.hit = new Set(); this.travelled = 0; this.returning = false; this.chain_left = o.chain_left || 0;
      this.tower_factor = o.tower_factor != null ? o.tower_factor : spec.tower_factor;
      let dir = o.direction;
      if (!dir) { const dx = this.tx - x, dy = this.ty - y; const d = Math.hypot(dx, dy) || 1; dir = [dx / d, dy / d]; }
      this.dir = dir; this.alive = true;
    }
  }

  class Effect {
    constructor(sim, spec, owner, x, y, delay = 0) {
      this.sim = sim; this.spec = spec; this.owner = owner; this.x = x; this.y = y; this.delay = delay;
      this.time_left = Math.max(spec.duration, DT); this.next_hit = 0; this.started = false; this.struck = new Set();
      this.spawn_next = spec.spawn_character ? spec.spawn_initial_delay : null; this.alive = true;
    }
  }

  class Sim {
    constructor(db, decks, seed = 1, shuffleUpdates = true, rngFn = null) {
      this.db = db;
      for (const d of decks) {
        if (d.length !== 8 || new Set(d).size !== 8) throw new Error("a deck needs 8 different cards: " + d);
        for (const n of d) if (!db.cards[n]) throw new Error("unknown card " + n);
      }
      this.rng = rngFn || ClashCore.rng(seed);
      this.shuffleUpdates = shuffleUpdates; this.decks = [decks[0].slice(), decks[1].slice()];
      this.reset();
    }
    reset() {
      this.time = 0; this._uid = 0; this.units = []; this.projectiles = []; this.effects = []; this.events = [];
      this._dead_towers = [];
      this.players = this.decks.map((deck) => {
        const order = this.rng.perm(deck.length);
        return { deck: deck.slice(), hand: order.slice(0, 4).map((i) => deck[i]), queue: order.slice(4).map((i) => deck[i]),
                 elixir: START_ELIXIR, crowns: 0, leaked: 0, spent: 0, last_card: null };
      });
      for (const owner of [0, 1]) {
        LANE_X.forEach((x, lane) => this.units.push(new Unit(this, null, owner, x, absY(owner, 6.5), null, null, "princess", lane)));
        const king = new Unit(this, null, owner, WIDTH / 2, absY(owner, 3.0), null, null, "king");
        king.active = false;
        this.units.push(king);
      }
      this.done = false; this.winner = null; this.overtime = false;
    }
    tower(owner, kind, lane = null) {
      for (const u of this.units.concat(this._dead_towers)) if (u.tower === kind && u.owner === owner && (kind === "king" || u.lane === lane)) return u;
      throw new Error("no tower");
    }
    towers(owner = null) { return this.units.filter((u) => u.tower).concat(this._dead_towers).filter((u) => owner == null || u.owner === owner); }
    laneTarget(owner, lane) { const p = this.tower(1 - owner, "princess", lane); return p.alive ? p : this.tower(1 - owner, "king"); }
    get elixirMultiplier() { return this.time >= TRIPLE_AT ? 3 : this.time >= DOUBLE_AT ? 2 : 1; }
    cardCost(player, name) {
      const c = this.db.cards[name];
      if (c.special === "mirror") { const last = this.players[player].last_card; return last ? this.db.cards[last].elixir + 1 : 99; }
      return c.elixir;
    }
    championAlive(player) { return this.units.some((u) => u.alive && u.owner === player && u.card && this.db.cards[u.card].champion && !u.parent); }
    canPlay(player, name) {
      const p = this.players[player];
      if (this.done || !p.hand.includes(name) || p.elixir < this.cardCost(player, name)) return false;
      let c = this.db.cards[name];
      if (c.special === "mirror") { const last = p.last_card; if (!last || this.db.cards[last].champion) return false; c = this.db.cards[last]; }
      if (c.champion && this.championAlive(player)) return false;
      return true;
    }
    validPosition(player, name, x, y) {
      if (!(x >= 0.5 && x <= WIDTH - 0.5 && y >= 0.5 && y <= HEIGHT - 0.5)) return false;
      let c = this.db.cards[name];
      if (c.special === "mirror") c = this.players[player].last_card ? this.db.cards[this.players[player].last_card] : c;
      if (c.type === "spell" && !c.summons.length) return true;
      if (c.type === "spell" || c.deploy_anywhere) return sideOf(y) !== null;
      const fy = frameY(player, y);
      if (fy <= RIVER_LO - 0.5) return true;
      const ep = this.tower(1 - player, "princess", laneOf(x));
      return fy <= 20 && !ep.alive && c.type !== "building";
    }
    play(player, name, x, y) {
      if (!this.canPlay(player, name) || !this.validPosition(player, name, x, y)) return false;
      const p = this.players[player];
      const cost = this.cardCost(player, name);
      p.elixir -= cost; p.spent += cost;
      p.hand[p.hand.indexOf(name)] = p.queue.shift(); p.queue.push(name);
      let card = this.db.cards[name];
      if (card.special === "mirror") card = this.db.cards[p.last_card]; else p.last_card = name;
      this.events.push({ t: +this.time.toFixed(2), kind: "play", player, card: card.name, x, y });
      this._deploy(player, card, x, y);
      return true;
    }
    _deploy(player, card, x, y) {
      const king = this.tower(player, "king");
      if (card.projectile && card.type === "spell") {
        const spec = card.projectile;
        if (card.special === "rolling") {
          const d = player === 0 ? 1 : -1;
          this.projectiles.push(new Projectile(this, spec, player, x, y, { tx: x, ty: y + d, pierce: true, direction: [0, d] }));
        } else {
          for (let w = 0; w < card.waves; w++) {
            const pr = new Projectile(this, spec, player, king.x, king.y, { tx: x, ty: y });
            pr.travelled = -w * 0.3 * spec.speed;
            this.projectiles.push(pr);
          }
        }
      }
      if (card.effect) {
        let delay = 0;
        if (card.type === "spell" && !card.summons.length) delay = card.effect.duration <= 0.01 ? 0.3 : 0;
        this.effects.push(new Effect(this, card.effect, player, x, y, delay));
      }
      if (!card.summons.length) return;
      let tunnel = 0;
      if (card.deploy_anywhere && card.type !== "spell") tunnel = dist(king.x, king.y, x, y) / TUNNEL_SPEED;
      const positions = this._formation(card, player, x, y);
      let k = 0;
      for (const [spec, count] of card.summons) {
        for (let i = 0; i < count; i++) {
          const [px, py] = positions[k % positions.length];
          k += 1;
          const u = new Unit(this, spec, player, px, py, card.name, spec.deploy_time + tunnel + 0.1 * (k - 1) * (card.radius > 0 ? 1 : 0));
          this._add(u);
          if (spec.spawn_attach && spec.spawn_character) {
            for (let j = 0; j < Math.max(1, spec.spawn_number); j++) {
              const child = new Unit(this, spec.spawn_character, player, px, py, card.name, u.deploy_left);
              child.parent = u; u.children.push(child); this._add(child);
            }
          }
        }
      }
      if (card.projectile && card.type !== "spell") {
        const pr = new Projectile(this, card.projectile, player, x, y, { tx: x, ty: y });
        pr.travelled = -1.0 * card.projectile.speed;
        this.projectiles.push(pr);
      }
    }
    _formation(card, player, x, y) {
      const n = card.summons.reduce((s, [, c]) => s + c, 0);
      if (n <= 1) return [[x, y]];
      if (card.full_lane) { const out = []; for (let i = 0; i < n; i++) out.push([2.5 + (WIDTH - 5) * i / (n - 1), y]); return out; }
      if (card.name === "Royal Hogs") return [-1.8, -0.6, 0.6, 1.8].map((dx) => [clip(x + dx, 0.5, WIDTH - 0.5), y]);
      const r = card.radius || 0.7, out = [];
      for (let i = 0; i < n; i++) {
        const a = 2 * Math.PI * i / n + Math.PI / 2;
        out.push([clip(x + r * Math.cos(a), 0.5, WIDTH - 0.5), clip(y + r * Math.sin(a), 0.5, HEIGHT - 0.5)]);
      }
      return out;
    }
    _add(u) {
      this.units.push(u);
      if (u.spec && u.spec.spawn_effect && !u.parent) this.effects.push(new Effect(this, u.spec.spawn_effect, u.owner, u.x, u.y, u.deploy_left));
      return u;
    }
    useAbility(player) {
      for (const u of this.units) {
        if (u.alive && u.owner === player && u.spec && u.spec.ability && u.deploy_left <= 0) {
          const ab = u.spec.ability, p = this.players[player];
          if (u.ability_cd > 0 || p.elixir < ab.cost) return false;
          p.elixir -= ab.cost; p.spent += ab.cost; u.ability_cd = ab.cooldown;
          this._ability(u, ab);
          this.events.push({ t: +this.time.toFixed(2), kind: "ability", player, card: u.card });
          return true;
        }
      }
      return false;
    }
    _ability(u, ab) {
      if (ab.duration) u.ability_left = ab.duration;
      if (ab.chain) {
        const hit = new Set();
        for (let i = 0; i < ab.chain; i++) {
          const cands = this.units.filter((o) => this._enemy(u, o) && !o.building && !hit.has(o.uid) && dist(u.x, u.y, o.x, o.y) <= u.spec.dash_secondary_range);
          if (!cands.length) break;
          const o = minBy(cands, (o) => dist(u.x, u.y, o.x, o.y));
          u.x = o.x; u.y = o.y; this._damage(o, u.spec.dash_damage, u); hit.add(o.uid);
        }
      }
      if (ab.summon) {
        const n = Math.trunc(clip(ab.min_count + u.souls, ab.min_count, ab.max_count));
        u.souls = 0;
        const spec = this.db.characters[ab.summon];
        for (let i = 0; i < n; i++) { const a = 2 * Math.PI * i / n; this._add(new Unit(this, spec, u.owner, u.x + 1.5 * Math.cos(a), u.y + 1.5 * Math.sin(a), null, 0.5)); }
      }
      if (ab.bomb_damage) { this._areaDamage(u.owner, u.x, u.y, ab.bomb_radius, ab.bomb_damage, true, true, 1.0, { source: u }); u.x = WIDTH - u.x; u.target = null; }
    }
    order(n) { return this.shuffleUpdates ? this.rng.perm(n) : [...Array(n).keys()]; }
    step() {
      if (this.done) return;
      const dt = DT;
      this.time += dt;
      const rate = ELIXIR_RATE * this.elixirMultiplier;
      for (const p of this.players) { const gain = rate * dt; p.leaked += Math.max(0, gain - (MAX_ELIXIR - p.elixir)); p.elixir = Math.min(MAX_ELIXIR, p.elixir + gain); }
      for (const e of this.effects.slice()) this._updateEffect(e, dt);
      this.effects = this.effects.filter((e) => e.alive);
      for (const pr of this.projectiles.slice()) this._updateProjectile(pr, dt);
      this.projectiles = this.projectiles.filter((p) => p.alive);
      const units = this.units.slice();
      for (const i of this.order(units.length)) { const u = units[i]; if (u.alive) this._updateUnit(u, dt); }
      this._collide();
      for (const u of units) if (u.alive && u.hp <= 0 && !u.timed) this._kill(u);
      this.units = this.units.filter((u) => u.alive);
      this._checkEnd();
    }
    _checkEnd() {
      const c0 = this.players[0].crowns, c1 = this.players[1].crowns;
      if (this.done) return;
      if (this.time >= REGULATION - 1e-9 && !this.overtime) {
        if (c0 !== c1) return this._finish();
        this.overtime = true; this._ot = [c0, c1];
      }
      if (this.overtime && (c0 !== this._ot[0] || c1 !== this._ot[1])) return this._finish();
      if (this.time >= OVERTIME_END - 1e-9) {
        const lows = [0, 1].map((o) => Math.min(...this.towers(o).map((t) => t.hp / t.max_hp)));
        this.done = true; this.winner = lows[0] > lows[1] ? 0 : lows[1] > lows[0] ? 1 : null;
      }
    }
    _finish() { this.done = true; const c0 = this.players[0].crowns, c1 = this.players[1].crowns; this.winner = c0 > c1 ? 0 : c1 > c0 ? 1 : null; }
    _enemy(u, o) { return o.alive && o.owner !== u.owner; }
    _targetable(a, o) {
      if (!this._enemy(a, o) || o.timed || o.parent || o.invisible) return false;
      if (o.deploy_left > 0) return false;
      const s = a.spec;
      if (a.tower) return true;
      if (o.flying && !s.attacks_air) return false;
      if (!o.flying && !s.attacks_ground) return false;
      if (s.buildings_only && !o.building) return false;
      if (s.troops_only && o.building) return false;
      return true;
    }
    _gap(a, b) { return dist(a.x, a.y, b.x, b.y) - b.radius - (a.tower ? 0 : a.radius); }
    _damage(o, amount, source = null, towerFactor = 1, buildingFactor = 1) {
      if (!o.alive || amount <= 0 || o.timed) return;
      if (o.tower) amount *= towerFactor; else if (o.building) amount *= buildingFactor;
      if (o.spec && o.spec.ability && o.ability_left > 0 && o.spec.ability.damage_taken) amount *= o.spec.ability.damage_taken;
      if (o.shield > 0) o.shield = Math.max(0, o.shield - amount); else o.hp -= amount;
      o.idle = 0;
      if (o.tower === "king") o.active = true;
      if (source && source.alive && o.spec && o.spec.reflect_damage && dist(o.x, o.y, source.x, source.y) <= o.spec.reflect_radius + source.radius + o.radius) {
        this._damage(source, o.spec.reflect_damage, null, 0.8);
        this._buff(source, this.db.buffs.ZapFreeze, 0.5, o.owner);
      }
    }
    _buff(o, buff, time, owner) {
      const skip = !buff || !o.alive || (o.tower && (buff.speed === 0 || buff.switch_team));
      if (skip && !(buff && o.tower && buff.speed === 0)) return;
      if (buff.hit_speed <= 0 && !o.tower) {
        o.lock_time = 0; o.charging = false; o.walked = 0;
        if (o.spec && o.spec.load_first_hit) o.cooldown = o.spec.hit_speed;
      }
      const cur = o.buffs.get(buff.name);
      if (!cur || cur[1] < time) o.buffs.set(buff.name, [buff, time, owner, 0]);
    }
    _areaDamage(owner, x, y, radius, damage, air, ground, towerFactor, o = {}) {
      const hit = [];
      for (const u of this.units) {
        if (!u.alive || u.owner === owner || u.timed || u.parent) continue;
        if ((u.flying && !air) || (!u.flying && !ground)) continue;
        if (dist(x, y, u.x, u.y) <= radius + u.radius) {
          if (o.exclude && o.exclude.has(u.uid)) continue;
          this._damage(u, damage, o.source || null, towerFactor);
          if (o.buff) this._buff(u, o.buff, o.buffTime || 0, owner);
          if (o.pushback && !u.building && !(u.spec && u.spec.ignore_pushback)) this._push(u, x, y, o.pushback);
          hit.push(u);
        }
      }
      return hit;
    }
    _push(u, x, y, d0) {
      const dx = u.x - x, dy = u.y - y, d = Math.hypot(dx, dy) || 1;
      u.x = clip(u.x + dx / d * d0, 0.5, WIDTH - 0.5); u.y = clip(u.y + dy / d * d0, 0.5, HEIGHT - 0.5);
      u.walked = 0; u.charging = false;
    }
    _updateUnit(u, dt) {
      for (const name of [...u.buffs.keys()]) {
        let [b, t, owner, acc] = u.buffs.get(name);
        if (b.dps || b.heal_per_second) {
          const freq = b.hit_frequency || 1.0;
          acc += dt;
          while (acc >= freq) {
            acc -= freq;
            if (b.dps) this._damage(u, b.dps * freq, null, b.tower_factor, b.building_factor);
            if (b.heal_per_second && !u.tower) u.hp = Math.min(u.max_hp, u.hp + b.heal_per_second * freq);
          }
        }
        t -= dt;
        if (t <= 0) u.buffs.delete(name); else u.buffs.set(name, [b, t, owner, acc]);
      }
      u.ability_cd = Math.max(0, u.ability_cd - dt);
      u.ability_left = Math.max(0, u.ability_left - dt);
      if (u.timed) { u.deploy_left -= dt; if (u.deploy_left <= 0) this._kill(u); return; }
      if (u.deploy_left > 0) { u.deploy_left -= dt; return; }
      if (u.tower) { if (u.active) this._attackLogic(u, dt, false); return; }
      const s = u.spec;
      if (u.life_left != null) {
        u.life_left -= dt; u.hp -= u.max_hp / s.life_time * dt;
        if (u.life_left <= 0) { u.hp = 0; return; }
      }
      if (s.elixir_every) {
        u.elixir_timer += dt;
        if (u.elixir_timer >= s.elixir_every) { u.elixir_timer -= s.elixir_every; const p = this.players[u.owner]; p.elixir = Math.min(MAX_ELIXIR, p.elixir + 1); }
      }
      if (u.parent) { if (!u.parent.alive) { u.hp = 0; return; } u.x = u.parent.x; u.y = u.parent.y; }
      if (u.spawn_timer != null) {
        u.spawn_timer -= dt * u.factor("spawn_speed");
        if (u.spawn_timer <= 0) {
          const n = Math.max(1, s.spawn_number);
          for (let i = 0; i < n; i++) {
            const a = 2 * Math.PI * (i + u.spawned) / Math.max(n, 3), r = s.spawn_radius || 1.0;
            this._add(new Unit(this, s.spawn_character, u.owner, u.x + r * Math.cos(a) * 0.7, u.y + r * Math.sin(a) * 0.7, u.card, 0.4));
          }
          u.spawned += n;
          u.spawn_timer = s.spawn_pause || s.spawn_interval || 5.0;
          if (s.spawn_limit && u.spawned >= s.spawn_limit) { u.alive = false; return; }
        }
      }
      if (s.heal_when_idle && u.idle >= 1.0) u.hp = Math.min(u.max_hp, u.hp + 0.03 * u.max_hp * dt);
      u.dash_cd = Math.max(0, u.dash_cd - dt); u.hook_cd = Math.max(0, u.hook_cd - dt);
      if (u.dashing > 0) { u.dashing -= dt; return; }
      this._attackLogic(u, dt, s.speed > 0 && !u.parent);
    }
    _findTarget(u) {
      const sight = u.tower ? TOWER[u.tower].range : u.spec.sight;
      let best = null, bestD = sight;
      for (const o of this.units) if (this._targetable(u, o)) { const d = this._gap(u, o); if (d <= bestD) { best = o; bestD = d; } }
      if (!best && !u.tower && u.spec.speed > 0) {
        if (u.spec.buildings_only) { const b = this.units.filter((o) => this._targetable(u, o)); if (b.length) best = minBy(b, (o) => dist(u.x, u.y, o.x, o.y)); }
        if (!best && !u.spec.troops_only) best = this.laneTarget(u.owner, laneOf(u.x));
      }
      return best;
    }
    _attackLogic(u, dt, canMove) {
      const hitF = u.factor("hit_speed");
      let t = u.target;
      const rng = u.tower ? TOWER[u.tower].range : u.spec.range;
      if (t && (!t.alive || t.invisible || t.parent)) t = null;
      if (t && !t.building && !u.tower && this._gap(u, t) > u.spec.sight + 1.5) t = null;
      if (t && u.tower && this._gap(u, t) > rng) t = null;
      u.retarget -= dt;
      if (!t || (u.retarget <= 0 && this._gap(u, t) > rng)) {
        u.retarget = 0.5;
        const nt = this._findTarget(u);
        if (nt !== t) { u.lock_time = 0; if (nt && !u.tower) u.cooldown = Math.max(u.cooldown, u.spec.load_time * 0.5); }
        t = nt;
      }
      u.target = t;
      if (hitF <= 0) return;
      u.cooldown -= dt * hitF;
      u.idle += dt;
      if (!t) return;
      const gap = this._gap(u, t);
      const minRng = u.tower ? 0 : u.spec.min_range;
      if (!u.tower && this._specialMove(u, t, gap)) return;
      if (gap <= rng && gap >= minRng - 1e-6) {
        u.lock_time += dt; u.idle = 0;
        if (u.cooldown <= 0) { this._attack(u, t); u.cooldown = u.tower ? TOWER[u.tower].hit_speed : u.spec.hit_speed; u.walked = 0; u.charging = false; }
        return;
      }
      if (canMove) this._move(u, t, dt);
    }
    _specialMove(u, t, gap) {
      const s = u.spec;
      if (s.dash_damage && s.dash_max && u.dash_cd <= 0 && s.dash_min <= gap && gap <= s.dash_max && !t.building && !u.charging) {
        const dx = t.x - u.x, dy = t.y - u.y, d = Math.hypot(dx, dy) || 1;
        u.x = t.x - dx / d * (t.radius + u.radius); u.y = t.y - dy / d * (t.radius + u.radius);
        if (s.dash_radius) this._areaDamage(u.owner, u.x, u.y, s.dash_radius, s.dash_damage, false, true, 1.0, { source: u, pushback: 0.5 });
        else this._damage(t, s.dash_damage, u);
        u.dash_cd = s.dash_cooldown || 1.0; u.dashing = 0.3; u.cooldown = Math.max(u.cooldown, s.hit_speed * 0.5);
        return true;
      }
      if (s.special_range && u.hook_cd <= 0 && s.special_min_range <= gap && gap <= s.special_range) {
        if (t.building) {
          const dx = t.x - u.x, dy = t.y - u.y, d = Math.hypot(dx, dy) || 1;
          u.x = t.x - dx / d * (t.radius + u.radius + 0.5); u.y = t.y - dy / d * (t.radius + u.radius + 0.5);
        } else if (!(t.spec && t.spec.ignore_pushback && t.spec.mass >= 18)) {
          const dx = u.x - t.x, dy = u.y - t.y, d = Math.hypot(dx, dy) || 1;
          t.x = u.x - dx / d * (t.radius + u.radius + 0.3); t.y = u.y - dy / d * (t.radius + u.radius + 0.3);
          if (s.special_projectile && s.special_projectile.buff) this._buff(t, s.special_projectile.buff, s.special_projectile.buff_time, u.owner);
        }
        u.hook_cd = 8.0;
        return true;
      }
      return false;
    }
    _move(u, t, dt) {
      const s = u.spec;
      let speed = s.speed * u.factor("speed");
      if (s.charge_damage && u.walked >= CHARGE_AFTER) u.charging = true;
      if (u.charging) speed *= s.charge_speed || 2.0;
      if (speed <= 0) return;
      let wx = t.x, wy = t.y;
      if (!u.flying && !s.jump) {
        const su = sideOf(u.y); let st = sideOf(t.y);
        if (st === null) st = t.y < RIVER_Y ? 0 : 1;
        if (su !== null && su !== st) { wx = Math.abs(LANE_X[0] - u.x) <= Math.abs(LANE_X[1] - u.x) ? LANE_X[0] : LANE_X[1]; wy = RIVER_Y; }
      }
      const dx = wx - u.x, dy = wy - u.y, d = Math.hypot(dx, dy);
      if (d < 1e-6) return;
      const st = Math.min(d, speed * dt);
      u.x += dx / d * st; u.y += dy / d * st; u.walked += st;
    }
    _attack(u, t) {
      if (u.tower) { this._damage(t, TOWER[u.tower].damage, u); return; }
      const s = u.spec;
      let dmg = s.damage, pushback = 0;
      if (s.vd) {
        if (s.vd.mode === "time") {
          const st = s.vd.stage_time;
          const stage = u.lock_time < st[0] ? 0 : u.lock_time < st[0] + st[1] ? 1 : 2;
          dmg = s.vd.damage[stage];
        } else {
          dmg = s.vd.damage[u.combo % 3];
          if (u.combo % 3 === 2) pushback = s.vd.pushback;
          u.combo += 1;
        }
      }
      if (u.charging && s.charge_damage) dmg = s.charge_damage;
      let targets = [t];
      if (s.targets > 1) {
        const others = this.units.filter((o) => o !== t && this._targetable(u, o) && this._gap(u, o) <= s.range)
          .sort((a, b) => this._gap(u, a) - this._gap(u, b));
        targets = targets.concat(others.slice(0, s.targets - 1));
      }
      for (const tg of targets) {
        if (s.projectile) this._shoot(u, tg, s.projectile);
        else if (s.splash) {
          const cx = s.self_splash ? u.x : tg.x, cy = s.self_splash ? u.y : tg.y;
          this._areaDamage(u.owner, cx, cy, s.splash, dmg, s.attacks_air, s.attacks_ground, s.tower_factor, { source: u, pushback });
        } else {
          this._damage(tg, dmg, u, s.tower_factor);
          if (pushback && !tg.building) this._push(tg, u.x, u.y, pushback);
        }
        if (s.buff_on_damage) this._buff(tg, s.buff_on_damage, s.buff_on_damage_time, u.owner);
      }
      if (s.attack_pushback) this._push(u, t.x, t.y, s.attack_pushback * 0.5);
      if (s.heal_when_idle) for (const o of this.units) if (o.alive && o.owner === u.owner && !o.building && dist(o.x, o.y, u.x, u.y) <= 4.0) o.hp = Math.min(o.max_hp, o.hp + 48);
      if (s.kamikaze) u.hp = 0;
    }
    _shoot(u, t, spec) {
      const s = u.spec;
      if (s.projectiles > 1 && spec.pierce_range) {
        const base = Math.atan2(t.y - u.y, t.x - u.x);
        for (let i = 0; i < s.projectiles; i++) {
          const a = base + (i - (s.projectiles - 1) / 2) * 0.06;
          this.projectiles.push(new Projectile(this, spec, u.owner, u.x, u.y, { tx: t.x, ty: t.y, source: u, pierce: true, stop_on_hit: true, direction: [Math.cos(a), Math.sin(a)] }));
        }
        return;
      }
      if (spec.pierce_range > 1.0) { this.projectiles.push(new Projectile(this, spec, u.owner, u.x, u.y, { tx: t.x, ty: t.y, source: u, pierce: true })); return; }
      this.projectiles.push(new Projectile(this, spec, u.owner, u.x, u.y, { target: t, source: u, chain_left: spec.chain_count }));
    }
    _updateProjectile(pr, dt) {
      const spec = pr.spec, step = spec.speed * dt;
      if (pr.travelled < 0) { pr.travelled += step; return; }
      if (pr.pierce) {
        pr.x += pr.dir[0] * step; pr.y += pr.dir[1] * step; pr.travelled += step;
        const r = spec.pierce_radius || spec.radius || 0.5;
        for (const o of this.units) {
          if (!o.alive || o.owner === pr.owner || pr.hit.has(o.uid) || o.timed || o.parent) continue;
          if ((o.flying && !spec.air) || (!o.flying && !spec.ground)) continue;
          if (dist(pr.x, pr.y, o.x, o.y) <= r + o.radius) {
            pr.hit.add(o.uid);
            this._damage(o, pr.damage, pr.source, pr.tower_factor);
            if (spec.buff) this._buff(o, spec.buff, spec.buff_time, pr.owner);
            if (spec.pushback && !o.building && !(o.spec && o.spec.ignore_pushback)) {
              o.x = clip(o.x + pr.dir[0] * spec.pushback, 0.5, WIDTH - 0.5); o.y = clip(o.y + pr.dir[1] * spec.pushback, 0.5, HEIGHT - 0.5);
            }
            if (pr.stop_on_hit) { pr.alive = false; return; }
          }
        }
        const limit = spec.pierce_range || 6.0;
        if (pr.travelled >= limit || !(pr.x >= 0 && pr.x <= WIDTH && pr.y >= 0 && pr.y <= HEIGHT)) {
          if (spec.returns && !pr.returning) { pr.returning = true; pr.travelled = 0; pr.hit.clear(); pr.dir = [-pr.dir[0], -pr.dir[1]]; return; }
          if (spec.spawn_character) for (let i = 0; i < Math.max(1, spec.spawn_count); i++) this._add(new Unit(this, spec.spawn_character, pr.owner, pr.x, pr.y, null, 0.5));
          pr.alive = false;
        }
        return;
      }
      if (spec.homing && pr.target && pr.target.alive) { pr.tx = pr.target.x; pr.ty = pr.target.y; }
      const dx = pr.tx - pr.x, dy = pr.ty - pr.y, d = Math.hypot(dx, dy);
      if (d <= step) { pr.x = pr.tx; pr.y = pr.ty; this._impact(pr); pr.alive = false; }
      else { pr.x += dx / d * step; pr.y += dy / d * step; }
    }
    _impact(pr) {
      const spec = pr.spec;
      let hit = [];
      if (spec.radius > 0) {
        hit = this._areaDamage(pr.owner, pr.x, pr.y, spec.radius, pr.damage, spec.air || !spec.ground, spec.ground || !spec.air, pr.tower_factor,
          { source: pr.source, buff: spec.buff, buffTime: spec.buff_time, pushback: spec.pushback });
      } else if (pr.target && pr.target.alive) {
        this._damage(pr.target, pr.damage, pr.source, pr.tower_factor);
        if (spec.buff) this._buff(pr.target, spec.buff, spec.buff_time, pr.owner);
        hit = [pr.target];
      }
      if (spec.chain_count && hit.length) {
        const done = new Set(hit.map((o) => o.uid));
        let cur = hit[0];
        for (let i = 0; i < spec.chain_count - 1; i++) {
          const cands = this.units.filter((o) => o.alive && o.owner !== pr.owner && !done.has(o.uid) && !o.timed && !o.parent && dist(cur.x, cur.y, o.x, o.y) <= spec.chain_radius);
          if (!cands.length) break;
          const c0 = cur;
          cur = minBy(cands, (o) => dist(c0.x, c0.y, o.x, o.y));
          done.add(cur.uid);
          this._damage(cur, pr.damage, pr.source, pr.tower_factor);
          if (spec.buff) this._buff(cur, spec.buff, spec.buff_time, pr.owner);
        }
      }
      if (spec.spawn_character) {
        const n = Math.max(1, spec.spawn_count);
        for (let i = 0; i < n; i++) {
          const a = 2 * Math.PI * i / n, m = spec.spawn_count > 1 ? 1 : 0;
          this._add(new Unit(this, spec.spawn_character, pr.owner, pr.x + 0.6 * Math.cos(a) * m, pr.y + 0.6 * Math.sin(a) * m, null, 0.5));
        }
      }
      if (spec.spawn_projectile) {
        for (let i = 0; i < Math.max(1, spec.spawn_projectile_count); i++) {
          const a = Math.atan2(pr.dir[1], pr.dir[0]) + (i - 2) * 0.12;
          this.projectiles.push(new Projectile(this, spec.spawn_projectile, pr.owner, pr.x, pr.y, { source: pr.source, pierce: true, direction: [Math.cos(a), Math.sin(a)], tx: pr.x + Math.cos(a), ty: pr.y + Math.sin(a) }));
        }
      }
    }
    _updateEffect(e, dt) {
      if (e.delay > 0) { e.delay -= dt; return; }
      const spec = e.spec;
      if (!e.started) {
        e.started = true;
        if (spec.damage && !spec.hit_biggest) this._areaDamage(e.owner, e.x, e.y, spec.radius, spec.damage, spec.air, spec.ground, spec.tower_factor, { buff: spec.buff, buffTime: spec.buff_time, pushback: spec.pushback });
        if (spec.clone) {
          for (const o of this.units.slice()) {
            if (o.alive && o.owner === e.owner && !o.building && !o.tower && !o.parent && dist(o.x, o.y, e.x, e.y) <= spec.radius + o.radius) {
              const c = new Unit(this, o.spec, o.owner, o.x + 0.4, o.y, o.card, 0.0);
              c.hp = c.max_hp = 1; c.shield = 0; this._add(c);
            }
          }
        }
      }
      e.next_hit -= dt;
      if (e.next_hit <= 0) {
        e.next_hit = spec.hit_interval || 1e9;
        if (spec.hit_biggest && spec.projectile) {
          const cands = this.units.filter((o) => o.alive && o.owner !== e.owner && !o.timed && !o.parent && !e.struck.has(o.uid) && dist(o.x, o.y, e.x, e.y) <= spec.radius + o.radius);
          if (cands.length && e.struck.size < 3) {
            const o = maxBy(cands, (o) => o.hp + o.shield);
            e.struck.add(o.uid);
            const p = spec.projectile;
            this._damage(o, p.damage, null, p.tower_factor);
            if (p.buff) this._buff(o, p.buff, p.buff_time, e.owner);
          }
        } else if (spec.buff && !(spec.damage && spec.duration <= 0.01)) {
          for (const o of this.units) {
            if (!o.alive || o.timed || o.parent) continue;
            if (spec.only_own !== (o.owner === e.owner)) continue;
            if (spec.ignore_buildings && o.building) continue;
            if ((o.flying && !spec.air) || (!o.flying && !spec.ground)) continue;
            if (dist(o.x, o.y, e.x, e.y) <= spec.radius + o.radius) this._buff(o, spec.buff, spec.buff_time || spec.hit_interval, e.owner);
          }
        }
      }
      if (spec.buff && spec.buff.pull) {
        for (const o of this.units) {
          if (o.alive && o.owner !== e.owner && !o.building && !o.parent && dist(o.x, o.y, e.x, e.y) <= spec.radius + o.radius) {
            const dx = e.x - o.x, dy = e.y - o.y, d = Math.hypot(dx, dy);
            if (d > 0.2) { const k = Math.min(d, 4.5 * dt / Math.max(1, (o.spec ? o.spec.mass : 6) / 6)); o.x += dx / d * k; o.y += dy / d * k; }
          }
        }
      }
      if (e.spawn_next != null) {
        e.spawn_next -= dt;
        if (e.spawn_next <= 0) {
          e.spawn_next = spec.spawn_interval || 0.5;
          const a = this.rng.random() * 2 * Math.PI, r = 0.5 + this.rng.random() * (spec.radius - 0.5);
          this._add(new Unit(this, spec.spawn_character, e.owner, clip(e.x + r * Math.cos(a), 0.5, WIDTH - 0.5), clip(e.y + r * Math.sin(a), 0.5, HEIGHT - 0.5), null, 0.5));
        }
      }
      e.time_left -= dt;
      if (e.time_left <= 0) e.alive = false;
    }
    _collide() {
      const movers = this.units.filter((u) => u.alive && !u.building && !u.timed && !u.parent && u.deploy_left <= 0);
      const solids = this.units.filter((u) => u.alive && u.building && !u.timed);
      for (let i = 0; i < movers.length; i++) {
        const a = movers[i];
        for (let j = i + 1; j < movers.length; j++) {
          const b = movers[j];
          if (a.flying !== b.flying) continue;
          let dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
          const overlap = a.radius + b.radius - d;
          if (overlap > 0) {
            if (d < 1e-6) { dx = 0.01 * (a.uid < b.uid ? 1 : -1); dy = 0; d = 0.01; }
            const ma = a.spec.mass, mb = b.spec.mass, wa = mb / (ma + mb), wb = ma / (ma + mb);
            const push = Math.min(overlap, 0.3) * 0.5;
            a.x -= dx / d * push * wa * 2; a.y -= dy / d * push * wa * 2;
            b.x += dx / d * push * wb * 2; b.y += dy / d * push * wb * 2;
          }
        }
        if (!a.flying) {
          for (const b of solids) {
            const dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy), overlap = a.radius + b.radius - d;
            if (overlap > 0 && d > 1e-6) { a.x += dx / d * Math.min(overlap, 0.3); a.y += dy / d * Math.min(overlap, 0.3); }
          }
        }
        a.x = clip(a.x, 0.5, WIDTH - 0.5); a.y = clip(a.y, 0.5, HEIGHT - 0.5);
      }
    }
    _kill(u) {
      if (!u.alive && !u.timed) return;
      u.alive = false;
      const s = u.spec;
      if (u.tower) {
        this._dead_towers.push(u); u.hp = 0;
        const attacker = 1 - u.owner;
        this.events.push({ t: +this.time.toFixed(2), kind: "tower", player: attacker, tower: u.tower, lane: u.lane });
        if (u.tower === "king") { this.players[attacker].crowns = 3; this._finish(); }
        else { this.players[attacker].crowns += 1; this.tower(u.owner, "king").active = true; }
        return;
      }
      for (const c of u.children) if (c.alive) c.hp = 0;
      if (s.death_damage) this._areaDamage(u.owner, u.x, u.y, s.death_damage_radius, s.death_damage, true, true, s.tower_factor, { pushback: s.death_pushback });
      for (const [spec, count] of [[s.death_spawn, s.death_spawn_count], [s.death_spawn2, s.death_spawn_count2]]) {
        if (!spec) continue;
        for (let i = 0; i < count; i++) {
          const a = 2 * Math.PI * i / Math.max(1, count), r = count > 1 ? s.death_spawn_radius : 0;
          this._add(new Unit(this, spec, u.owner, u.x + r * Math.cos(a), u.y + r * Math.sin(a), u.card, 0.3));
        }
      }
      if (s.death_projectile) this._impact(new Projectile(this, s.death_projectile, u.owner, u.x, u.y, { tx: u.x, ty: u.y }));
      if (s.death_effect) this.effects.push(new Effect(this, s.death_effect, u.owner, u.x, u.y));
      if (s.elixir_on_death) { const p = this.players[1 - u.owner]; p.elixir = Math.min(MAX_ELIXIR, p.elixir + s.elixir_on_death); }
      for (const [b, , owner] of u.buffs.values()) if (b.death_spawn && owner !== u.owner && !u.building) this._add(new Unit(this, b.death_spawn, owner, u.x, u.y, null, 0.5));
      if (!u.building && !u.timed) {
        for (const o of this.units) if (o.alive && o.owner !== u.owner && o.spec && o.spec.ability && o.spec.ability.summon && dist(o.x, o.y, u.x, u.y) <= 5.0) o.souls += 1;
      }
    }
  }

  // ------------------------------------------------------------ strategy.py
  const THREAT_LINE = 20;
  const HEAVY = new Set(["Golem", "Lava Hound", "Electro Giant", "Goblin Giant", "Giant", "Elixir Golem", "Royal Giant"]);
  const DROP_ON_TOWER = new Set(["Miner", "Goblin Barrel", "Graveyard", "Goblin Drill"]);
  const SIEGE = new Set(["X-Bow", "Mortar"]);

  function hitsAir(card) {
    if (card.type === "spell") {
      if (card.effect) return !!card.effect.air;
      if (card.projectile) return !!(card.projectile.air || (card.projectile.radius > 0 && !card.projectile.ground));
      return false;
    }
    return card.summons.some(([spec]) => spec.attacks_air || (spec.spawn_character && spec.spawn_character.attacks_air));
  }

  class View {
    constructor(sim, me) { this.sim = sim; this.me = me; this.foe = 1 - me; this.p = sim.players[me]; this.db = sim.db; }
    fy(y) { return frameY(this.me, y); }
    get elixir() { return this.p.elixir; }
    hand() { return this.p.hand.slice(); }
    playable() { return this.p.hand.filter((c) => this.sim.canPlay(this.me, c)); }
    enemies() { return this.sim.units.filter((u) => u.alive && u.owner === this.foe && !u.tower && !u.timed && !u.parent); }
    mine() { return this.sim.units.filter((u) => u.alive && u.owner === this.me && !u.tower && !u.timed && !u.parent); }
    threats(lane) { return this.enemies().filter((u) => laneOf(u.x) === lane && this.fy(u.y) < THREAT_LINE); }
    threatHp(lane) { return this.threats(lane).reduce((s, u) => s + u.hp + u.shield, 0); }
    defenders(lane) { return this.mine().filter((u) => laneOf(u.x) === lane && this.fy(u.y) < RIVER_LO + 1 && !u.building); }
    pushers(lane) { return this.mine().filter((u) => laneOf(u.x) === lane && this.fy(u.y) >= 9 && !u.building); }
    enemyBackTank() { for (const u of this.enemies()) if (HEAVY.has(u.card) && this.fy(u.y) > 24) return laneOf(u.x); return null; }
    towerFrac(mine, lane) { const t = this.sim.tower(mine ? this.me : this.foe, "princess", lane); return Math.max(t.hp, 0) / t.max_hp; }
    weakLane() { const f = [this.towerFrac(false, 0), this.towerFrac(false, 1)]; if (f[0] === f[1]) return this.sim.rng.int(2); return f[0] <= f[1] ? 0 : 1; }
    enemyElixir() { return this.sim.players[this.foe].elixir; }
    profile(units) {
      const p = { air: false, building_targeter: false, swarm: false, splash: false, ranged: false, tank_killer: false, tank: false, building: false, hp: 0 };
      for (const u of units) {
        const s = u.spec;
        p.hp += u.hp + u.shield; p.air = p.air || u.flying; p.building = p.building || u.building;
        if (u.building) continue;
        p.building_targeter = p.building_targeter || !!s.buildings_only;
        p.tank = p.tank || u.max_hp >= 2000;
        p.splash = p.splash || !!(s.splash || (s.projectile && s.projectile.radius > 0));
        p.ranged = p.ranged || s.range >= 4.0;
        p.tank_killer = p.tank_killer || (s.damage / Math.max(s.hit_speed, 0.1) >= 300 && s.range < 3);
      }
      p.swarm = units.filter((u) => !u.building && u.max_hp <= 400).length >= 3;
      return p;
    }
  }

  function cardValue(db, u) {
    if (!u.card || !db.cards[u.card]) return 0.5;
    const c = db.cards[u.card];
    return c.elixir / Math.max(1, c.summons.reduce((s, [, k]) => s + k, 0));
  }

  function spellSpot(view, name, lane = null) {
    const c = view.db.cards[name];
    let radius, air, ground, dmg;
    if (c.effect) {
      radius = c.effect.radius; air = c.effect.air; ground = c.effect.ground;
      dmg = c.effect.damage || (c.effect.buff ? c.effect.buff.dps * c.effect.duration : 0);
      if (c.effect.projectile) dmg = c.effect.projectile.damage;
    } else if (c.projectile) {
      const p = c.projectile;
      radius = p.radius || p.pierce_radius || 1.5; air = p.air || !p.ground; ground = p.ground || !p.air;
      dmg = p.damage * Math.max(1, c.waves);
    } else return [0, 0, 0];
    if (["Freeze", "Rage", "Clone", "Mirror", "Tornado", "Graveyard"].includes(name)) return [0, 0, 0];
    let best = [0, 0, 0];
    const en = view.enemies();
    for (const u of en) {
      if (lane !== null && laneOf(u.x) !== lane) continue;
      let v = 0;
      for (const o of en) {
        if ((o.flying && !air) || (!o.flying && !ground)) continue;
        if (dist(u.x, u.y, o.x, o.y) <= radius + o.radius) v += Math.min(1, dmg / Math.max(o.hp + o.shield, 1)) * cardValue(view.db, o);
      }
      if (v > best[0]) best = [v, u.x, u.y];
    }
    return best;
  }

  // strategy.PLACE_DEFAULTS / COACH_DEFAULTS
  const PLACE = { building_y: 9.0, melee_ahead: 2.5, ranged_y: 4.5, ranged_dx: 1.5, bridge_y: 14.0, support_behind: 2.0 };
  const COACH = { enough_defense: 0.8, hold_line: 10.0, trade_margin: 1.5, push_hp: 800.0, punish_below: 2.5,
                  leak_single: 9.8, leak_double: 6.5 };

  function place(view, name, lane) {
    const P = PLACE;
    const db = view.db, me = view.me;
    let c = db.cards[name];
    if (c.special === "mirror" && view.p.last_card) { c = db.cards[view.p.last_card]; name = c.name; }
    const lx = LANE_X[lane], tc = lane === 0 ? 1 : -1;
    const et = view.sim.laneTarget(me, lane);
    if (DROP_ON_TOWER.has(name)) {
      const off = (name === "Miner" || name === "Goblin Drill") ? 0 : -1;
      return [clip(et.x + tc * 0.5, 0.5, WIDTH - 0.5), absY(me, view.fy(et.y) + off - 1.5)];
    }
    if (c.type === "spell" && !c.summons.length) {
      if (["Rage", "Freeze", "Clone"].includes(name)) {
        const pushers = view.pushers(lane).length ? view.pushers(lane) : view.mine();
        if (pushers.length) { const lead = maxBy(pushers, (u) => view.fy(u.y)); return [lead.x, lead.y]; }
        return [lx, absY(me, 20)];
      }
      if (name === "Tornado") {
        const th = view.threats(lane);
        if (th.length) { const lead = minBy(th, (u) => view.fy(u.y)); return [clip(lead.x + tc * 1.5, 0.5, WIDTH - 0.5), lead.y]; }
      }
      const [v, x, y] = spellSpot(view, name, lane);
      if (c.special === "rolling") return v > 0 ? [x, absY(me, Math.max(0.6, view.fy(y) - 2))] : [lx, absY(me, 16.5)];
      if (v > 0) return [x, y];
      return [et.x, et.y];
    }
    if (c.type === "spell") {
      const th = view.threats(lane);
      if (th.length) { const lead = minBy(th, (u) => view.fy(u.y)); return [lead.x, absY(me, Math.min(14, view.fy(lead.y)))]; }
      return [lx, absY(me, 10)];
    }
    const threats = view.threats(lane);
    if (c.type === "building") {
      if (SIEGE.has(name)) return [lx + tc * 1.0, absY(me, name === "X-Bow" ? 14 : 12.5)];
      if (c.role === "spawner") return [WIDTH / 2 - tc * 2, absY(me, 2.5)];
      return [WIDTH / 2 - tc * 1.5, absY(me, P.building_y)];
    }
    if (threats.length) {
      const lead = minBy(threats, (u) => view.fy(u.y)), lfy = view.fy(lead.y);
      if (["ranged", "splash_air", "champion"].includes(c.role) && !["Golden Knight", "Mighty Miner", "Monk"].includes(c.name)) return [lx + P.ranged_dx * tc, absY(me, P.ranged_y)];
      if (lfy > RIVER_LO) return [lx + tc, absY(me, 11)];
      if (["swarm", "cycle", "air"].includes(c.role)) return [lead.x, absY(me, Math.max(0.5, lfy - 1))];
      return [clip(lead.x + tc, 0.5, WIDTH - 0.5), absY(me, Math.max(0.5, lfy - P.melee_ahead))];
    }
    if (c.role === "win_condition") {
      if (HEAVY.has(name) && view.elixir < 9 && view.sim.elixirMultiplier === 1) return [WIDTH / 2 - tc * 1.5, absY(me, 1)];
      return [lx, absY(me, P.bridge_y)];
    }
    const pushers = view.pushers(lane);
    if (pushers.length) { const lead = maxBy(pushers, (u) => view.fy(u.y)); return [clip(lead.x, 0.5, WIDTH - 0.5), absY(me, Math.min(14, view.fy(lead.y) - P.support_behind))]; }
    if (["ranged", "splash_air"].includes(c.role)) return [lx, absY(me, 10)];
    return [lx, absY(me, P.bridge_y)];
  }

  const COUNTER_ROLES = {
    building_targeter: ["tank_killer", "building", "swarm", "cycle", "frontline", "champion", "splash_ground"],
    air: ["splash_air", "ranged", "air", "building", "champion", "small_spell"],
    swarm: ["splash_ground", "splash_air", "small_spell", "frontline"],
    tank_killer: ["swarm", "cycle", "building", "frontline", "splash_air"],
    ranged: ["frontline", "tank_killer", "small_spell", "big_spell", "air"],
    tank: ["tank_killer", "building", "swarm", "champion"],
    other: ["frontline", "tank_killer", "splash_ground", "ranged", "swarm", "champion", "cycle"],
  };

  const Coach = {
    cardFit(view, name, prof) {
      const c = view.db.cards[name];
      if (prof.air && !hitsAir(c)) {
        const onlyAir = prof.air && !prof.building_targeter;
        if (onlyAir || view.enemies().filter((u) => view.fy(u.y) < THREAT_LINE).every((u) => u.flying)) return false;
      }
      if (prof.swarm && (c.role === "small_spell" || c.role === "big_spell")) return spellSpot(view, name)[0] >= c.elixir * 0.8;
      if (c.type === "spell" && !c.summons.length) return spellSpot(view, name)[0] >= c.elixir + 0.5;
      if (["win_condition", "spawner", "utility"].includes(c.role) && c.name !== "Goblin Giant") return false;
      return true;
    },
    suggest(view) { const [card, lane] = this.plan(view); return [card, lane]; },
    // [card or null, lane, reason]: finish, defend, hold, trade, counterpush, punish, leak or null
    plan(view) {
      const db = view.db, play = view.playable(), roles = {};
      for (const n of play) roles[n] = db.cards[n].role;
      for (const n of play) {
        const c = db.cards[n];
        if (c.type === "spell" && c.projectile && c.special !== "rolling") {
          const dmg = c.projectile.damage * Math.max(1, c.waves) * c.projectile.tower_factor;
          for (const lane of [0, 1]) { const t = view.sim.tower(view.foe, "princess", lane); if (t.alive && t.hp <= dmg) return [n, lane, "finish"]; }
        }
      }
      const lanes = [0, 1].sort((a, b) => view.threatHp(b) - view.threatHp(a));
      for (const lane of lanes) {
        const threats = view.threats(lane);
        if (!threats.length) continue;
        const prof = view.profile(threats);
        const defHp = view.defenders(lane).reduce((s, u) => s + u.hp, 0);
        if (defHp >= COACH.enough_defense * prof.hp && !prof.building_targeter) continue;
        let keys = ["air", "building_targeter", "tank", "swarm", "tank_killer", "ranged"].filter((k) => prof[k]);
        if (!keys.length) keys = ["other"];
        for (const key of keys) for (const role of COUNTER_ROLES[key]) {
          const cands = play.filter((n) => roles[n] === role && this.cardFit(view, n, prof));
          if (cands.length) return [minBy(cands, (n) => db.cards[n].elixir), lane, "defend"];
        }
        if (Math.min(...threats.map((u) => view.fy(u.y))) < COACH.hold_line) return [null, lane, "hold"];
      }
      for (const n of play) {
        const c = db.cards[n];
        if (c.role === "small_spell" || c.role === "big_spell") for (const lane of [0, 1]) if (spellSpot(view, n, lane)[0] >= c.elixir + COACH.trade_margin) return [n, lane, "trade"];
      }
      for (const lane of [0, 1]) {
        const push = view.pushers(lane), pushHp = push.reduce((s, u) => s + u.hp, 0);
        if (pushHp >= COACH.push_hp && view.elixir >= 4) {
          if (push.some((u) => view.fy(u.y) > 20) && pushHp >= 1500) for (const n of play) if ((n === "Rage" || n === "Freeze") && view.threats(lane).length === 0) return [n, lane, "counterpush"];
          for (const role of ["ranged", "splash_air", "splash_ground", "air", "swarm", "frontline", "champion"]) {
            const cands = play.filter((n) => roles[n] === role);
            if (cands.length) return [cands[0], lane, "counterpush"];
          }
        }
      }
      const wins = play.filter((n) => roles[n] === "win_condition");
      const back = view.enemyBackTank();
      if (wins.length && back !== null) return [wins[0], 1 - back, "punish"];
      if (wins.length && view.enemyElixir() <= COACH.punish_below && view.elixir >= db.cards[wins[0]].elixir) return [wins[0], view.weakLane(), "punish"];
      const full = view.sim.elixirMultiplier === 1 ? COACH.leak_single : COACH.leak_double;
      if (view.elixir >= full) {
        const lane = view.weakLane();
        if (wins.length) return [wins[0], lane, "leak"];
        for (const role of ["spawner", "champion", "frontline", "tank_killer", "splash_ground", "ranged", "swarm", "cycle"]) {
          const cands = play.filter((n) => roles[n] === role);
          if (cands.length) return [cands[0], lane, "leak"];
        }
      }
      return [null, 0, null];
    },
  };

  // ----------------------------------------------------------------- guard.py
  const TAKE_OVER = new Set(["finish", "defend"]), WHEN_IDLE = new Set(["trade", "counterpush", "punish", "leak"]);
  // The fly decides; coach rules step in where it is weak. Returns [card or null, lane, who].
  function guard(view, card, lane) {
    const [cCard, cLane, reason] = Coach.plan(view);
    if (TAKE_OVER.has(reason) && cCard !== null) return [cCard, cLane, `coach:${reason}`];
    if (card === null) {
      if (WHEN_IDLE.has(reason) && cCard !== null) return [cCard, cLane, `coach:${reason}`];
      return [null, lane, "fly"];
    }
    const c = view.db.cards[card];
    if (c.type === "spell" && !c.summons.length && card !== cCard && spellSpot(view, card)[0] < c.elixir) return [null, lane, "veto"];
    if (![0, 1].some((ln) => view.threats(ln).length)) {
      if (c.role === "building") return [null, lane, "veto"];
      const win = view.p.hand.find((n) => view.db.cards[n].role === "win_condition");
      if (win !== undefined && card !== win && c.role !== "win_condition") {
        if (view.elixir >= view.db.cards[win].elixir) return [win, cCard !== null ? cLane : view.weakLane(), "coach:win_condition"];
        return [null, lane, "save"];
      }
    }
    if (cCard !== null) lane = cLane;
    return [card, lane, "fly"];
  }

  function autoAbility(sim, player) {
    for (const u of sim.units) {
      if (u.alive && u.owner === player && u.spec && u.spec.ability && u.deploy_left <= 0) {
        const ab = u.spec.ability;
        if (u.ability_cd > 0 || sim.players[player].elixir < ab.cost + 1) return false;
        const near = sim.units.some((o) => o.alive && o.owner !== player && !o.timed && dist(o.x, o.y, u.x, u.y) <= 5.5);
        if (near || (u.target && u.target.building && dist(u.x, u.y, u.target.x, u.target.y) < 4)) return sim.useAbility(player);
      }
    }
    return false;
  }

  // ----------------------------------------------------------------- env.py
  const N_ROLES = ROLES.length;
  function channelNames() {
    const n = [[0, 2], [2, 3], [3, 4], [4, 5], [5, 7], [7, 9], [9, 10]].map(([a, b]) => `elixir ${a}-${b}`);
    for (const r of ROLES) n.push(`${r} ready`);
    for (const r of ROLES) n.push(`${r} in hand`);
    for (const s of ["left", "right"]) {
      n.push(`${s}: no threat`, `${s}: small threat`, `${s}: medium threat`, `${s}: big threat`);
      for (const k of ["building-targeter", "air", "swarm", "splash", "ranged", "tank killer", "building"]) n.push(`${s}: enemy ${k}`);
      n.push(`${s}: threat at my tower`, `${s}: I'm defending`);
      n.push(`${s}: no push`, `${s}: small push`, `${s}: strong push`, `${s}: my win condition pushing`);
      n.push(`${s}: my tower low`, `${s}: my tower down`, `${s}: enemy tower low`, `${s}: enemy tower down`);
      n.push(`${s}: enemy tank building at back`);
    }
    n.push("double/triple elixir", "overtime", "enemy low on elixir", "enemy high on elixir", "big spell has value",
           "small spell has value", "my champion on field", "anti-air in hand");
    return n;
  }
  const CHANNELS = channelNames();

  function features(view) {
    const db = view.db, f = new Float32Array(CHANNELS.length);
    let i = 0;
    f[i + searchRight([2, 3, 4, 5, 7, 9], view.elixir)] = 1; i += 7;
    const playable = new Set(view.playable());
    for (const n of view.hand()) { const r = ROLES.indexOf(db.cards[n].role); f[i + N_ROLES + r] = 1; if (playable.has(n)) f[i + r] = 1; }
    i += 2 * N_ROLES;
    for (const lane of [0, 1]) {
      const th = view.threats(lane), prof = view.profile(th);
      f[i + searchRight([1, 700, 2000], prof.hp)] = 1;
      ["building_targeter", "air", "swarm", "splash", "ranged", "tank_killer", "building"].forEach((k, j) => { f[i + 4 + j] = prof[k] ? 1 : 0; });
      f[i + 11] = th.some((u) => view.fy(u.y) < 10) ? 1 : 0;
      f[i + 12] = view.defenders(lane).length ? 1 : 0;
      const push = view.pushers(lane);
      f[i + 13 + searchRight([1, 1000], push.reduce((s, u) => s + u.hp, 0))] = 1;
      f[i + 16] = push.some((u) => u.card && db.cards[u.card].role === "win_condition") ? 1 : 0;
      const mine = view.towerFrac(true, lane), theirs = view.towerFrac(false, lane);
      f[i + 17] = mine > 0 && mine < 0.5 ? 1 : 0; f[i + 18] = mine === 0 ? 1 : 0;
      f[i + 19] = theirs > 0 && theirs < 0.5 ? 1 : 0; f[i + 20] = theirs === 0 ? 1 : 0;
      f[i + 21] = view.enemyBackTank() === lane ? 1 : 0;
      i += 22;
    }
    f[i] = view.sim.elixirMultiplier > 1 ? 1 : 0;
    f[i + 1] = view.sim.overtime ? 1 : 0;
    f[i + 2] = view.enemyElixir() <= 3 ? 1 : 0;
    f[i + 3] = view.enemyElixir() >= 8 ? 1 : 0;
    const big = view.hand().filter((n) => db.cards[n].role === "big_spell"), small = view.hand().filter((n) => db.cards[n].role === "small_spell");
    f[i + 4] = big.some((n) => spellSpot(view, n)[0] >= db.cards[n].elixir + 0.5) ? 1 : 0;
    f[i + 5] = small.some((n) => spellSpot(view, n)[0] >= db.cards[n].elixir) ? 1 : 0;
    f[i + 6] = view.sim.championAlive(view.me) ? 1 : 0;
    f[i + 7] = view.hand().some((n) => hitsAir(db.cards[n]) && db.cards[n].type !== "spell") ? 1 : 0;
    return f;
  }

  function roleMask(view) {
    const m = new Array(N_ROLES + 1).fill(false); m[0] = true;
    for (const n of view.playable()) m[1 + ROLES.indexOf(view.db.cards[n].role)] = true;
    return m;
  }

  function pickCard(view, role) {
    const db = view.db, cands = view.playable().filter((n) => db.cards[n].role === role);
    if (!cands.length) return null;
    if (cands.length === 1) return cands[0];
    const [suggested] = Coach.suggest(view);
    if (cands.includes(suggested)) return suggested;
    const threats = [0, 1].flatMap((l) => view.threats(l));
    const prof = view.profile(threats);
    const fit = cands.filter((n) => Coach.cardFit(view, n, prof));
    return minBy(fit.length ? fit : cands, (n) => db.cards[n].elixir);
  }

  return { WIDTH, HEIGHT, RIVER_LO, RIVER_HI, LANE_X, DT, TOWER, ROLES, CHANNELS, REGULATION, OVERTIME_END, DOUBLE_AT, TRIPLE_AT,
    buildDB, Sim, View, place, spellSpot, Coach, guard, PLACE, COACH, autoAbility, features, roleMask, pickCard, hitsAir, frameY, absY, laneOf };
})();

if (typeof module !== "undefined") module.exports = RoyaleCore;
