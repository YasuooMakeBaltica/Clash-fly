// Runs a scripted battle through the JS port and prints the state as JSON (tests/test_web_royale.py).
// usage: node web/royale_parity.js scenario.json [brain.json]
const fs = require("fs");
global.ClashCore = require("./clash_core.js");
const R = require("./royale_core.js");
const sc = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const db = R.buildDB(JSON.parse(fs.readFileSync(__dirname + "/../flybrain/envs/royale/data/cards.json", "utf8")));
const sim = new R.Sim(db, sc.decks, 1, false);
sim.players.forEach((p, i) => { p.hand = sc.hands[i].slice(); p.queue = sc.queues[i].slice(); p.elixir = 10; });
sim.tower(1, "princess", 0).hp -= 1;   // avoid random tie-breaks in weak_lane
let tick = 0;
const out = { snapshots: [], votes: [] };
const ev = sc.events.slice();
for (const at of sc.snapshot_ticks) {
  while (tick < at) {
    while (ev.length && ev[0][0] === tick) {
      const [, kind, player, name, x, y] = ev.shift();
      if (kind === "deploy") sim._deploy(player, db.cards[name], x, y);
      else if (kind === "ability") { sim.players[player].elixir = 10; sim.useAbility(player); }
    }
    sim.step(); tick++;
  }
  const views = [0, 1].map((p) => new R.View(sim, p));
  out.snapshots.push({
    tick,
    units: sim.units.map((u) => [u.uid, u.spec ? u.spec.name : u.tower, u.owner, u.x, u.y, u.hp, u.shield]),
    projectiles: sim.projectiles.length, effects: sim.effects.length,
    elixir: sim.players.map((p) => p.elixir),
    features: views.map((v) => Array.from(R.features(v))),
    masks: views.map((v) => R.roleMask(v)),
    coach: views.map((v) => R.Coach.suggest(v)),
    place: views.map((v) => v.hand().map((n) => [0, 1].map((l) => R.place(v, n, l)))),
    picks: views.map((v) => R.ROLES.map((r) => R.pickCard(v, r))),
  });
}
if (process.argv[3]) {
  const d = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
  d.noise = 0; d.mbon_noise = 0;
  const brain = new ClashCore.Brain(d);
  for (const f of sc.brain_features) { const r = brain.decide(Float32Array.from(f), new Array(16).fill(true)); out.votes.push([r.cardVotes, r.laneVotes]); }
}
console.log(JSON.stringify(out));
