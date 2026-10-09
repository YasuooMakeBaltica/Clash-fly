// Runs a scenario through the JS port and prints results as JSON (used by tests/test_web_port.py).
// usage: node web/parity_check.js scenario.json [brain.json]
const fs = require("fs");
const C = require("./clash_core.js");
const sc = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = { snapshots: [], votes: [] };
const sim = new C.Sim(1, false);  // fixed update order so results are comparable
sim.players.forEach((p, i) => { p.hand = sc.hands[i].slice(); p.queue = sc.queues[i].slice(); p.elixir = sc.elixir[i]; });
let tick = 0;
for (const [atTick, player, card, x, y] of sc.plays) {
  while (tick < atTick) { sim.step(); tick++; }
  sim.play(player, card, x, y);
}
for (const at of sc.snapshot_ticks) {
  while (tick < at) { sim.step(); tick++; }
  out.snapshots.push({
    tick,
    units: sim.units.map((u) => [u.uid, u.owner, u.idx, u.x, u.y, u.hp]),
    towers: sim.towers.map((t) => t.hp),
    elixir: sim.players.map((p) => p.elixir),
    features: [0, 1].map((pl) => Array.from(C.features(new C.View(sim, pl)))),
    place: [0, 1].map((pl) => [0, 1, 2, 3, 4, 5, 6, 7].map((c) => [0, 1].map((ln) => C.place(new C.View(sim, pl), c, ln)))),
  });
}
if (process.argv[3]) {
  const d = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
  d.noise = 0; d.mbon_noise = 0;
  const brain = new C.Brain(d);
  for (const f of sc.brain_features) {
    const r = brain.decide(Float32Array.from(f), new Array(9).fill(true));
    out.votes.push([r.cardVotes, r.laneVotes]);
  }
}
console.log(JSON.stringify(out));
