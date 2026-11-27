// The demo's ranking (q8 model, e5 prefixes, mean pooling) on demo/examples.json.
import { env, pipeline } from "@huggingface/transformers";
import { readFileSync } from "node:fs";
env.localModelPath = new URL("../../models/release/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
env.allowRemoteModels = false;
const ex = JSON.parse(readFileSync(new URL("../examples.json", import.meta.url), "utf8"));
const docs = ex.docs.map((d) => d[1]);
const embed = await pipeline("feature-extraction", "indic-e5-small", { dtype: "q8" });
const enc = async (t) => (await embed(t, { pooling: "mean", normalize: true })).tolist();
const D = await enc(docs.map((d) => "passage: " + d));
const Q = await enc(ex.queries.map(([q]) => "query: " + q));
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
let hits = 0;
ex.queries.forEach(([q, gold], i) => {
  const best = docs.map((d, j) => [d, dot(Q[i], D[j])]).sort((a, b) => b[1] - a[1])[0][0];
  if (best === gold) hits++;
});
console.log(`q8 in transformers.js: top-1 ${hits}/${ex.queries.length}`);
