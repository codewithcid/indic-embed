// Load the release folder with transformers.js (as a browser would) and compare with sentence-transformers.
import { env, pipeline } from "@huggingface/transformers";
import { readFileSync } from "node:fs";

env.localModelPath = new URL("../../models/release/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
env.allowRemoteModels = false;
const ref = JSON.parse(readFileSync(new URL("./ref.json", import.meta.url), "utf8"));
for (const dtype of ["fp32", "q8"]) {
  const embed = await pipeline("feature-extraction", "indic-e5-small", { dtype });
  const out = (await embed(ref.texts, { pooling: "mean", normalize: true })).tolist();
  const cos = out.map((v, i) => v.reduce((s, x, j) => s + x * ref.emb[i][j], 0));
  console.log(dtype, "cosine to sentence-transformers:", cos.map((c) => c.toFixed(4)).join(" "));
}
