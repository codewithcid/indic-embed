import { env, pipeline } from "@huggingface/transformers";
env.cacheDir = process.argv[2];
const embed = await pipeline("feature-extraction", "cidcodes/indic-e5-small", { dtype: "q8" });
const out = (await embed(["query: mujhe kal office jana hai", "passage: मुझे कल ऑफिस जाना है", "passage: আজ খুব বৃষ্টি হচ্ছে"], { pooling: "mean", normalize: true })).tolist();
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
console.log("transformers.js from the Hub:", dot(out[0], out[1]).toFixed(3), dot(out[0], out[2]).toFixed(3));
