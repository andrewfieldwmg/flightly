import { readFileSync, writeFileSync } from "node:fs";
import { preferenceFeatures } from "../src/lib/search/preference-model.ts";
const [input, output] = process.argv.slice(2);
const rows = JSON.parse(readFileSync(input, "utf8"));
writeFileSync(output, JSON.stringify(rows.map((row) => ({ ...row, chosen_features: preferenceFeatures(row.chosen), rejected_features: preferenceFeatures(row.rejected) }))));
