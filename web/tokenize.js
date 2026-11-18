// Port of tiny_voca.text (ipa_pieces + tokenize): stressed IPA (RenikudPlus) -> token ids and stress.
const SPECIALS = ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>"];
const PUNCT = [",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…"];
const PHONES = ["p","b","t","d","k","ɡ","f","v","s","z","ʃ","ʒ","χ","h","ʁ","m","n","l","j","w","ʔ","ts","tʃ","dʒ","a","e","i","o","u"];
const SYMBOLS = [...SPECIALS, ...PUNCT, ...PHONES];
const ID = Object.fromEntries(SYMBOLS.map((s, i) => [s, i]));
const PHONE_SET = new Set(PHONES), VOWELS = new Set(["a","e","i","o","u"]), PUNCT_SET = new Set(PUNCT);
const STRESS = { "ˈ": 1, "ˌ": 2 }, AFFRICATES = ["ts", "tʃ", "dʒ"];
const FALLBACK = { x: ["χ"], r: ["ʁ"], g: ["ɡ"], "ʤ": ["dʒ"], "ʧ": ["tʃ"], "ʦ": ["ts"], c: ["ts"] };
const QUOTES = { "‘":"'", "’":"'", "‚":"'", "‛":"'", "`":"'", "´":"'", "׳":"'", "“":'"', "”":'"', "„":'"', "‟":'"', "«":'"', "»":'"', "״":'"', "–":"—", "―":"—", "[":"(", "{":"(", "]":")", "}":")" };

function normalizeText(text) {
  text = [...text.normalize("NFC")].map((c) => QUOTES[c] ?? c).join("");
  text = text.replace(/\.{3,}/g, "…").replace(/-{2,}|\s-\s/g, "—");
  return text.replace(/\s+/g, " ").trim();
}

function splitWord(word) {
  const units = []; let pending = 0, i = 0;
  const chars = [...word];
  while (i < chars.length) {
    const c = chars[i];
    if (c in STRESS) { pending = STRESS[c]; i++; continue; }
    if (/[\p{Mn}\p{Cf}]/u.test(c) || "ː-‐־".includes(c)) { i++; continue; }
    const rest = chars.slice(i).join("");
    const phone = AFFRICATES.find((a) => rest.startsWith(a)) ?? c;
    i += [...phone].length;
    const mapped = PHONE_SET.has(phone) ? [phone] : FALLBACK[phone];
    if (!mapped) { units.push([phone, 0]); continue; }
    for (const item of mapped) {
      let level = 0;
      if (VOWELS.has(item) && pending) { level = pending; pending = 0; }
      units.push([item, level]);
    }
  }
  return units;
}

function ipaPieces(ipa) {
  const pieces = []; let words = [];
  for (const raw of normalizeText(ipa).split(" ")) {
    const chars = [...raw]; let start = 0;
    chars.forEach((c, i) => {
      if (!PUNCT_SET.has(c)) return;
      const before = chars.slice(start, i).join("");
      if (before) words.push(splitWord(before));
      if (words.length) { pieces.push([false, words]); words = []; }
      pieces.push([true, c]);
      start = i + 1;
    });
    const tail = chars.slice(start).join("");
    if (tail) words.push(splitWord(tail));
  }
  if (words.length) pieces.push([false, words]);
  return pieces;
}

function tokenizeIpa(ipa) {
  const symbols = ["<bos>"], stress = [0], unknown = [];
  for (const [isPunct, value] of ipaPieces(ipa)) {
    if (isPunct) { if (symbols.at(-1) !== value) { symbols.push(value); stress.push(0); } continue; }
    for (const word of value) {
      if (!word.length) continue;
      const last = symbols.at(-1);
      if (PHONE_SET.has(last) || last === "<unk>") { symbols.push("<sp>"); stress.push(0); }
      for (let [phone, level] of word) {
        if (!PHONE_SET.has(phone)) { unknown.push(phone); phone = "<unk>"; }
        symbols.push(phone); stress.push(level);
      }
    }
  }
  symbols.push("<eos>"); stress.push(0);
  return { symbols, ids: symbols.map((s) => ID[s]), stress, unknown,
           speakable: symbols.some((s) => PHONE_SET.has(s) || s === "<unk>") };
}
if (typeof module !== "undefined") module.exports = { tokenizeIpa };
