"""
Livestock Sentinel AI - understanding farmer SMS reports (Swahili, English, mixed)

Two engines:
  * AI engine  : a Claude language model extracts structured data (used when an API key is set).
  * Fallback   : a transparent keyword matcher, so the demo still works without a key.
Every report gets a confidence level. Unclear reports go to a human review queue.
The system never diagnoses: it flags "possible FMD signs" for a vet to verify.
"""
import json
import re
import requests

SYMPTOMS = {
    "drooling": ["mate", "udenda", "drool", "saliva", "slobber", "froth", "povu"],
    "mouth sores or blisters": ["vidonda mdomoni", "vidonda kwa mdomo", "mdomo", "mouth sore",
                                "sores in the mouth", "sores on the mouth", "ulimi", "tongue"],
    "foot sores or lameness": ["chechemea", "kuchechemea", "kwato", "miguu", "mguu", "lame", "limp", "hoof", "hooves", "feet", "foot"],
    "fever": ["homa", "joto", "fever", "hot body"],
    "not eating": ["hawali", "hakuli", "hawakuli", "kataa kula", "not eating", "refuse to eat", "won't eat", "stopped eating"],
    "milk drop": ["maziwa yamepungua", "maziwa kidogo", "maziwa", "milk"],
    "death": ["amekufa", "wamekufa", "kufa", "died", "dead", "death"],
}
FMD_CORE = {"drooling", "mouth sores or blisters", "foot sores or lameness"}
SPECIES = {"cattle": ["ng'ombe", "ngombe", "ng`ombe", "cow", "cattle", "bull", "heifer", "calf", "ndama", "fahali"],
           "goats": ["mbuzi", "goat"], "sheep": ["kondoo", "sheep"], "camels": ["ngamia", "camel"]}
SW_NUM = {"moja": 1, "mbili": 2, "tatu": 3, "nne": 4, "tano": 5, "sita": 6, "saba": 7, "nane": 8, "tisa": 9, "kumi": 10,
          "wawili": 2, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
TOWNS = {"kitengela": "Kajiado East", "isinya": "Kajiado East", "ngong": "Kajiado North", "kiserian": "Kajiado North",
         "ilbissil": "Kajiado Central", "magadi": "Kajiado West", "nanyuki": "Laikipia East", "rumuruti": "Laikipia West",
         "doldol": "Laikipia North", "kilgoris": "Transmara West", "suswa": "Narok East", "ololulunga": "Narok South",
         "mai mahiu": "Naivasha", "gilgil": "Gilgil", "molo": "Kuresoi North", "njoro": "Njoro", "modogashe": "Lagdera",
         "garbatulla": "Isiolo South", "merti": "Isiolo North", "garissa": "Garissa Township", "loitokitok": "Loitokitok"}


def _reply(lang, needs_review, fmd):
    if lang == "sw":
        base = "Asante, ripoti yako imepokelewa."
        tail = (" Afisa wa mifugo amejulishwa na atawasiliana nawe. Tafadhali usiuze wala kuhamisha ng'ombe hawa kwa sasa."
                if fmd else " Afisa wa mifugo ataipitia.")
        if needs_review:
            tail = " Afisa wa mifugo ataipitia na anaweza kukupigia simu kwa maelezo zaidi."
        return base + tail
    base = "Thank you, your report has been received."
    tail = (" A livestock officer has been notified and will contact you. Please do not sell or move these animals for now."
            if fmd else " A livestock officer will review it.")
    if needs_review:
        tail = " A livestock officer will review it and may call you for more details."
    return base + tail


def review_rules(r):
    reasons = []
    if r["confidence"] < 0.7: reasons.append("low confidence in understanding the message")
    if not r["location"]: reasons.append("location unclear")
    if r["species"] not in ("cattle", "unknown"): reasons.append(f"species is {r['species']} (model covers cattle)")
    if not r["symptoms"]: reasons.append("no recognised symptoms")
    r["needs_review"] = bool(reasons)
    r["review_reasons"] = reasons
    r["reply"] = _reply(r["language"], r["needs_review"], r["fmd_signs"] == "strong")
    return r


def parse_fallback(text, registered_area, area_names):
    t = text.lower()
    found = [s for s, words in SYMPTOMS.items() if any(w in t for w in words)]
    if any(w in t for w in ["malengelenge", "blister"]):   # blisters: mouth or feet, depending on context
        where = "foot sores or lameness" if any(w in t for w in ["kwato", "miguu", "hoof", "feet", "foot"]) else "mouth sores or blisters"
        if where not in found: found.append(where)
    if "milk drop" in found and not re.search(r"(pungua|kidogo|drop|less|reduc|low)", t):
        found.remove("milk drop")
    species = next((s for s, w in SPECIES.items() if any(x in t for x in w)), "unknown")
    n = None
    m = re.search(r"\b(\d{1,3})\b", t)
    if m: n = int(m.group(1))
    else:
        for w, v in SW_NUM.items():
            if re.search(rf"\b\w*{w}\b", t): n = v; break
    loc = next((a for a in area_names if a.lower() in t), None) or next((v for k, v in TOWNS.items() if k in t), None)
    loc = loc or registered_area
    sw_words = ["ng'ombe", "ngombe", "wangu", "wana", "mate", "na ", "yangu", "tafadhali", "wamekufa", "homa"]
    lang = "sw" if sum(w in t for w in sw_words) >= 2 else "en"
    core = len(FMD_CORE & set(found))
    fmd = "strong" if core >= 2 else "possible" if core == 1 else "none"
    conf = 0.35 + 0.15 * min(len(found), 3) + (0.1 if species != "unknown" else 0) + (0.1 if n else 0)
    r = dict(engine="keyword fallback", language=lang, species=species, n_animals=n, symptoms=found,
             fmd_signs=fmd, location=loc, confidence=round(min(conf, 0.95), 2),
             summary=f"{n or 'Some'} {species if species != 'unknown' else 'animals'} with " +
                     (", ".join(found) if found else "no clear symptoms"))
    return review_rules(r)


PROMPT = """You extract structured data from farmer SMS reports about sick livestock in Kenya.
Messages may be in Swahili, English, Sheng or a mix, with spelling errors.
Registered location of the sender: {reg}. Known sub-counties: {areas}.
Return ONLY a JSON object, no other text, with these keys:
"language": "sw" or "en" (main language of the message),
"species": "cattle", "goats", "sheep", "camels" or "unknown",
"n_animals": integer or null,
"symptoms": list using only these labels: {labels},
"fmd_signs": "strong" if two or more of drooling / mouth sores or blisters / foot sores or lameness are present, "possible" if one, else "none",
"location": the sub-county from the known list if the message names it or a town inside it, otherwise the registered location,
"confidence": number 0-1 for how sure you are you understood the message,
"summary": one short English sentence.
Do not diagnose disease. Message: \"\"\"{msg}\"\"\""""


def parse_ai(text, registered_area, area_names, api_key, model="claude-haiku-4-5-20251001"):
    resp = requests.post("https://api.anthropic.com/v1/messages", timeout=30, headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": PROMPT.format(
            reg=registered_area, areas=", ".join(area_names), labels=", ".join(SYMPTOMS), msg=text)}]})
    resp.raise_for_status()
    raw = "".join(b.get("text", "") for b in resp.json()["content"])
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    r = json.loads(raw)
    r["symptoms"] = [s for s in r.get("symptoms", []) if s in SYMPTOMS]
    r["location"] = r.get("location") if r.get("location") in area_names else registered_area
    r["confidence"] = float(r.get("confidence", 0.5))
    r["species"] = r.get("species", "unknown")
    r["language"] = r.get("language", "en")
    r["fmd_signs"] = r.get("fmd_signs", "none")
    r["engine"] = "AI (Claude)"
    return review_rules(r)


def parse(text, registered_area, area_names, api_key=None):
    if api_key:
        try:
            return parse_ai(text, registered_area, area_names, api_key)
        except Exception as e:  # fall back rather than fail in front of a user
            r = parse_fallback(text, registered_area, area_names)
            r["engine"] = f"keyword fallback (AI unavailable: {type(e).__name__})"
            return r
    return parse_fallback(text, registered_area, area_names)


SAMPLES = [
    ("+2547••••312", "Laikipia East", "Ng'ombe wangu watatu wanatoa mate mengi na wanachechemea tangu jana. Nanyuki area."),
    ("+2547••••845", "Narok East", "Habari daktari, ng'ombe 5 wana vidonda mdomoni na hawali. Tuko karibu na soko la Suswa."),
    ("+2547••••027", "Kajiado Central", "My cows are drooling and 2 are limping, blisters on the tongue. Milk has dropped a lot."),
    ("+2547••••590", "Garissa Township", "Ngamia wangu mmoja ana homa"),
    ("+2547••••118", "Naivasha", "ng'ombe yangu moja amekataa kula leo"),
    ("+2547••••764", "Kuresoi North", "Cows sick pls come"),
    ("+2547••••433", "Narok South", "Ndama wawili wana malengelenge kwa kwato na mate yanatoka sana, maziwa yamepungua kwa wale wakubwa"),
]
