"""The Chief Retention Officer: data payload and HTML for the dashboard's interactive story game.

The game itself runs in the browser (churn/game_assets/game.html + engine.js), so every click
reacts instantly. Python supplies the *real* data it plays with:

- personas: real held-out test customers (one seeded random draw per archetype and round, never
  chosen by score or outcome), with every model's churn probability and their real outcome;
- each model's cross-validated ROC-AUC and its CV-optimised (max-F1) threshold;
- at-scale statistics on all test customers for every model and threshold on a 0.01 grid
  (true/false positives and negatives, and the customer value behind them).

Money values (customer value, incentive costs and success rates) are the game assumptions in
churn.game.Economics; the game shows them as such.
"""

import json
from pathlib import Path

import numpy as np

from churn import config
from churn.game import PERSONAS, Economics, select_personas

ASSETS = Path(__file__).resolve().parent / "game_assets"
N_ROUNDS = 5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)

INCENTIVE_META = {  # Economics key -> (id, icon, short name, blurb)
    "📞 Personal call": ("call", "📞", "Personal call", "A relationship manager calls to listen and fix issues."),
    "💳 Fee waiver": ("waiver", "💳", "Fee waiver", "Twelve months of account fees waived."),
    "👑 VIP package": ("vip", "👑", "VIP package", "Premium rates, a dedicated advisor and lounge access."),
}
INCENTIVE_META_HE = {  # id -> (short name, blurb)
    "call": ("שיחה אישית", "מנהל/ת קשרי לקוחות מתקשר/ת כדי להקשיב ולפתור בעיות."),
    "waiver": ("ויתור על עמלות", "שנים-עשר חודשים ללא עמלות ניהול חשבון."),
    "vip": ("חבילת VIP", "ריביות מועדפות, יועץ/ת אישי/ת וגישה לטרקלין."),
}

# Persona text in Hebrew. Each persona's own first-person lines use their grammatical gender
# (David, Michael and Lukas: masculine; Sarah and Emma: feminine).
PERSONA_HE = {
    "david": ("דוד", "בעל עסק, לקוח בעל ערך גבוה",
              "מנהל חברה צומחת ושומר יתרה גבוהה בבנק. עסוק מאוד, כמעט לא משתמש באפליקציה."),
    "sarah": ("שרה", "הסטודנטית המנותקת", "פתחה חשבון באוניברסיטה. לא התחברה כבר חודשים."),
    "michael": ("מיכאל", "המשקיע הוותיק", "בפנסיה, עם חסכונות משמעותיים. מעריך שירות אישי ויציבות."),
    "emma": ("אמה", "לקוחה משפחתית ותיקה", "לקוחה פעילה וותיקה עם שני מוצרים: חשבון עובר ושב ותוכנית חיסכון."),
    "lukas": ("לוקאס", "הלקוח עמוס המוצרים", "קיבל שלושה מוצרים או יותר בגרמניה. מתחיל לתהות על העמלות."),
}

PERSONA_LINES_HE = {
    "david": {
        "idle": "אני מנהל חברה. בקצרה, בבקשה.",
        "click": "חשבונות העסק שלי כאן. בינתיים.",
        "call": "שיחה מהבנק? בסדר, חמש דקות.",
        "waiver": "פחות עמלות על חשבונות העסק... זו התחלה.",
        "vip": "ככה מתייחסים ללקוח עסקי.",
        "stay_targeted": "תודה על ההטבות, אבל לא התכוונתי לזוז לשום מקום.",
        "stay": "עסקים כרגיל. אני נשאר.",
        "left_targeted": "עמדתי להעביר את החשבונות. ההצעה שלכם גרמה לי לחשוב שוב.",
        "left": "העברתי את העסק למתחרה. אף אחד לא התקשר.",
    },
    "sarah": {
        "idle": "אני בכלל עוד משתמשת בחשבון הזה?",
        "click": "פתחתי אותו באוניברסיטה. לא נכנסתי כבר המון זמן.",
        "call": "מישהו באמת התקשר אליי? זה... נחמד.",
        "waiver": "בלי עמלות? אוקיי, אולי אשאיר אותו.",
        "vip": "VIP? בשבילי? אני סטודנטית...",
        "stay_targeted": "הטבות זה נחמד, אבל ממילא לא התכוונתי לטרוח ולעבור.",
        "stay": "כנראה פשוט אשאיר אותו פתוח. סגירה זה יותר מדי טרחה.",
        "left_targeted": "עמדתי לעבור לבנק דיגיטלי. תפסתם את תשומת הלב שלי.",
        "left": "עברתי לבנק שכולו אפליקציה. אפילו לא שמתם לב שהלכתי.",
    },
    "michael": {
        "idle": "בגילי, אני מצפה לשירות טוב.",
        "click": "אני מעריך יציבות. ובן אדם אמיתי על הקו.",
        "call": "סוף סוף, אדם אמיתי. תודה שהתקשרתם.",
        "waiver": "חיסכון בעמלות תמיד מתקבל בברכה בגילי.",
        "vip": "יועץ אישי? עכשיו אפשר לדבר על תיק ההשקעות שלי.",
        "stay_targeted": "נחמד מצדכם, אבל הייתי מרוצה לגמרי גם ככה.",
        "stay": "אני נשאר במקום. אני לא אוהב שינויים.",
        "left_targeted": "האחיין שלי רצה שאעביר את החסכונות. שיניתם את דעתי.",
        "left": "העברתי את החסכונות למקום אחר. נראה שלאף אחד לא היה אכפת.",
    },
    "emma": {
        "idle": "שני ילדים, שני חשבונות, אפס זמן פנוי.",
        "click": "אנחנו אתכם כבר שנים. הכול בסדר.",
        "call": "יש בעיה בחשבון שלי?",
        "waiver": "פחות עמלות זה יותר כסף לילדים. תודה!",
        "vip": "זה נדיב. לא ציפיתי לשום דבר.",
        "stay_targeted": "מחווה מקסימה, אבל ממילא התכוונו להישאר.",
        "stay": "טוב לנו כאן. אין סיבה לעבור.",
        "left_targeted": "בדקנו משכנתא בבנק אחר. בואו נדבר.",
        "left": "לקחנו את המשכנתא ואת החסכונות לבנק אחר.",
    },
    "lukas": {
        "idle": "למה אני משלם על שלושה מוצרים?",
        "click": "היועץ שלכם מכר לי מוצרים שאני לא משתמש בהם.",
        "call": "יופי, יש לי רשימת תלונות מוכנה.",
        "waiver": "ויתור על העמלות זה המינימום.",
        "vip": "יחס של VIP? סוף סוף לוקחים אותי ברצינות.",
        "stay_targeted": "בסדר, אני נשאר. רק רטנתי.",
        "stay": "עדיין מעצבן, אבל אני לא עוזב.",
        "left_targeted": "טפסי הסגירה כבר היו מלאים. נראה אם תעמדו בהבטחות.",
        "left": "סגרתי את שלושת המוצרים. הייתם צריכים להקשיב קודם.",
    },
}

PERSONA_LINES = {
    "david": {
        "idle": "I run a company. Make it quick.",
        "click": "My business accounts are here. For now.",
        "call": "A call from the bank? Fine, five minutes.",
        "waiver": "Lower fees on business accounts... that is a start.",
        "vip": "Now that's how you treat a business client.",
        "stay_targeted": "Thanks for the perks, but I wasn't going anywhere.",
        "stay": "Business as usual. I'm staying.",
        "left_targeted": "I was about to move my accounts. Your offer made me think twice.",
        "left": "I've moved my business to a competitor. Nobody called.",
    },
    "sarah": {
        "idle": "Do I even still use this account?",
        "click": "I opened it at university. Haven't logged in for ages.",
        "call": "Oh, someone actually called me? That's... nice.",
        "waiver": "No fees? OK, maybe I'll keep it.",
        "vip": "VIP? For me? I'm a student...",
        "stay_targeted": "Free stuff is cool, but I was never going to bother switching.",
        "stay": "I'll probably just leave it open. Too much hassle to close.",
        "left_targeted": "I was about to switch to a neobank. You got my attention.",
        "left": "Switched to an app-only bank. Didn't even notice you were gone.",
    },
    "michael": {
        "idle": "At my age, I expect good service.",
        "click": "I value stability. And a human being on the phone.",
        "call": "Finally, a real person. Thank you for calling.",
        "waiver": "Saving on fees is always welcome at my age.",
        "vip": "A dedicated advisor? Now we can talk about my portfolio.",
        "stay_targeted": "Kind of you, but I was perfectly happy already.",
        "stay": "I'm staying put. I don't like change.",
        "left_targeted": "My nephew wanted me to move my savings. You changed my mind.",
        "left": "I moved my savings elsewhere. Nobody seemed to care.",
    },
    "emma": {
        "idle": "Two kids, two accounts, zero free time.",
        "click": "We've been with you for years. Everything is fine.",
        "call": "Is something wrong with my account?",
        "waiver": "Fewer fees means more for the kids. Thanks!",
        "vip": "That's generous. I didn't expect anything.",
        "stay_targeted": "Lovely gesture, but we were staying anyway.",
        "stay": "We're happy here. No reason to move.",
        "left_targeted": "We were looking at a mortgage elsewhere. Let's talk.",
        "left": "We took our mortgage and our savings to another bank.",
    },
    "lukas": {
        "idle": "Why am I paying for three products?",
        "click": "Your adviser sold me products I never use.",
        "call": "Good, I have a list of complaints ready.",
        "waiver": "Waiving the fees is the least you can do.",
        "vip": "VIP treatment? You're finally taking me seriously.",
        "stay_targeted": "Fine, I'll stay. I was only grumbling.",
        "stay": "Still annoyed, but I'm not leaving.",
        "left_targeted": "I had the closure forms filled in. Let's see if you deliver.",
        "left": "Closed all three products. Should have listened earlier.",
    },
}


def _round_customers(analysis, econ: Economics, round_no: int, lang: str = "en") -> list[dict]:
    P = select_personas(analysis.X_test, round_no)
    positions = [analysis.X_test.index.get_loc(i) for i in P.index]
    values = econ.customer_value(P).to_numpy()
    churned = analysis.y_test.loc[P.index].to_numpy().astype(bool)
    customers = []
    for k, (persona, (_, row)) in enumerate(zip(PERSONAS, P.iterrows())):
        name, title, story = PERSONA_HE[persona.key] if lang == "he" else (persona.name, persona.title, persona.story)
        lines = (PERSONA_LINES_HE if lang == "he" else PERSONA_LINES)[persona.key]
        customers.append({
            "key": persona.key, "name": name, "avatar": persona.avatar, "title": title,
            "story": story, "lines": lines,
            "age": int(row["Age"]), "geo": str(row["Geography"]), "balance": float(row["Balance"]),
            "products": int(row["NumOfProducts"]), "active": bool(row["IsActiveMember"]), "tenure": int(row["Tenure"]),
            "value": float(values[k]), "churned": bool(churned[k]),
            "p": {name: float(analysis.test_scores[name][positions[k]]) for name in analysis.models},
        })
    return customers


def scale_grid(scores, churned, values, extra: tuple = ()) -> dict:
    """Confusion counts and the customer value behind them for every threshold on THRESHOLD_GRID.

    extra adds exact thresholds (e.g. the model's CV optimum, 0.2776) so the game reports the same
    Precision/Recall at the optimum as the rest of the dashboard, instead of the nearest 0.01 step.
    """
    scores, churned, values = np.asarray(scores), np.asarray(churned, bool), np.asarray(values, float)
    grid = np.unique(np.concatenate([THRESHOLD_GRID, np.asarray(extra, float)]))
    flagged = scores[None, :] >= grid[:, None]
    tp, fp = flagged & churned, flagged & ~churned
    fn, tn = ~flagged & churned, ~flagged & ~churned
    return {
        "thresholds": grid.tolist(),
        "tp": tp.sum(1).tolist(), "fp": fp.sum(1).tolist(), "fn": fn.sum(1).tolist(), "tn": tn.sum(1).tolist(),
        "v_tp": (tp * values).sum(1).round(2).tolist(), "v_fn": (fn * values).sum(1).round(2).tolist(),
    }


def build_payload(analysis, best: str, colors: dict, econ: Economics | None = None, budget: float = 10_000,
                  n_rounds: int = N_ROUNDS, weighting: str = "None", lang: str = "en") -> dict:
    """The game's data. lang ("en" or "he") selects the persona and incentive text; numbers never change."""
    econ = econ or Economics()
    cv = analysis.cv_summary()
    values = econ.customer_value(analysis.X_test).to_numpy()
    churned = analysis.y_test.to_numpy().astype(bool)
    def incentive(key, inc):
        inc_id, icon, name, blurb = INCENTIVE_META[key]
        if lang == "he":
            name, blurb = INCENTIVE_META_HE[inc_id]
        return {"id": inc_id, "icon": icon, "name": name, "blurb": blurb, "cost": inc.cost, "success": inc.success}

    return {
        "lang": lang,
        "best": best,
        "weighting": weighting,
        "budget": budget,
        "base_rate": float(analysis.base_rate),
        "default_threshold": 0.5,
        "models": [
            {"name": name, "color": colors.get(name, "#8a8984"), "cv_auc": float(cv.loc[name, "ROC_AUC mean"]),
             "cv_auc_std": float(cv.loc[name, "ROC_AUC std"]), "threshold": float(analysis.tuned_thresholds[name])}
            for name in analysis.models
        ],
        "econ": {
            "base": econ.base_value, "per_product": econ.per_product, "share": econ.balance_share,
            "incentives": [incentive(k, inc) for k, inc in econ.incentives.items()],
        },
        "rounds": [{"no": r, "customers": _round_customers(analysis, econ, r, lang)} for r in range(1, n_rounds + 1)],
        "scale": {
            "n": int(len(churned)), "churners": int(churned.sum()),
            "models": {name: scale_grid(analysis.test_scores[name], churned, values, (analysis.tuned_thresholds[name],))
                       for name in analysis.models},
        },
        "seed": config.RANDOM_STATE,
    }


def assets_fingerprint() -> str:
    """Changes whenever a game file changes: part of the dashboard's cache key, so edits show up."""
    import hashlib

    digest = hashlib.sha256()
    for name in ("game.html", "engine.js", "i18n.js"):
        digest.update((ASSETS / name).read_bytes())
    return digest.hexdigest()[:16]


def render_html(payload: dict) -> str:
    """The self-contained game page: template + engine + interface text + JSON payload (no external requests)."""
    template = (ASSETS / "game.html").read_text(encoding="utf-8")
    engine = (ASSETS / "engine.js").read_text(encoding="utf-8")
    i18n = (ASSETS / "i18n.js").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")  # safe inside <script>
    return (template.replace("/*__ENGINE__*/", engine).replace("/*__I18N__*/", i18n)
            .replace("/*__PAYLOAD__*/null", data))
