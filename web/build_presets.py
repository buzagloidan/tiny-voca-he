"""Phonemize the demo page's preset texts (RenikudPlus, the default front end). Writes web/presets.json."""
import json
import re
from tiny_voca.text import to_ipa, phonemize

WIKI = ("יעקב שלמה אחימאיר (נולד ב־21 ביולי 1938) הוא עיתונאי, פובליציסט ואיש טלוויזיה ורדיו ישראלי. "
        "חתן פרס ישראל לתקשורת לשנת אלפיים ושתים עשרה. אחימאיר נולד ברמת גן בשנת 1938, וגדל בירושלים. "
        "מראשית שנות השבעים, היה יעקב אחימאיר שדר בקול ישראל ובערוץ הראשון. הוא הגיש את התוכנית \"היום הזה\", "
        "עבד כמראיין ומנחה ב\"מוקד\", ושימש כעורך ומגיש ב\"יומן\" וככתב בוושינגטון.")
PRESETS = [
    ("greeting", "ברכה", "בוקר טוב! מה שלומך היום? ישנת טוב בלילה?"),
    ("reminder", "תזכורת", "רק להזכיר לך, בשעה 10:15 יש לך תור לרופאת המשפחה. אתה רוצה שאזמין לך מונית?"),
    ("numbers", "מספרים", "הלחץ שלך היום 130 על 85, והדופק 72. יש לך 3 תרופות לקחת היום."),
    ("question", "שאלה", "מה אכלת היום לארוחת צהריים? היה טעים?"),
    ("story", "סיפור", "כשהייתי ילד, סבא שלי היה לוקח אותי כל יום שישי לשוק. הוא היה קונה תפוזים, לחם חם וקצת חלבה, "
                        "ותמיד היה מתווכח עם המוכרים על המחיר. בדרך הביתה הוא היה מספר לי סיפורים על העלייה לארץ, ואני הייתי מקשיב בשקט."),
    ("wikipedia", "ויקיפדיה", WIKI),
]
out = []
for key, label, text in PRESETS:
    items = []
    for sentence in re.split(r"(?<=[.?!])\s+", text.strip()):
        ipa = to_ipa(sentence)
        p = phonemize(ipa)
        items.append({"text": sentence, "ipa": ipa, "ids": list(p.ids), "stress": list(p.stress)})
    out.append({"key": key, "label": label, "text": text, "sentences": items})
json.dump(out, open("web/presets.json", "w", encoding="utf-8"), ensure_ascii=False)
for preset in out:
    print(preset["key"], len(preset["sentences"]), "|", " / ".join(s["ipa"] for s in preset["sentences"])[:300])
