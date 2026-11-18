"""Hebrew text handling: the two phonemizers, numbers and sentence splitting.

The phonikud cases need its model (PHONIKUD_MODEL); the RenikudPlus cases need
``renikud-plus`` (its model downloads on first use). Both skip otherwise.
"""

import os
import unittest

from tiny_voca import text
from tiny_voca.infer import sentences

HAVE_PHONIKUD = os.path.isfile(os.environ.get(text.DIACRITIZER_VARIABLE, ""))
# The front end checkpoints up to he-v2 were trained with.
PHONIKUD_FINGERPRINT = "b5bddd5e867d0df0202ab50f2f4ac0a96a6c030e092f34348eb7305aa31ea6e5"
try:
    import renikud_onnx  # noqa: F401
    HAVE_RENIKUD = True
except ImportError:
    HAVE_RENIKUD = False


class PrefixHyphenTests(unittest.TestCase):
    def test_only_a_hyphen_between_a_hebrew_letter_and_a_digit_is_dropped(self):
        for before, after in (("ב-21 ביולי", "ב21 ביולי"), ("ה-12 בחודש", "ה12 בחודש"), ("מ־1948", "מ1948"),
                              ("-5 מעלות", "-5 מעלות"),          # a real minus sign
                              ("תל-אביב", "תל-אביב"),            # a word hyphen
                              ("3-4 ימים", "3-4 ימים")):         # a range
            self.assertEqual(text._PREFIX_HYPHEN.sub("", before), after, before)


@unittest.skipUnless(HAVE_PHONIKUD, "needs the phonikud-onnx model in PHONIKUD_MODEL")
class PhonikudNumberTests(unittest.TestCase):
    def setUp(self):
        text.set_frontend("hebrew")

    def tearDown(self):
        text.set_frontend("renikud")

    def test_a_number_after_a_prefix_is_not_read_as_minus(self):
        for written in ("נולד ב-21 ביולי", "ה-12 בחודש"):
            self.assertNotIn("minus", text.to_ipa(written).replace("ˈ", ""), written)
        self.assertIn("minus", text.to_ipa("-5 מעלות").replace("ˈ", ""))

    def test_twelve_keeps_its_vowels(self):
        self.assertIn("ʃtˈem ʔesʁˈe", text.to_ipa("ה-12 בחודש"))
        self.assertIn("ʃnˈem ʔˈeseʁ", text.to_ipa("12000 איש"))
        self.assertEqual(text.phonemize("בשנת 2012").unknown, ())


class FrontEndTests(unittest.TestCase):
    def tearDown(self):
        text.set_frontend("renikud")

    def test_renikud_is_the_default_and_phonikud_keeps_the_he_v2_fingerprint(self):
        self.assertEqual(text.FRONTENDS[0], "renikud")
        text.set_frontend("hebrew")
        self.assertEqual(text.frontend_fingerprint(), PHONIKUD_FINGERPRINT)
        text.set_frontend("renikud")
        self.assertNotEqual(text.frontend_fingerprint(), PHONIKUD_FINGERPRINT)

    @unittest.skipUnless(HAVE_RENIKUD, "needs renikud-plus and its model")
    def test_renikud_reads_numbers_and_stays_in_the_inventory(self):
        text.set_frontend("renikud")
        self.assertEqual(text.to_ipa("ה-12 בחודש").replace("ˈ", ""), "haʃnem ʔasaʁ beχodeʃ")
        self.assertIn("ʔeseʁ vaʁeva", text.to_ipa("בשעה 10:15").replace("ˈ", ""))
        self.assertEqual(text.phonemize("נולד ב-21 ביולי 1938, ברמת גן.").unknown, ())
        self.assertEqual(text.phonemize("יצא לאור ב־10 בנובמבר 2016.").unknown, ())  # maqaf


class HebrewSentenceTests(unittest.TestCase):
    def test_hebrew_sentences_are_split(self):
        self.assertEqual(sentences("שלום לכולם. מה שלומכם? הכול טוב!"), ["שלום לכולם.", "מה שלומכם?", "הכול טוב!"])
        self.assertEqual(sentences('הוא אמר: "בוקר טוב." ואז הלך.'), ['הוא אמר: "בוקר טוב."', "ואז הלך."])


if __name__ == "__main__":
    unittest.main()
