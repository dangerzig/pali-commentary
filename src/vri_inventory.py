#!/usr/bin/env python3
"""Generate a master inventory CSV of all VRI/CST4 raw text files.

Maps each file to its correct text name, true category label,
piṭaka membership, genre, and PTS volume correspondence.
"""

import csv
from pathlib import Path
from collections import Counter

VRI_RAW_DIR = Path.home() / "pali-canon" / "data" / "vri-raw"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "vri_inventory.csv"

# ── Master file identification ──
# Format: vri_file -> (text_name, correct_label, pitaka, genre, pts_volumes)
#
# correct_label: canon, commentary, subcommentary, sub-subcommentary,
#                paracanonical, manual, grammar, chronicle, devotional,
#                niti, council-qa, miscellaneous
# pitaka: sutta, vinaya, abhidhamma, extra
# pts_volumes: PTS edition volume(s) if applicable, empty string if none

FILE_INFO = {
    # ══════════════════════════════════════════════════════════════════
    # SUTTA PIṬAKA — Canon (.mul)
    # ══════════════════════════════════════════════════════════════════
    "s0101m.mul": ("Dīgha Nikāya I (Sīlakkhandhavagga)", "canon", "sutta", "sutta", "D I"),
    "s0102m.mul": ("Dīgha Nikāya II (Mahāvagga)", "canon", "sutta", "sutta", "D II"),
    "s0103m.mul": ("Dīgha Nikāya III (Pāthikavagga)", "canon", "sutta", "sutta", "D III"),
    "s0201m.mul": ("Majjhima Nikāya I (Mūlapaṇṇāsa)", "canon", "sutta", "sutta", "M I"),
    "s0202m.mul": ("Majjhima Nikāya II (Majjhimapaṇṇāsa)", "canon", "sutta", "sutta", "M II"),
    "s0203m.mul": ("Majjhima Nikāya III (Uparipaṇṇāsa)", "canon", "sutta", "sutta", "M III"),
    "s0301m.mul": ("Saṃyutta Nikāya I (Sagāthāvagga)", "canon", "sutta", "sutta", "S I"),
    "s0302m.mul": ("Saṃyutta Nikāya II (Nidānavagga)", "canon", "sutta", "sutta", "S II"),
    "s0303m.mul": ("Saṃyutta Nikāya III (Khandhavagga)", "canon", "sutta", "sutta", "S III"),
    "s0304m.mul": ("Saṃyutta Nikāya IV (Saḷāyatanavagga)", "canon", "sutta", "sutta", "S IV"),
    "s0305m.mul": ("Saṃyutta Nikāya V (Mahāvagga)", "canon", "sutta", "sutta", "S V"),
    "s0401m.mul": ("Aṅguttara Nikāya: Ekakanipāta", "canon", "sutta", "sutta", "A I (part)"),
    "s0402m1.mul": ("Aṅguttara Nikāya: Dukanipāta", "canon", "sutta", "sutta", "A I (part)"),
    "s0402m2.mul": ("Aṅguttara Nikāya: Tikanipāta", "canon", "sutta", "sutta", "A I (part)"),
    "s0402m3.mul": ("Aṅguttara Nikāya: Catukkanipāta", "canon", "sutta", "sutta", "A II"),
    "s0403m1.mul": ("Aṅguttara Nikāya: Pañcakanipāta", "canon", "sutta", "sutta", "A III (part)"),
    "s0403m2.mul": ("Aṅguttara Nikāya: Chakkanipāta", "canon", "sutta", "sutta", "A III (part)"),
    "s0403m3.mul": ("Aṅguttara Nikāya: Sattakanipāta", "canon", "sutta", "sutta", "A IV (part)"),
    "s0404m1.mul": ("Aṅguttara Nikāya: Aṭṭhakanipāta", "canon", "sutta", "sutta", "A IV (part)"),
    "s0404m2.mul": ("Aṅguttara Nikāya: Navakanipāta", "canon", "sutta", "sutta", "A IV (part)"),
    "s0404m3.mul": ("Aṅguttara Nikāya: Dasakanipāta", "canon", "sutta", "sutta", "A V (part)"),
    "s0404m4.mul": ("Aṅguttara Nikāya: Ekādasakanipāta", "canon", "sutta", "sutta", "A V (part)"),
    "s0501m.mul": ("Khuddakapāṭha", "canon", "sutta", "sutta", "Khp"),
    "s0502m.mul": ("Dhammapada", "canon", "sutta", "verse", "Dhp"),
    "s0503m.mul": ("Udāna", "canon", "sutta", "sutta", "Ud"),
    "s0504m.mul": ("Itivuttaka", "canon", "sutta", "sutta", "It"),
    "s0505m.mul": ("Suttanipāta", "canon", "sutta", "verse", "Sn"),
    "s0506m.mul": ("Vimānavatthu", "canon", "sutta", "verse", "Vv"),
    "s0507m.mul": ("Petavatthu", "canon", "sutta", "verse", "Pv"),
    "s0508m.mul": ("Theragāthā", "canon", "sutta", "verse", "Th"),
    "s0509m.mul": ("Therīgāthā", "canon", "sutta", "verse", "Thī"),
    "s0510m1.mul": ("Apadāna I", "canon", "sutta", "verse", "Ap I"),
    "s0510m2.mul": ("Apadāna II", "canon", "sutta", "verse", "Ap II"),
    "s0511m.mul": ("Buddhavaṃsa", "canon", "sutta", "verse", "Bv"),
    "s0512m.mul": ("Cariyāpiṭaka", "canon", "sutta", "verse", "Cp"),
    "s0513m.mul": ("Jātaka I (VRI)", "canon", "sutta", "narrative", "J I–III"),
    "s0514m.mul": ("Jātaka II (VRI)", "canon", "sutta", "narrative", "J IV–VI"),
    "s0515m.mul": ("Mahāniddesa", "canon", "sutta", "exegetical", "Nidd I"),
    "s0516m.mul": ("Cūḷaniddesa", "canon", "sutta", "exegetical", "Nidd II"),
    "s0517m.mul": ("Paṭisambhidāmagga", "canon", "sutta", "exegetical", "Paṭis I–II"),
    "s0519m.mul": ("Nettippakaraṇa", "canon", "sutta", "exegetical", "Nett"),

    # Sutta — Paracanonical/Disputed (.nrf with .mul suffix in filename)
    "s0518m.nrf": ("Milindapañha", "paracanonical", "sutta", "paracanonical", "Mil"),
    "s0520m.nrf": ("Peṭakopadesa", "paracanonical", "sutta", "paracanonical", "Peṭ"),

    # ══════════════════════════════════════════════════════════════════
    # SUTTA PIṬAKA — Commentary (.att)
    # ══════════════════════════════════════════════════════════════════
    "s0101a.att": ("Sumaṅgalavilāsinī I", "commentary", "sutta", "commentary", "Sv I"),
    "s0102a.att": ("Sumaṅgalavilāsinī II", "commentary", "sutta", "commentary", "Sv II"),
    "s0103a.att": ("Sumaṅgalavilāsinī III", "commentary", "sutta", "commentary", "Sv III"),
    "s0201a.att": ("Papañcasūdanī I", "commentary", "sutta", "commentary", "Ps I"),
    "s0202a.att": ("Papañcasūdanī II", "commentary", "sutta", "commentary", "Ps II"),
    "s0203a.att": ("Papañcasūdanī III", "commentary", "sutta", "commentary", "Ps III"),
    "s0301a.att": ("Sāratthappakāsinī I", "commentary", "sutta", "commentary", "Spk I"),
    "s0302a.att": ("Sāratthappakāsinī II", "commentary", "sutta", "commentary", "Spk II"),
    "s0303a.att": ("Sāratthappakāsinī III", "commentary", "sutta", "commentary", "Spk III"),
    "s0304a.att": ("Sāratthappakāsinī IV", "commentary", "sutta", "commentary", ""),
    "s0305a.att": ("Sāratthappakāsinī V", "commentary", "sutta", "commentary", ""),
    "s0401a.att": ("Manorathapūraṇī I", "commentary", "sutta", "commentary", "Mp I"),
    "s0402a.att": ("Manorathapūraṇī II", "commentary", "sutta", "commentary", "Mp II"),
    "s0403a.att": ("Manorathapūraṇī III", "commentary", "sutta", "commentary", "Mp III"),
    "s0404a.att": ("Manorathapūraṇī IV", "commentary", "sutta", "commentary", "Mp IV–V"),
    "s0501a.att": ("Paramatthajotikā I (Khuddakapāṭha-aṭṭhakathā)", "commentary", "sutta", "commentary", "Pj I"),
    "s0502a.att": ("Dhammapadaṭṭhakathā", "commentary", "sutta", "commentary", "DhpA I–IV"),
    "s0503a.att": ("Paramatthadīpanī (Udāna-aṭṭhakathā)", "commentary", "sutta", "commentary", "Ud-a"),
    "s0504a.att": ("Paramatthadīpanī (Itivuttaka-aṭṭhakathā)", "commentary", "sutta", "commentary", "It-a I–II"),
    "s0505a.att": ("Paramatthajotikā II (Suttanipāta-aṭṭhakathā)", "commentary", "sutta", "commentary", "Pj II"),
    "s0506a.att": ("Paramatthadīpanī (Vimānavatthu-aṭṭhakathā)", "commentary", "sutta", "commentary", "Vv-a"),
    "s0507a.att": ("Paramatthadīpanī (Petavatthu-aṭṭhakathā)", "commentary", "sutta", "commentary", "Pv-a"),
    "s0508a1.att": ("Paramatthadīpanī (Theragāthā-aṭṭhakathā) I", "commentary", "sutta", "commentary", "Th-a I"),
    "s0508a2.att": ("Paramatthadīpanī (Theragāthā-aṭṭhakathā) II", "commentary", "sutta", "commentary", "Th-a II"),
    "s0509a.att": ("Paramatthadīpanī (Therīgāthā-aṭṭhakathā)", "commentary", "sutta", "commentary", "Thī-a"),
    "s0510a.att": ("Visuddhajanavilāsinī (Apadāna-aṭṭhakathā)", "commentary", "sutta", "commentary", ""),
    "s0511a.att": ("Madhuratthavilāsinī (Buddhavaṃsa-aṭṭhakathā)", "commentary", "sutta", "commentary", "Bv-a"),
    "s0512a.att": ("Paramatthadīpanī (Cariyāpiṭaka-aṭṭhakathā)", "commentary", "sutta", "commentary", "Cp-a"),
    "s0513a1.att": ("Jātakaṭṭhavaṇṇanā I", "commentary", "sutta", "commentary", "Ja I"),
    "s0513a2.att": ("Jātakaṭṭhavaṇṇanā II", "commentary", "sutta", "commentary", "Ja II"),
    "s0513a3.att": ("Jātakaṭṭhavaṇṇanā III", "commentary", "sutta", "commentary", "Ja III"),
    "s0513a4.att": ("Jātakaṭṭhavaṇṇanā IV", "commentary", "sutta", "commentary", ""),
    "s0514a1.att": ("Jātakaṭṭhavaṇṇanā V", "commentary", "sutta", "commentary", ""),
    "s0514a2.att": ("Jātakaṭṭhavaṇṇanā VI", "commentary", "sutta", "commentary", ""),
    "s0514a3.att": ("Jātakaṭṭhavaṇṇanā VII", "commentary", "sutta", "commentary", ""),
    "s0515a.att": ("Saddhammapajjotikā (Mahāniddesa-aṭṭhakathā)", "commentary", "sutta", "commentary", "Nidd-a I"),
    "s0516a.att": ("Saddhammapajjotikā (Cūḷaniddesa-aṭṭhakathā)", "commentary", "sutta", "commentary", "Nidd-a II"),
    "s0517a.att": ("Saddhammappakāsinī (Paṭisambhidāmagga-aṭṭhakathā)", "commentary", "sutta", "commentary", ""),
    "s0519a.att": ("Nettippakaraṇa-aṭṭhakathā", "commentary", "sutta", "commentary", ""),

    # ══════════════════════════════════════════════════════════════════
    # SUTTA PIṬAKA — Sub-commentary (.tik)
    # ══════════════════════════════════════════════════════════════════
    "s0101t.tik": ("Līnatthavaṇṇanā (DN I)", "subcommentary", "sutta", "subcommentary", ""),
    "s0102t.tik": ("Līnatthavaṇṇanā (DN II)", "subcommentary", "sutta", "subcommentary", ""),
    "s0103t.tik": ("Līnatthavaṇṇanā (DN III)", "subcommentary", "sutta", "subcommentary", ""),
    "s0201t.tik": ("Līnatthavaṇṇanā (MN I)", "subcommentary", "sutta", "subcommentary", ""),
    "s0202t.tik": ("Līnatthavaṇṇanā (MN II)", "subcommentary", "sutta", "subcommentary", ""),
    "s0203t.tik": ("Līnatthavaṇṇanā (MN III)", "subcommentary", "sutta", "subcommentary", ""),
    "s0301t.tik": ("Līnatthavaṇṇanā (SN I)", "subcommentary", "sutta", "subcommentary", ""),
    "s0302t.tik": ("Līnatthavaṇṇanā (SN II)", "subcommentary", "sutta", "subcommentary", ""),
    "s0303t.tik": ("Līnatthavaṇṇanā (SN III)", "subcommentary", "sutta", "subcommentary", ""),
    "s0304t.tik": ("Līnatthavaṇṇanā (SN IV)", "subcommentary", "sutta", "subcommentary", ""),
    "s0305t.tik": ("Līnatthavaṇṇanā (SN V)", "subcommentary", "sutta", "subcommentary", ""),
    "s0401t.tik": ("Sāratthamañjūsā (AN I)", "subcommentary", "sutta", "subcommentary", ""),
    "s0402t.tik": ("Sāratthamañjūsā (AN II)", "subcommentary", "sutta", "subcommentary", ""),
    "s0403t.tik": ("Sāratthamañjūsā (AN III)", "subcommentary", "sutta", "subcommentary", ""),
    "s0404t.tik": ("Sāratthamañjūsā (AN IV)", "subcommentary", "sutta", "subcommentary", ""),
    "s0519t.tik": ("Nettippakaraṇa-ṭīkā", "subcommentary", "sutta", "subcommentary", ""),

    # Sutta — Mislabelled .nrf (actually sub-sub-commentary or sub-commentary)
    "s0104t.nrf": ("Sīlakkhandhavagga-abhinavaṭīkā I", "sub-subcommentary", "sutta", "sub-subcommentary", ""),
    "s0105t.nrf": ("Sīlakkhandhavagga-abhinavaṭīkā II", "sub-subcommentary", "sutta", "sub-subcommentary", ""),
    "s0501t.nrf": ("Nettivibhāvinī", "subcommentary", "sutta", "subcommentary", ""),

    # ══════════════════════════════════════════════════════════════════
    # VINAYA PIṬAKA — Canon (.mul)
    # ══════════════════════════════════════════════════════════════════
    "vin01m.mul": ("Pārājikapāḷi", "canon", "vinaya", "vinaya", "Vin III"),
    "vin02m1.mul": ("Pācittiyapāḷi", "canon", "vinaya", "vinaya", "Vin IV"),
    "vin02m2.mul": ("Mahāvaggapāḷi", "canon", "vinaya", "vinaya", "Vin I"),
    "vin02m3.mul": ("Cūḷavaggapāḷi", "canon", "vinaya", "vinaya", "Vin II"),
    "vin02m4.mul": ("Parivārapāḷi", "canon", "vinaya", "vinaya", "Vin V"),

    # Vinaya — Commentary (.att)
    "vin01a.att": ("Samantapāsādikā I", "commentary", "vinaya", "commentary", "Sp I"),
    "vin02a1.att": ("Samantapāsādikā II", "commentary", "vinaya", "commentary", "Sp II"),
    "vin02a2.att": ("Samantapāsādikā III", "commentary", "vinaya", "commentary", "Sp III"),
    "vin02a3.att": ("Samantapāsādikā IV", "commentary", "vinaya", "commentary", ""),
    "vin02a4.att": ("Samantapāsādikā V", "commentary", "vinaya", "commentary", ""),

    # Vinaya — Sub-commentary (.tik)
    "vin01t1.tik": ("Sāratthadīpanī-ṭīkā I", "subcommentary", "vinaya", "subcommentary", ""),
    "vin01t2.tik": ("Sāratthadīpanī-ṭīkā II", "subcommentary", "vinaya", "subcommentary", ""),
    "vin02t.tik": ("Sāratthadīpanī-ṭīkā III", "subcommentary", "vinaya", "subcommentary", ""),

    # Vinaya — Mislabelled .nrf (various ṭīkās and manuals)
    "vin04t.nrf": ("Dvemātikāpāḷi", "manual", "vinaya", "manual", ""),
    "vin05t.nrf": ("Vinayasaṅgaha-aṭṭhakathā", "commentary", "vinaya", "commentary", ""),
    "vin06t.nrf": ("Vajirabuddhi-ṭīkā", "subcommentary", "vinaya", "subcommentary", ""),
    "vin07t.nrf": ("Vimativinodanī-ṭīkā", "subcommentary", "vinaya", "subcommentary", ""),
    "vin08t.nrf": ("Vinayālaṅkāra-ṭīkā", "subcommentary", "vinaya", "subcommentary", ""),
    "vin09t.nrf": ("Kaṅkhāvitaraṇī-purāṇaṭīkā", "subcommentary", "vinaya", "subcommentary", ""),
    "vin10t.nrf": ("Vinayavinicchaya", "manual", "vinaya", "manual", ""),
    "vin11t.nrf": ("Vinayavinicchaya-ṭīkā", "subcommentary", "vinaya", "subcommentary", ""),
    "vin12t.nrf": ("Pācityādiyojanā", "manual", "vinaya", "manual", ""),
    "vin13t.nrf": ("Khuddasikkhā-Mūlasikkhā", "manual", "vinaya", "manual", ""),

    # ══════════════════════════════════════════════════════════════════
    # ABHIDHAMMA PIṬAKA — Canon (.mul)
    # ══════════════════════════════════════════════════════════════════
    "abh01m.mul": ("Dhammasaṅgaṇī", "canon", "abhidhamma", "abhidhamma", "Dhs"),
    "abh02m.mul": ("Vibhaṅga", "canon", "abhidhamma", "abhidhamma", "Vibh"),
    "abh03m1.mul": ("Dhātukathā", "canon", "abhidhamma", "abhidhamma", "Dhātuk"),
    "abh03m2.mul": ("Puggalapaññatti", "canon", "abhidhamma", "abhidhamma", "Pp"),
    "abh03m3.mul": ("Kathāvatthu", "canon", "abhidhamma", "abhidhamma", "Kv"),
    "abh03m4.mul": ("Yamaka I", "canon", "abhidhamma", "abhidhamma", "Yam I"),
    "abh03m5.mul": ("Yamaka II", "canon", "abhidhamma", "abhidhamma", "Yam II"),
    "abh03m6.mul": ("Yamaka III", "canon", "abhidhamma", "abhidhamma", ""),
    "abh03m7.mul": ("Paṭṭhāna I (Duka-paṭṭhāna)", "canon", "abhidhamma", "abhidhamma", "Paṭṭh I"),
    "abh03m8.mul": ("Paṭṭhāna II", "canon", "abhidhamma", "abhidhamma", ""),
    "abh03m9.mul": ("Paṭṭhāna III", "canon", "abhidhamma", "abhidhamma", ""),
    "abh03m10.mul": ("Paṭṭhāna IV", "canon", "abhidhamma", "abhidhamma", ""),
    "abh03m11.mul": ("Paṭṭhāna V", "canon", "abhidhamma", "abhidhamma", ""),

    # Abhidhamma — Commentary (.att)
    "abh01a.att": ("Aṭṭhasālinī", "commentary", "abhidhamma", "commentary", "As"),
    "abh02a.att": ("Sammohavinodanī", "commentary", "abhidhamma", "commentary", ""),
    "abh03a.att": ("Pañcappakaraṇaṭṭhakathā", "commentary", "abhidhamma", "commentary", ""),

    # Abhidhamma — Sub-commentary (.tik)
    "abh01t.tik": ("Dhammasaṅgaṇī-mūlaṭīkā", "subcommentary", "abhidhamma", "subcommentary", ""),
    "abh02t.tik": ("Vibhaṅga-mūlaṭīkā", "subcommentary", "abhidhamma", "subcommentary", ""),
    "abh03t.tik": ("Pañcapakaraṇa-mūlaṭīkā", "subcommentary", "abhidhamma", "subcommentary", ""),

    # Abhidhamma — Mislabelled .nrf
    "abh04t.nrf": ("Dhammasaṅgaṇī-anuṭīkā", "sub-subcommentary", "abhidhamma", "sub-subcommentary", ""),
    "abh05t.nrf": ("Pañcapakaraṇa-anuṭīkā", "sub-subcommentary", "abhidhamma", "sub-subcommentary", ""),
    "abh06t.nrf": ("Abhidhammāvatāra", "manual", "abhidhamma", "manual", ""),
    "abh07t.nrf": ("Abhidhammatthasaṅgaha", "manual", "abhidhamma", "manual", ""),
    "abh08t.nrf": ("Abhidhammāvatāra-purāṇaṭīkā", "subcommentary", "abhidhamma", "subcommentary", ""),
    "abh09t.nrf": ("Abhidhammamātikāpāḷi", "manual", "abhidhamma", "manual", ""),

    # ══════════════════════════════════════════════════════════════════
    # E-SERIES — Miscellaneous later Pāli literature
    # ══════════════════════════════════════════════════════════════════

    # Visuddhimagga and related
    "e0101n.mul": ("Visuddhimagga I", "manual", "extra", "manual", "Vism"),
    "e0102n.mul": ("Visuddhimagga II", "manual", "extra", "manual", "Vism"),
    "e0103n.att": ("Visuddhimagga-mahāṭīkā I", "subcommentary", "extra", "subcommentary", ""),
    "e0104n.att": ("Visuddhimagga-mahāṭīkā II", "subcommentary", "extra", "subcommentary", ""),
    "e0105n.nrf": ("Visuddhimagga-nidānakathā", "miscellaneous", "extra", "miscellaneous", ""),

    # Abhidhamma-related treatises
    "e0201n.nrf": ("Niruttidīpanī", "grammar", "extra", "grammar", ""),
    "e0301n.nrf": ("Paramatthadīpanī (Abhidhammatthasaṅgaha-mahāṭīkā)", "subcommentary", "extra", "subcommentary", ""),
    "e0401n.nrf": ("Anudīpanī", "sub-subcommentary", "extra", "sub-subcommentary", ""),
    "e0501n.nrf": ("Paṭṭhānuddesa-dīpanī", "manual", "extra", "manual", ""),

    # Devotional and liturgical texts
    "e0601n.nrf": ("Namakkārapāḷi", "devotional", "extra", "devotional", ""),
    "e0602n.nrf": ("Mahāpaṇāmapāṭha", "devotional", "extra", "devotional", ""),
    "e0603n.nrf": ("Lakkhaṇāto", "devotional", "extra", "devotional", ""),
    "e0604n.nrf": ("Suttavandanā", "devotional", "extra", "devotional", ""),
    "e0605n.nrf": ("Jinālaṅkāra", "devotional", "extra", "devotional", ""),
    "e0606n.nrf": ("Kamalāñjali", "devotional", "extra", "devotional", ""),
    "e0607n.nrf": ("Pajjamadhu", "devotional", "extra", "devotional", ""),
    "e0608n.nrf": ("Buddhaguṇagāthāvalī", "devotional", "extra", "devotional", ""),

    # Chronicles
    "e0701n.nrf": ("Cūḷaganthavaṃsa", "chronicle", "extra", "chronicle", ""),
    "e0702n.nrf": ("Sāsanavaṃsappadīpikā", "chronicle", "extra", "chronicle", ""),
    "e0703n.nrf": ("Mahāvaṃsa", "chronicle", "extra", "chronicle", "Mhv"),

    # Grammar texts
    "e0801n.nrf": ("Moggallānasutta", "grammar", "extra", "grammar", ""),
    "e0802n.nrf": ("Kaccāyanabyākaraṇa", "grammar", "extra", "grammar", ""),
    "e0803n.nrf": ("Saddanīti (Padamālā)", "grammar", "extra", "grammar", ""),
    "e0804n.nrf": ("Saddanīti (Dhātumālā)", "grammar", "extra", "grammar", ""),
    "e0805n.nrf": ("Padarūpasiddhi", "grammar", "extra", "grammar", ""),
    "e0806n.nrf": ("Moggallāna-pañcikāṭīkā", "grammar", "extra", "grammar", ""),
    "e0807n.nrf": ("Payogasiddhi", "grammar", "extra", "grammar", ""),
    "e0808n.nrf": ("Vuttodaya", "grammar", "extra", "grammar", ""),
    "e0809n.nrf": ("Abhidhānappadīpikā", "grammar", "extra", "grammar", ""),
    "e0810n.nrf": ("Abhidhānappadīpikā-ṭīkā", "grammar", "extra", "grammar", ""),
    "e0811n.nrf": ("Subodhālaṅkāra", "grammar", "extra", "grammar", ""),
    "e0812n.nrf": ("Subodhālaṅkāra-ṭīkā", "grammar", "extra", "grammar", ""),
    "e0813n.nrf": ("Bālāvatāra", "grammar", "extra", "grammar", ""),

    # Council Q&A (Chaṭṭhasaṃgāyana pucchā-vissajjanā)
    "e0901n.nrf": ("Saṃgāyana pucchā-vissajjanā (Suttanta I)", "council-qa", "extra", "council-qa", ""),
    "e0902n.nrf": ("Saṃgāyana pucchā-vissajjanā (Suttanta II)", "council-qa", "extra", "council-qa", ""),
    "e0903n.nrf": ("Saṃgāyana pucchā-vissajjanā (Suttanta III)", "council-qa", "extra", "council-qa", ""),
    "e0904n.nrf": ("Saṃgāyana pucchā-vissajjanā (Suttanta IV)", "council-qa", "extra", "council-qa", ""),
    "e0905n.nrf": ("Saṃgāyana pucchā-vissajjanā (Vinaya)", "council-qa", "extra", "council-qa", ""),
    "e0906n.nrf": ("Saṃgāyana pucchā-vissajjanā (Abhidhamma)", "council-qa", "extra", "council-qa", ""),
    "e0907n.nrf": ("Saṃgāyana pucchā-vissajjanā (Aṭṭhakathā)", "council-qa", "extra", "council-qa", ""),

    # Nīti literature
    "e1001n.nrf": ("Kavidappaṇanīti", "niti", "extra", "niti", ""),
    "e1002n.nrf": ("Nītimañjarī", "niti", "extra", "niti", ""),
    "e1003n.nrf": ("Dhammanīti", "niti", "extra", "niti", ""),
    "e1004n.nrf": ("Mahārahanīti", "niti", "extra", "niti", ""),
    "e1005n.nrf": ("Lokanīti", "niti", "extra", "niti", ""),
    "e1006n.nrf": ("Suttantanīti", "niti", "extra", "niti", ""),
    "e1007n.nrf": ("Sūrassatīnīti", "niti", "extra", "niti", ""),
    "e1008n.nrf": ("Cāṇakyanīti", "niti", "extra", "niti", ""),
    "e1009n.nrf": ("Naradakkhadīpanī", "miscellaneous", "extra", "miscellaneous", ""),
    "e1010n.nrf": ("Caturārakkhadīpanī", "miscellaneous", "extra", "miscellaneous", ""),

    # Narrative and miscellaneous
    "e1101n.nrf": ("Rasavāhinī", "narrative", "extra", "narrative", ""),
    "e1102n.nrf": ("Sīmavisodhanī", "manual", "extra", "manual", ""),
    "e1103n.nrf": ("Vessantarāgīti", "devotional", "extra", "devotional", ""),

    # Further grammars, chronicles, and miscellaneous (e12xx)
    "e1201n.nrf": ("Kaccāyanavutti", "grammar", "extra", "grammar", ""),
    "e1202n.nrf": ("Thūpavaṃsa", "chronicle", "extra", "chronicle", ""),
    "e1203n.nrf": ("Unidentified verse text", "miscellaneous", "extra", "miscellaneous", ""),
    "e1204n.nrf": ("Dhātupāṭhavilāsinī", "grammar", "extra", "grammar", ""),
    "e1205n.nrf": ("Dhātuvaṃsa", "chronicle", "extra", "chronicle", ""),
    "e1206n.nrf": ("Unidentified verse text", "miscellaneous", "extra", "miscellaneous", ""),
    "e1207n.nrf": ("Unidentified verse collection", "miscellaneous", "extra", "miscellaneous", ""),
    "e1208n.nrf": ("Jinavaṃsadīpa", "chronicle", "extra", "chronicle", ""),
    "e1209n.nrf": ("Unidentified (Sri Lankan royal text)", "miscellaneous", "extra", "miscellaneous", ""),
    "e1210n.nrf": ("Unidentified chronicle/narrative", "chronicle", "extra", "chronicle", ""),
    "e1211n.nrf": ("Padamañjarī", "grammar", "extra", "grammar", ""),
    "e1212n.nrf": ("Padasādhana", "grammar", "extra", "grammar", ""),
    "e1213n.nrf": ("Saddabindu", "grammar", "extra", "grammar", ""),
    "e1214n.nrf": ("Unidentified grammar text", "grammar", "extra", "grammar", ""),
    "e1215n.nrf": ("Unidentified text (by Vedeha thera)", "miscellaneous", "extra", "miscellaneous", ""),
}


def get_file_key(filename: str) -> str:
    """Extract the key (e.g., 's0101m.mul') from filename like 's0101m.mul.txt'."""
    return filename.rsplit(".", 1)[0]  # strip '.txt'


def count_words(filepath: Path) -> int:
    """Count whitespace-delimited words in a file."""
    text = filepath.read_text(encoding="utf-8")
    return len(text.split())


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    for filepath in sorted(VRI_RAW_DIR.glob("*.txt")):
        if filepath.name == "README.txt":
            continue

        key = get_file_key(filepath.name)
        word_count = count_words(filepath)

        if key in FILE_INFO:
            text_name, correct_label, pitaka, genre, pts_vol = FILE_INFO[key]
        else:
            text_name = "UNIDENTIFIED"
            correct_label = "unknown"
            pitaka = "unknown"
            genre = "unknown"
            pts_vol = ""
            missing.append(filepath.name)

        # Extract VRI suffix (mul/att/tik/nrf)
        parts = filepath.name.split(".")
        vri_suffix = parts[1] if len(parts) >= 3 else ""

        # VRI's automatic label
        vri_label_map = {"mul": "canon", "att": "commentary", "tik": "subcommentary", "nrf": "paracanonical"}
        vri_label = vri_label_map.get(vri_suffix, "unknown")

        mislabelled = "yes" if vri_label != correct_label else ""

        rows.append({
            "vri_file": filepath.name,
            "text_name": text_name,
            "vri_suffix": vri_suffix,
            "vri_label": vri_label,
            "correct_label": correct_label,
            "mislabelled": mislabelled,
            "pitaka": pitaka,
            "genre": genre,
            "pts_volume": pts_vol,
            "word_count": word_count,
        })

    # Write CSV
    fieldnames = [
        "vri_file", "text_name", "vri_suffix", "vri_label", "correct_label",
        "mislabelled", "pitaka", "genre", "pts_volume", "word_count",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    if missing:
        print(f"\nWARNING: {len(missing)} unidentified files:")
        for m in missing:
            print(f"  {m}")

    # Summary statistics
    from collections import Counter
    labels = Counter(r["correct_label"] for r in rows)
    pitakas = Counter(r["pitaka"] for r in rows)
    mislabelled_count = sum(1 for r in rows if r["mislabelled"])

    print(f"\nBy correct label:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count}")

    print(f"\nBy piṭaka:")
    for p, count in sorted(pitakas.items()):
        print(f"  {p}: {count}")

    print(f"\nMislabelled by VRI: {mislabelled_count}")


if __name__ == "__main__":
    main()
