#!/usr/bin/env python3
"""Save ClinCalc Top 300 Drugs (2023 edition) to a TSV.

Source: https://clincalc.com/DrugStats/Top300Drugs.aspx (2023 data, version 2025.08).
Generic-only (ClinCalc's public table doesn't include brand names).
Combination drugs (joined by ';' on ClinCalc) are split into one row per generic;
the rank is shared across the combo's generics.

Output: stapes/data/clincalc_top300.tsv
Columns: rank, generic (lowercase, single name per row, deduped on first occurrence)
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR.parent / 'data' / 'clincalc_top300.tsv'
OUT.parent.mkdir(parents=True, exist_ok=True)

RAW = """1\tAtorvastatin
2\tMetformin
3\tLevothyroxine
4\tLisinopril
5\tAmlodipine
6\tMetoprolol
7\tAlbuterol
8\tLosartan
9\tGabapentin
10\tOmeprazole
11\tSertraline
12\tRosuvastatin
13\tPantoprazole
14\tEscitalopram
15\tDextroamphetamine; Dextroamphetamine Saccharate; Amphetamine; Amphetamine Aspartate
16\tHydrochlorothiazide
17\tBupropion
18\tFluoxetine
19\tSemaglutide
20\tMontelukast
21\tTrazodone
22\tSimvastatin
23\tAmoxicillin
24\tTamsulosin
25\tAcetaminophen; Hydrocodone
26\tFluticasone
27\tMeloxicam
28\tApixaban
29\tFurosemide
30\tInsulin Glargine
31\tDuloxetine
32\tIbuprofen
33\tFamotidine
34\tEmpagliflozin
35\tCarvedilol
36\tTramadol
37\tAlprazolam
38\tPrednisone
39\tHydroxyzine
40\tBuspirone
41\tClopidogrel
42\tGlipizide
43\tCitalopram
44\tPotassium Chloride
45\tAllopurinol
46\tAspirin
47\tCyclobenzaprine
48\tErgocalciferol
49\tOxycodone
50\tMethylphenidate
51\tVenlafaxine
52\tSpironolactone
53\tOndansetron
54\tZolpidem
55\tCetirizine
56\tEstradiol
57\tPravastatin
58\tHydrochlorothiazide; Lisinopril
59\tLamotrigine
60\tQuetiapine
61\tFluticasone; Salmeterol
62\tClonazepam
63\tDulaglutide
64\tAzithromycin
65\tHydrochlorothiazide; Losartan
66\tAmoxicillin; Clavulanate
67\tLatanoprost
68\tCholecalciferol
69\tPropranolol
70\tEzetimibe
71\tTopiramate
72\tParoxetine
73\tDiclofenac
74\tBudesonide; Formoterol
75\tAtenolol
76\tLisdexamfetamine
77\tDoxycycline
78\tPregabalin
79\tEthinyl Estradiol; Norethindrone
80\tGlimepiride
81\tTizanidine
82\tClonidine
83\tFenofibrate
84\tInsulin Lispro
85\tValsartan
86\tCephalexin
87\tBaclofen
88\tRivaroxaban
89\tFerrous Sulfate
90\tAmitriptyline
91\tFinasteride
92\tDapagliflozin
93\tAcetaminophen; Oxycodone
94\tFolic Acid
95\tAripiprazole
96\tOlmesartan
97\tEthinyl Estradiol; Norgestimate
98\tValacyclovir
99\tMirtazapine
100\tLorazepam
101\tLevetiracetam
102\tInsulin Aspart
103\tNaproxen
104\tCyanocobalamin
105\tLoratadine
106\tDiltiazem
107\tSumatriptan
108\tTriamcinolone
109\tHydralazine
110\tTirzepatide
111\tCelecoxib
112\tAcetaminophen
113\tAlendronate
114\tOxybutynin
115\tHydrochlorothiazide; Triamterene
116\tWarfarin
117\tProgesterone
118\tFluticasone; Umeclidinium; Vilanterol
119\tTestosterone
120\tNifedipine
121\tMethocarbamol
122\tBenzonatate
123\tSitagliptin
124\tChlorthalidone
125\tIsosorbide
126\tDonepezil
127\tDexmethylphenidate
128\tSulfamethoxazole; Trimethoprim
129\tClobetasol
130\tMethotrexate
131\tHydroxychloroquine
132\tLovastatin
133\tPioglitazone
134\tIrbesartan
135\tMethylprednisolone
136\tNorethindrone
137\tMeclizine
138\tEthinyl Estradiol; Levonorgestrel
139\tFluticasone; Vilanterol
140\tKetoconazole
141\tThyroid
142\tAzelastine
143\tNitrofurantoin
144\tAdalimumab
145\tMemantine
146\tPrednisolone
147\tEsomeprazole
148\tDocusate
149\tClindamycin
150\tAcyclovir
151\tSildenafil
152\tInsulin Degludec
153\tInsulin Detemir
154\tDrospirenone; Ethinyl Estradiol
155\tCiprofloxacin
156\tMorphine
157\tInsulin Human; Insulin Isophane Human
158\tLevocetirizine
159\tNirmatrelvir; Ritonavir
160\tValproate
161\tAtomoxetine
162\tBudesonide
163\tTiotropium
164\tMelatonin
165\tCefdinir
166\tDoxepin
167\tOlanzapine
168\tPhentermine
169\tOfloxacin
170\tEthinyl Estradiol; Etonogestrel
171\tMupirocin
172\tBenazepril
173\tTimolol
174\tMagnesium Salts
175\tFluconazole
176\tRisperidone
177\tVerapamil
178\tLinaclotide
179\tCyclosporine
180\tDoxazosin
181\tAlbuterol; Ipratropium
182\tHydrocortisone
183\tDiazepam
184\tTelmisartan
185\tCarbamazepine
186\tAmlodipine; Benazepril
187\tLithium
188\tEvolocumab
189\tDesvenlafaxine
190\tDorzolamide
191\tNebivolol
192\tDicyclomine
193\tTorsemide
194\tAnastrozole
195\tEnalapril
196\tPolyethylene Glycol 3350
197\tTretinoin
198\tTadalafil
199\tSacubitril; Valsartan
200\tCalcium
201\tPramipexole
202\tMesalamine
203\tMetronidazole
204\tNortriptyline
205\tEmtricitabine; Tenofovir
206\tRimegepant
207\tNitroglycerin
208\tRizatriptan
209\tLiraglutide
210\tAcetaminophen; Codeine
211\tRamipril
212\tRopinirole
213\tBrimonidine
214\tMirabegron
215\tColchicine
216\tTicagrelor
217\tTerazosin
218\tAmiodarone
219\tFexofenadine
220\tLiothyronine
221\tBisoprolol
222\tOmega-3-acid Ethyl Esters
223\tFlecainide
224\tOxcarbazepine
225\tDesogestrel; Ethinyl Estradiol
226\tAscorbic Acid
227\tSodium Salts
228\tKetorolac
229\tDorzolamide; Timolol
230\tPromethazine
231\tLevofloxacin
232\tLabetalol
233\tNystatin
234\tCyproheptadine
235\tErythromycin
236\tDutasteride
237\tMoxifloxacin
238\tBimatoprost
239\tPrimidone
240\tSucralfate
241\tBetamethasone; Clotrimazole
242\tSenna; Docusate
243\tBumetanide
244\tIcosapent Ethyl
245\tSolifenacin
246\tDexamethasone
247\tEpinephrine
248\tPenicillin V
249\tCalcitriol
250\tOseltamivir
251\tPolymyxin B; Trimethoprim
252\tDextromethorphan; Promethazine
253\tTerbinafine
254\tLinagliptin
255\tMethimazole
256\tMetoclopramide
257\tMedroxyprogesterone
258\tPancrelipase Amylase; Pancrelipase Lipase; Pancrelipase Protease
259\tClotrimazole
260\tDexamethasone; Neomycin; Polymyxin B
261\tCalcium Phosphate; Cholecalciferol
262\tAcetaminophen; Butalbital; Caffeine
263\tGuanfacine
264\tSodium Fluoride
265\tCodeine; Guaifenesin
266\tLactulose
267\tFluorouracil
268\tIpratropium
269\tOlopatadine
270\tChlorhexidine
271\tNabumetone
272\tMometasone
273\tPolyethylene Glycol 3350 With Electrolytes
274\tHydroquinone
275\tPhenazopyridine
276\tLoperamide
277\tLidocaine
278\tCiclopirox
279\tCefuroxime
280\tBetamethasone
281\tBrompheniramine; Dextromethorphan; Pseudoephedrine
282\tEthinyl Estradiol; Norgestrel
283\tCiprofloxacin; Dexamethasone
284\tDiphenhydramine
285\tEthinyl Estradiol; Norelgestromin
286\tAtropine; Diphenoxylate
287\tIndomethacin
288\tNiacin
289\tLactate
290\tVitamin E
291\tGuaifenesin
292\tPseudoephedrine
293\tBisacodyl
294\tRiboflavin
295\tIvermectin
296\tEtodolac
297\tLactobacillus Acidophilus
298\tTobramycin
299\tKetotifen
300\tLoratadine; Pseudoephedrine"""


def main():
    fh = open(OUT, 'w')
    fh.write('rank\tgeneric\n')
    fh.flush()
    seen = set()
    n_rows = 0
    n_unique = 0
    for line in RAW.strip().split('\n'):
        rank, names = line.split('\t', 1)
        for name in names.split(';'):
            name = name.strip().lower()
            if not name:
                continue
            n_rows += 1
            if name in seen:
                continue
            seen.add(name)
            n_unique += 1
            fh.write(f'{rank}\t{name}\n')
            fh.flush()
            print(f'  rank {rank}: {name}', flush=True)
    fh.close()
    print(f'\nWrote {n_unique} unique generics from {n_rows} (combo-expanded) rows to {OUT}', flush=True)


if __name__ == '__main__':
    main()
