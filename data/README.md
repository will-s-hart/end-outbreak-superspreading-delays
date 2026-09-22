# Data

## `equateur_2018_onsets.csv`

Daily case counts **by date of symptom onset** for the 2018 Équateur Province (DRC) Ebola
virus disease outbreak.

| Column | Meaning |
| --- | --- |
| `date` | ISO-8601 calendar date |
| `onsets` | number of cases whose symptoms began on that date |

### Provenance

The observed period (5 April – 2 June 2018, 59 rows, 54 cases) is the incidence series
supplied with the project starter materials. It reproduces the outbreak described in
Supplementary Analysis 2 of

> Thompson RN, Hart WS, Keita M, Fall IS, Gueye AS, Chamla D, Mossoko M, Ahuka-Mundeke S,
> Nsio-Mbeta J, Jombart T, Polonsky J (2024). *Using real-time modelling to inform the 2017
> Ebola outbreak response in DR Congo*. Nature Communications **15**:5667.

which reports 54 cases with onsets between 5 April and 2 June 2018.

### Padding

The file is **padded with 52 trailing zero-count days** (3 June – 24 July 2018), so that it
covers the period from the first onset to the actual withdrawal of the Ebola Response Team
(ERT), the day the outbreak was declared over.

The analysis window runs **twenty days further, to 13 August 2018**, and
`outbreak_data.load_onset_data` pads those days with zeros too. No further case occurred in
Équateur. The extension exists so that every risk curve the report draws settles below both
decision thresholds inside the window: ending at the withdrawal left DLO, the Poisson limit and
most of the under-reporting sweep still above 0.01 on the last day. The file itself stops at the
withdrawal because padding can lengthen a series but never shorten it, and the validation
studies still load the 0–110 window they were run on (`end_date=ERT_WITHDRAWAL_DATE`).

The trailing zeros are not filler: they are the observations that drive the estimated risk
of additional transmission down over the ERT period. Truncating the series at the last
non-zero day would discard exactly the evidence the analysis is built on.

### Key dates and day indices

Day 0 is the date of the first observed onset, 5 April 2018.

| Event | Date | Day index |
| --- | --- | --- |
| First onset | 2018-04-05 | 0 |
| ERT arrival | 2018-05-08 | 33 |
| Last observed onset | 2018-06-02 | 58 |
| ERT withdrawal (last row of the file) | 2018-07-24 | 110 |
| End of the analysis window (padded by the loader) | 2018-08-13 | 130 |

Totals: 111 rows in the file and 131 days in the window, summing to 54 cases; 28 cases in the
pre-ERT period (days 0–32) and 26 after the ERT's arrival.

### Under-reporting

No under-reporting adjustment is applied. Supplementary Analysis 3 of the reference above
sketches one; it is out of scope here.
