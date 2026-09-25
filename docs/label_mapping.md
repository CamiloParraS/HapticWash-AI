# Label mapping and axis frames (SPEC §5, M1)

Source label → canonical §1.3 id, for every M1 dataset. Implemented in
`haptic_ai/ingest/<key>.py` (`LABEL_MAP`). Anything that can't be mapped with confidence
goes to `WASH_OTHER` (inside a wash) or `-1` (unlabelled, excluded), never to a made-up
step.

## `zhang_who` (PART 1, `FRAME/FrameACM_<n>.xls`) — D5

| `Action` | WHO meaning | canonical | why |
|---|---|---|---|
| 0 | wet hands | `-1` | uwash can't separate wetting from walking, so it excludes it too (D18) |
| 0.5 | apply soap | `-1` | same |
| 1 | palm to palm | `PALM_TO_PALM` (1) | direct |
| 2.1, 2.2 | right palm over left dorsum / left over right | `BACK_OF_HAND` (2) | canonical has no side |
| 3 | palm to palm, fingers interlaced | `INTERLACED_FINGERS` (3) | direct |
| 4 | backs of fingers to opposing palms, interlocked | `WASH_OTHER` (6) | no canonical step; not merged into 3 (D5) |
| 5.1, 5.2 | rotational rubbing of thumbs (L / R) | `THUMBS` (4) | canonical has no side |
| 6.1, 6.2 | fingertips rubbed in palm (L / R) | `FINGERTIPS` (5) | canonical has no side |
| 7 | rinse | `-1` | same as wetting |
| *(no step)* | before the first step, after the last, short gaps between steps | `-1` | not annotated. At a sink but not idle, so it isn't `NULL` |

Annotation bounds are sample indices at 100 Hz, inclusive. Where a step starts on the
sample the previous one ends on (4 files), the later step wins. After resampling to
50 Hz each output sample takes the label of the 100 Hz sample it sits on (every 2nd).
Each wash ends at its last annotated sample (D18).

So `WASH_OTHER` means the same thing in both datasets: backs of fingers only.

## `uwash` (per-sample `label`) — D14

| `label` | gesture | canonical | why |
|---|---|---|---|
| 1 | palm to palm | `PALM_TO_PALM` (1) | direct |
| 2, 3 | back of hand (R over L / L over R) | `BACK_OF_HAND` (2) | canonical has no side |
| 4 | palm to palm, fingers interlaced | `INTERLACED_FINGERS` (3) | direct |
| 5 | backs of fingers, interlocked | `WASH_OTHER` (6) | same as `zhang_who` Action 4 |
| 6, 7 | thumbs (L / R) | `THUMBS` (4) | canonical has no side |
| 8, 9 | fingertips (L / R) | `FINGERTIPS` (5) | canonical has no side |
| 0, within 5 s of a gesture | wet / soap / rinse, not separated | `-1` | would conflict with zhang's `WASH_OTHER` if trained as `NULL` |
| 0, elsewhere | walking, drying, idle between washes | `NULL` (0) | not washing |

The 5 s edge zone (`EDGE_ZONE_S`) is a tuning knob. Check it against the M1 plots.

## `ablutomania`

Not ingested yet: the files aren't downloaded, and the set has no step labels (D15), so
nothing in M2 uses it. The mapping goes here once the files are inspected.

## Axis frames (D10)

Target: the Android sensor frame (x to the right of the screen, y to its top, z out of
the screen). acc in m/s² including gravity, gyro in rad/s, right-handed. Measured
figures are means over the whole recording, so motion blurs them. They show the gravity
direction on average, not the full frame.

| dataset | device | wrist | units at source → canonical | mean acc (x, y, z) m/s² | mapping to Android |
|---|---|---|---|---|---|
| `zhang_who` | Byteflies | left | counts; acc / 4096 g, gyro / 131 °/s (D17) | (3.2, −5.5, −1.5) as recorded | (x, y, z) → (−x, y, −z), acc and gyro (D19) |
| `zhang_who` | Byteflies | right | same | (−3.9, −4.8, −4.8) as recorded | same as left |
| `uwash` | Samsung Gear Sport (Tizen) | left (paper photos, D19) | acc m/s²; gyro °/s → rad/s | (−3.6, −6.3, −2.0) | as-is; assumes Tizen = Android, **unverified** against a Wear OS watch |

Mean acc is the raw recording, before any mapping. Gravity alignment (§8.7) removes tilt
but not heading. A 180° flip about y turns into a 180° heading difference after alignment,
so the zhang mapping matters even for aligned data.

How D19 was found: per-step mean gravity directions (labels 1–6), fitted over all eight
axis-sign combinations. zhang left → uwash fits best with (−1, 1, −1) (mean error 30°, next
best 50°), and a free orthogonal fit gives the same flip plus ~24° mounting tilt. zhang
right → left fits best with x alone flipped, so the two zhang wrists are mirror images
across x, as `preprocess.mirror` assumes. The flip commutes with that mirror.

## Known data issues

- **`zhang_who` gyro is clipped** at ±32767 counts. With the corrected scale (D17) that is
  ±250 °/s ≈ 4.4 rad/s, the sensor's range, not an artefact. 1.7 % of per-axis samples sit
  at full scale, mostly X during thumbs and fingertips; uwash exceeds 250 °/s on 2.9 %.
  The clipped samples are left as they are. The model sees true motion above 4.4 rad/s
  flattened on this test set only.
- **`uwash` timing**: jittery (dt p1 / p99 = 1 / 34 ms around 20 ms). It has 2,125 gaps
  over 100 ms (median 138 ms), and each one splits the session. 95 % of rows (and of
  gesture rows) end up in sessions of 3 s or more.
