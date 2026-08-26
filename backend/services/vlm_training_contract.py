"""Prompt contract shipped with the 2026-08-17 gait LoRA bundle.

The text and ordering in this module are intentionally stable. The C0B adapter
was trained with SYSTEM -> ANCHOR -> GLOSSARY -> QUESTION -> VIDEO, so changing
the wording here changes the inference distribution.
"""

from __future__ import annotations

from typing import Any


SYSTEM_ANCHOR = (
    "You are a highly skilled and experienced movement disorder neurologist specialising in "
    "gait analysis in Parkinson's disease. Your expertise lies in evaluating gait abnormalities "
    "from video. You possess deep knowledge of biomechanics and motion analysis.\n"
    "Your aim: analyse the gait pattern in the video provided and give the MDS-UPDRS score for "
    "the scoring item defined below.\n"
    "End your response with a line of the exact form 'answer: <integer 0-4>'."
)

GAIT_ANCHOR = (
    "Scoring Item: MDS-UPDRS Part III, item 3.10 Gait\n"
    "Explanation: The patient walks away from and towards the examiner so that both sides of the "
    "body can be observed. A single score covers all of the following observations taken together: "
    "stride amplitude, stride speed, height of foot lift, heel strike during walking, turning, and "
    "arm swing. Freezing episodes, festinating (rapidly shortening, accelerating) steps and "
    "instability during turning are all gait impairments and count towards this score.\n"
    "Scoring Criteria:\n"
    "0 Normal: no problems.\n"
    "1 Slight: independent walking with minor gait impairment.\n"
    "2 Mild: independent walking but with substantial gait impairment.\n"
    "3 Moderate: requires an assistance device for safe walking (walking stick, walker) or "
    "external cueing, but not another person.\n"
    "4 Severe: cannot walk at all, or only with another person's assistance."
)

GAIT_GLOSSARY = (
    "Definitions of the clinical observation terms. Use exactly the wording given in brackets.\n"
    "- Gait speed [normal | mildly reduced | markedly reduced]\n"
    "  Overall walking velocity along the straight segments. Parkinsonian slowness does not "
    "resolve when the patient tries to walk faster. Judge this from the straight walking, not "
    "from the turn.\n"
    "- Stride length [normal | mildly reduced | markedly reduced]\n"
    "  Length of a single stride, seen in the distance between successive heel strikes and in "
    "how far the feet are lifted from the floor. Shortened strides give a shuffling appearance.\n"
    "- Left-right asymmetry [absent | mild | moderate to severe]\n"
    "  Difference in STEP LENGTH between the left and the right leg. This item is about the legs, "
    "not the arms. Limping from pain or from a leg length difference is NOT this item.\n"
    "- Arm swing asymmetry [absent | mild | moderate to severe]\n"
    "  Difference in arm swing amplitude between the two arms, up to one arm held almost still "
    "against the body. If a hand is in a pocket or the patient is carrying something, this item "
    "cannot be observed.\n"
    "- Festination [absent | mild | moderate to severe]\n"
    "  Steps that become progressively SHORTER and FASTER at the same time, with the trunk moving "
    "ahead of the feet, and which the patient cannot easily stop. It is visible only by comparing "
    "the start and the end of the walk. Walking quickly with preserved step length is NOT "
    "festination.\n"
    "- Freezing episodes [absent | present]\n"
    "  A sudden involuntary inability to move the feet forward although the patient intends to "
    "walk, lasting seconds. Trembling of the feet in place and very short failed steps also count. "
    "Most common at gait initiation, during turning and in narrow spaces. Standing still on "
    "purpose, waiting, or pausing to listen to the examiner is NOT freezing. Walking slowly is "
    "NOT freezing.\n"
    "- Postural stability [normal | mild | moderate to severe]\n"
    "  Severity of postural INSTABILITY: staggering, taking a rescue step to regain balance, or "
    "holding on to a wall or a device. Here 'mild' means mild instability, not mild stability.\n"
    "- Stooped posture [absent | mild | moderate to severe]\n"
    "  Forward flexion of the trunk and the neck that persists throughout walking.\n"
    "- Turning\n"
    "  Each time the patient reverses walking direction, given as <start-end> in seconds of video "
    "time together with how long the turn lasted. The frame timestamps in the video use the same "
    "clock, so these intervals point at specific frames. A healthy turn is brief and pivots on one "
    "foot. A parkinsonian turn is slow and is taken in several small steps with the body kept in "
    "one block. In this cohort the mean turn lasts about 0.6 s at grade 0, 1.2 s at grade 1 and "
    "2.8 s at grade 2 or above, so a LONGER turn indicates more severe impairment.\n"
    "- Freezing episodes, when present, are given in the same <start-end> form.\n"
    "These terms belong to four different MDS-UPDRS items (3.10 gait, 3.11 freezing, 3.12 "
    "postural stability, 3.13 posture). Do not add their severities together to obtain the 3.10 "
    "score."
)

GAIT_QUESTION = (
    "Score the gait task shown in the video (MDS-UPDRS 3.10). "
    "Consider stride amplitude, speed, heel strike, turning and arm swing."
)


def build_c0b_messages() -> list[dict[str, Any]]:
    """Return a fresh OpenAI-style message structure for the C0B adapter."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_ANCHOR}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": GAIT_ANCHOR},
                {"type": "text", "text": GAIT_GLOSSARY},
                {"type": "text", "text": GAIT_QUESTION},
            ],
        },
    ]
