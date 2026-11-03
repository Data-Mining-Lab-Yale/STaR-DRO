# EPPC Miner prompt

This is the instruction used in the study. Every training, validation and test record presents it verbatim to the model, followed by the input block for one sentence. It combines role and performance targets, direction-aware decision logic, the EPPC codebook with operational definitions, expert disambiguation rules, a four-step verification process, a JSON output schema, and a final quality gate. Performance-target values express instruction-time goals, not measured results.

## Instruction

```text
<role>Expert patient-centered communication analyst with >95% accuracy in medical message multi-label classification.</role>

<performance_target>
🎯 CRITICAL REQUIREMENTS:
• Code Accuracy: >95%
• Sub-code Accuracy: >95%
• Span Accuracy: >98% (character-perfect)
Every annotation must be defensible and verification-validated.
</performance_target>

<task>
Extract Code, Sub-code & Span triples from medical secure messages.
- INPUT: Context (Previous/Current/Next sentences) + Message Direction
- OUTPUT: JSON list of {Code, Sub-code, Span} objects
- CONSTRAINT: MULTI-LABEL task (one sentence → MULTIPLE triples possible)
</task>

<message_direction>
🔄 CRITICAL: Message direction determines Code selection
- TO_PAT_YN = "Y" → Provider speaking TO patient
- TO_PAT_YN = "N" → Patient speaking TO provider

USE CASE:
• Provider speaking TO patient: Use PartnershipProvider, SharedDecisionProvider for partnership/decision codes
• Patient speaking TO provider: Use PartnershipPatient, SharedDecisionPatient for partnership/decision codes
• All InfoGive/ InfoSeek/ InfoGiveSDOH/ InfoSeekSDOH/ Socioemotional/Empathy codes apply regardless of direction
</message_direction>

<critical_rules>
⚠️ RULE VIOLATIONS = ANNOTATION FAILURE ⚠️

1. Span Source: Extract spans ONLY from Current Sentence (Previous/Next = context ONLY, never a span source)

2. Code & Sub-code Validity: Every Sub-code MUST be valid for its Code. If pairing illogical → LOOP BACK, re-select.

3. Span Exactness: Copy EXACT text (preserve punctuation, capitalization, spacing). No paraphrasing.

4. Multi-Label: Identify ALL relevant Code & Sub-code pairs in Current Sentence.
</critical_rules>

<reasoning_process>
Follow this 4-step verification process:

STEP 1: CONTEXT & DIRECTION ANALYSIS
• Read all three sentences, identify Current Sentence position
• Determine message direction using TO_PAT_YN
• Understand conversational flow and speaker intent

STEP 2: PHRASE DECOMPOSITION & CODE MATCHING
• Break Current Sentence into semantic units
• For each phrase, identify intent:
  - InfoGive/InfoSeek (STATEMENT vs QUESTION for clinical info)
  - InfoGiveSDOH/InfoSeekSDOH (STATEMENT vs QUESTION for social factors)
  - PartnershipProvider/Patient (relationship building - use direction from TO_PAT_YN)
  - SharedDecisionProvider/Patient (decision-making - use direction from TO_PAT_YN)
  - Socioemotional/Empathy (emotional connection)
• Match phrase to Code definition using disambiguation rules below
• Verify Code & Sub-code pairing is valid and logical

STEP 3: SPAN EXTRACTION & VERIFICATION
• For each Code & Sub-code pair, extract MINIMUM complete phrase from Current Sentence ONLY
• Character-level verification: exact match, no modifications
• If span not in Current Sentence → REJECT annotation

STEP 4: CROSS-VALIDATION ⚠️ MOST IMPORTANT STEP ⚠️
Priority verification order: 
✓ Best semantic match: Is this the BEST Code & Sub-code for this span? (If NO → LOOP BACK to Step 2)
✓ Code & Sub-code validity: Sub-code in valid list for this Code?
✓ Span accuracy: Exact text from Current Sentence?
✓ Completeness: All phrases analyzed?
✓ Disambiguation rules applied correctly?
✓ High confidence for expert defense?
</reasoning_process>

<codes_definitions>
Based on authoritative codebook - these are ground-truth definitions:

InfoGiveSDOH: Providing/stating information about social, economic, environmental factors affecting health.
├─ EconomicStability: Employment, income, financial insecurity, housing instability, socioeconomic status, socioeconomic factors, work-related challenges
├─ HealthCareAccessAndQuality: Healthcare access, insurance, lifestyle, substance use/abuse, diet, activity, sleep
└─ NeighborhoodAndBuiltEnvironment: Housing quality, transportation, neighborhood safety, environment

InfoGive: Providing clear, actionable guidance or knowledge (STATEMENTS not questions) for patient/provider understanding.
├─ Symptoms: Stating subjective patient experiences (pain, fatigue, dizziness)
├─ Diagnostics: Providing test results, lab values, imaging findings
├─ Drugs: Providing medication info (dosage, side effects, interactions, purpose)
├─ Prognosis: Providing disease course, treatment outcomes, recovery timelines
├─ Generalinformation: Broad health information not in other subcategories
├─ Instruction: Specific directions for actions (medication, lifestyle, procedures)
├─ carecoordination: Info about provider communication, scheduling/logistics coordination
└─ SchedulingAppt: Refers to the Dates, times, and other appointment information related to seeing a provider, in person or virtually.

InfoSeek: Requesting information (QUESTIONS/requests) to understand patient condition, concerns, or care needs.
├─ Symptoms: Requesting information about subjective patient experiences (pain, fatigue, dizziness, discomfort)
├─ Diagnostics: Requesting test results, lab values, imaging findings, or diagnostic interpretations
├─ Drugs: Requesting medication information (dosage, side effects, interactions, purpose, compliance)
├─ Generalinformation: Requesting broad health information not fitting other subcategories
├─ carecoordination: Requesting information about provider communication, scheduling/logistics coordination
└─ SchedulingAppt: Refers to the Dates, times, and other appointment information related to seeing a provider, in person or virtually.

InfoSeekSDOH: Requesting information (QUESTIONS) about social determinants affecting health.
├─ EconomicStability: Requesting information about employment, income, financial insecurity, housing instability, socioeconomic Status, socioeconomic factors, work-related challenges
└─ HealthCareAccessAndQuality: Requesting information about healthcare access, insurance, lifestyle, substance use/abuse, diet, activity, sleep

PartnershipProvider: Provider actions (TO patient, TO_PAT_YN="Y") fostering collaborative, respectful relationship.
├─ salutation: Opening greeting at message start (Previous=None)
├─ signoff: Closing remark at message end (Next=None)
├─ connection: Building personal connection beyond clinical
├─ Appreciation/Gratitude: Expressing thanks/appreciation with context
├─ maintainCommunication: Keeping patient informed, ensuring dialogue
├─ build trust: Establishing/maintaining trust and reliability
├─ Alignment: Showing agreement/harmony with patient
├─ inviteCollaboration: Explicitly inviting patient into decision-making
└─ checkingUnderstanding/clarification: Checking understanding while providing clarification

PartnershipPatient: Patient actions (FROM patient, TO_PAT_YN="N") fostering collaborative relationship.
├─ salutation: Greeting the provider at the start of a message by using their name or professional title
├─ signoff: Indicating that the message is complete; ending the message in a polite manner
├─ Appreciation/Gratitude: Expressing thanks/appreciation for provider's care and efforts
├─ connection: Building personal connection beyond clinical matters
├─ Alignment: Showing agreement/harmony with provider
├─ build trust: Expressing confidence in provider's competence and intentions
├─ statePreferences: Articulating preferences/desires regarding care
├─ expressOpinions: Sharing thoughts/viewpoints on treatment
└─ activeParticipation/involvement: Actively engaging in care decisions

SharedDecisionProvider: Provider actions (TO patient, TO_PAT_YN="Y") facilitating collaborative decision-making.
├─ ShareOptions: Presenting available choices
├─ Approval/Reinforcement: Approving action while encouraging continuation
├─ EncourageQuestions: Prompts to ask additional questions or seek more information.
├─ SummarizeConfirmUnderstanding: The provider or decision-maker recaps the key points of the discussion in concise, clear language. This includes summarizing the options, risks, benefits, and any other important details that were shared. It may also involve highlighting the patient's values, concerns that were expressed. After summarizing, the provider checks with the patient or participant to ensure they understood the information accurately.
└─ MakeDecision: Making or guiding toward specific decision

SharedDecisionPatient: Patient actions (FROM patient, TO_PAT_YN="N") participating in shared decision-making.
├─ ApprovalofDecision/Reinforcement: Affirmation of a choice or decision after understanding available options.
├─ ExploreOptions: Indicators that a patient is interested in learning more about all options.
└─ SeekingApproval - Patient asks and/or seeks permission from the provider.

Socioemotional/Empathy: Recognizing and responding to emotional cues (concern, sadness, appreciation).
├─ ExpressConcern/unease: Expressing anxiety, worry, nervousness
├─ Sadness/fear: Expression of sadness, fear, dread, anxiety
├─ Reassurance: Alleviating concerns, reducing anxiety, providing comfort
├─ AcknowledgeSituation: Recognizing and addressing mistake/situation
└─ PositiveRemarks: Encouraging or complimentary statements
</codes_definitions>

<disambiguation_rules>
Apply systematically to resolve ambiguous cases:

salutation (Opening) vs signoff (Closing):
• "Hi", "Hello", "Dear [Name]" or soemthing similar at opening → salutation
• "Thank you", "Best regards", "Sincerely" or soemthing similar at closing → signoff
• Position determines classification with 90%+ confidence

Appreciation/Gratitude vs signoff:
• Simple closing: "Thanks", "Thank you" or soemthing similar at end → signoff
• With context: "Thank you for [specific]", "I appreciate [specific]" or soemthing similar not at end → Appreciation/Gratitude

InfoGive vs InfoSeek (applies to ALL Sub-codes: Symptoms, Diagnostics, Drugs, SchedulingAppt, Prognosis, Generalinformation, carecoordination):
• STATEMENT (providing/describing info) → InfoGive + Sub-code
• QUESTION (requesting/asking for info) → InfoSeek + Sub-code

InfoGiveSDOH/InfoSeekSDOH (applies to: EconomicStability, HealthCareAccessAndQuality, NeighborhoodAndBuiltEnvironment):
• STATEMENT about social factors → InfoGiveSDOH + Sub-code
• QUESTION about social factors → InfoSeekSDOH + Sub-code

HealthCareAccessAndQuality (InfoGiveSDOH/InfoSeekSDOH sub-code):
- Topics: Healthcare access, insurance, lifestyle, substance use/abuse, diet, activity, sleep
- STATEMENT about these topics → InfoGiveSDOH + HealthCareAccessAndQuality
- QUESTION about these topics → InfoSeekSDOH + HealthCareAccessAndQuality
- Examples: "Do you have health insurance?" (InfoSeekSDOH), "I exercise daily" (InfoGiveSDOH)

EconomicStability (InfoGiveSDOH/InfoSeekSDOH sub-code):
- Topics: Employment, income, financial insecurity, housing instability, socioeconomic status, socioeconomic factors, work-related challenges
- STATEMENT about these topics → InfoGiveSDOH + EconomicStability
- QUESTION about these topics → InfoSeekSDOH + EconomicStability
- Examples: "Are you currently employed?" (InfoSeekSDOH), "I lost my job" (InfoGiveSDOH)

PartnershipProvider vs PartnershipPatient:
• Use TO_PAT_YN to determine speaker
• If TO_PAT_YN="Y" (Provider speaking TO patient) → PartnershipProvider + Sub-code
• If TO_PAT_YN="N" (Patient speaking TO provider) → PartnershipPatient + Sub-code
• Applies to: salutation, signoff, Appreciation/Gratitude, connection, Alignment, build trust, inviteCollaboration

Symptoms vs Diagnostics (InfoGive/InfoSeek sub-codes):
• Diagnostics: Test results, lab values, objective measurements (e.g., "my PSA is elevated", "blood test shows", "imaging results")
• Symptoms: Subjective experiences, patient-reported sensations (e.g., "fever", "headache", "my urine is normal clear yellow")

maintainCommunication vs carecoordination (PartnershipProvider sub-codes):
• maintainCommunication: Keeping contact without providing information (e.g., "I will let you know as soon as he answers me", "Will update you soon")
• - carecoordination: Providing actual coordination information that involves another organization, provider, or person for coordinating patient care (e.g., "I contacted Dr. Smith about your referral")

connection vs Appreciation/Gratitude vs build trust:
• connection: Personal interest beyond clinical (e.g., "Take care", "Looking forward to seeing you")
• Appreciation/Gratitude: Explicit thanks with appreciate/thank/grateful (e.g., "Thank you for your patience")
• build trust: Specifically establishing trust/reliability (e.g., "You can count on us", "You're in good hands")

Alignment:
• Agreement or harmony in partnership context
• Provider aligning → PartnershipProvider + Alignment
• Patient aligning → PartnershipPatient + Alignment
</disambiguation_rules>

<output_format>
Return JSON with "results" array:
{
  "results": [
    {
      "Code": "exact Code name",
      "Sub-code": "exact Sub-code name",
      "Span": "EXACT text from Current Sentence"
    }
  ]
}

If NO annotations: {"results": []}
</output_format>

<quality_gate>
MANDATORY verification BEFORE submission (priority order):
1. ✓ JSON parseable
2. ✓ All Sub-codes valid for their Codes
3. ✓ All Spans exist character-perfectly in Current Sentence ONLY
4. ✓ Best semantic match verified (loop-back completed if needed)
5. ✓ All disambiguations applied correctly
6. ✓ High confidence (would defend to expert panel)

🎯 ACCURACY IS PARAMOUNT - Quality over speed - Expert-level confidence required
</quality_gate>
```

## Model input in the study

Each record places the instruction above, a blank line, and the following input block in a single user turn. The model is trained to produce the JSON `results` object after `OUTPUT:`.

```
INPUT: TO_PAT_YN: N  (Speaker: Patient speaking TO provider)
Previous sentence: <previous sentence of the same message, or None>
Current sentence: <target sentence; the only source of Spans>
Next sentence: <next sentence of the same message, or None>

OUTPUT:
```

`TO_PAT_YN` is `Y  (Speaker: Provider speaking TO patient)` for provider-authored messages and `N  (Speaker: Patient speaking TO provider)` for patient-authored messages. `Previous sentence` is `None` for the first sentence of a message and `Next sentence` is `None` for the last, which the salutation and signoff definitions use.

## Use with your records

To reproduce the study's input exactly, write each record's user turn as the instruction followed by the input block above, and store records in the `id`/`entries` or `query`/`answer` layout that the loader accepts ([data formats](../docs/data_format.md)).

Alternatively, copy this file to a task-specific instruction file and use the [preparation helper](../docs/data_format.md#step-by-step-dataset-preparation). Put the direction indicator in `metadata.TO_PAT_YN`, the target sentence in `target_text`, and the neighboring sentences in `context`; the helper appends these as `METADATA`, `CONTEXT` and `TARGET TEXT` sections rather than the `Previous`/`Current`/`Next sentence` lines, so state in your instruction that `TARGET TEXT` is the evidence source. Keep gold triples in `annotations`.

Use the [general modular template](structured_prediction.md) to adapt the architecture to another task.
