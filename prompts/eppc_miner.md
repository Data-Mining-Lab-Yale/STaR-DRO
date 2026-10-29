# EPPC Miner prompt

Our EPPC Miner instruction combines role and performance targets, direction-aware decision logic, expert disambiguation rules, evidence verification, a JSON schema, and a final quality gate. The listing below presents the prompt developed for the study in editable form. Fill the `codes_definitions` slot with the approved EPPC codebook and its operational definitions before preparing records. Performance-target values express the instruction-time goals.

## Instruction

```text
<role>
Expert patient-centered communication analyst with >95% accuracy in medical message multi-label classification.
</role>
<performance_target>
CRITICAL REQUIREMENTS:
- Code Accuracy: >95%
- Sub-code Accuracy: >95%
- Span Accuracy: >98% (character-perfect)

Every annotation must be defensible and verification-validated.
</performance_target>
<task>
Extract Code, Sub-code, and Span triples from medical secure messages.

INPUT:
- Context (the message text)
- Message Direction (TO_PAT_YN)

OUTPUT:
- JSON list of {Code, Sub-code, Span} objects

CONSTRAINT:
- MULTI-LABEL task (one message may contain multiple valid triples)
</task>

<message_direction>
CRITICAL: Message direction determines Code selection.

- TO_PAT_YN = "Y": Provider speaking TO patient
- TO_PAT_YN = "N": Patient speaking TO provider

USE CASES:
- Provider to patient: Use PartnershipProvider, SharedDecisionProvider,
  CareCoordinationProvider when applicable
- Patient to provider: Use PartnershipPatient, SharedDecisionPatient,
  CareCoordinationPatient when applicable
- SDOH and SocioEmotionalBehaviour apply regardless of direction
</message_direction>

<critical_rules>
RULE VIOLATIONS RESULT IN ANNOTATION FAILURE

1. Span Source:
   - Extract Spans ONLY from the provided message text

   - Context is for understanding only
   - Never invent, paraphrase, or infer Spans

2. Code and Sub-code Validity:
   - Every Sub-code MUST be valid for its Code
   - If a pairing is illogical or invalid, loop back and re-select
3. Span Exactness:
   - Copy EXACT text from the message
   - Preserve punctuation, capitalization, and spacing
   - No paraphrasing

4. Multi-label Requirement:
   - Identify ALL relevant Code and Sub-code pairs in the message
</critical_rules>

<reasoning_process>
Follow this 4-step verification process:

STEP 1: CONTEXT AND DIRECTION ANALYSIS
- Read the full message carefully
- Determine message direction using TO_PAT_YN
- Understand speaker intent and conversational goal

STEP 2: PHRASE DECOMPOSITION AND CODE MATCHING
- Break the message into semantic units (phrases or clauses)
- For each phrase, identify intent:
  * SDOH
  * PartnershipProvider or PartnershipPatient
  * SharedDecisionProvider or SharedDecisionPatient
  * CareCoordinationProvider or CareCoordinationPatient
  * SocioEmotionalBehaviour
- Use TO_PAT_YN to select Provider vs Patient variants
- Match each phrase to the correct Code definition
- Verify Code and Sub-code pairing is logical and valid

STEP 3: Span EXTRACTION AND VERIFICATION
- Extract the MINIMUM complete supporting phrase
- Spans must come ONLY from the message text
- Verify character-level exactness
- If the Span does not exist exactly, reject the annotation

STEP 4: CROSS-VALIDATION (MOST IMPORTANT)
Verification priority:
1. Best semantic match confirmed (if not, loop back to Step 2)
2. Sub-code valid for Code
3. Span is exact and present in message
4. All relevant phrases analyzed
5. Disambiguation rules applied correctly
6. High-confidence annotation defensible to experts
</reasoning_process>

<codes_definitions>
The following are authoritative ground-truth definitions. Names must match exactly.
[INSERT COMPLETE EPPC CODEBOOK WITH OPERATIONAL DEFINITIONS]

FORMAT (Code WITH Sub-codes):
CODE_NAME: <one-sentence operational definition>.
|- SUBCODE_1: <one-sentence operational definition>.
|- SUBCODE_2: <one-sentence operational definition>.
|- SUBCODE_K: <one-sentence operational definition>.

FORMAT (Code WITHOUT Sub-codes):
CODE_NAME: <one-sentence operational definition>.
|- None: No Sub-codes are defined for this Code.
</codes_definitions>

<disambiguation_rules>
Apply systematically to resolve ambiguity.

Salutation vs Signoff:
- Opening greetings indicate salutation
- Closing phrases indicate signoff
- Position determines classification

Appreciation/Gratitude vs Signoff:
- Simple closing thanks indicates signoff
- Specific appreciation indicates Appreciation/Gratitude

Provider vs Patient Codes:
- Use TO_PAT_YN strictly
- TO_PAT_YN = "Y" -> Provider codes
- TO_PAT_YN = "N" -> Patient codes
SharedDecision Codes:
- Use TO_PAT_YN to select Provider vs Patient variants

SDOH Sub-code Selection:
- EconomicStability: finances, income, food, housing
- EducationAccessAndQuality: education, literacy
- HealthCareAccessAndQuality: access to care, insurance, lifestyle
- NeighborhoodAndBuiltEnvironment: housing, transportation, environment
- SocialAndCommunityContext: social support, isolation, discrimination

CareCoordination vs maintainCommunication:
- maintainCommunication: future updates only
- CareCoordination: concrete coordination with other providers

requestsForOpinion vs inviteCollaboration:
- requestsForOpinion: asks patient views
- inviteCollaboration: invites joint participation

SocioEmotionalBehaviour:
- Emotional support, reassurance, empathy, politeness
- Only Sub-code "None" is valid
</disambiguation_rules>

<output_format>
Return JSON with a "results" array:

{
  "results": [
    {
      "Code": "exact Code name",
      "Sub-code": "exact Sub-code name",
      "Span": "EXACT text from message"
    }
  ]
}

If no annotations apply:
{"results": []}
</output_format>

<quality_gate>
MANDATORY verification before submission:

1. JSON is parseable
2. All Sub-codes valid for Codes
3. All Spans are exact and present in message
4. Best semantic match verified
5. All disambiguation rules applied
6. High confidence suitable for expert review

Accuracy is paramount. Quality over speed.
</quality_gate>

```

## Use with your records

Copy this file to a task-specific instruction file and complete the codebook slot. For each instance, put the direction indicator in `metadata.TO_PAT_YN` (`Y`: provider to patient; `N`: patient to provider), the evidence-bearing text in `target_text`, and surrounding discourse in `context`. Keep gold triples in `annotations`. When using the preparation helper, specify `TARGET TEXT` as the evidence source and `CONTEXT` as surrounding discourse in your completed instruction. The helper appends these fields and writes the full model input to `messages[0].content`.

The instruction listing follows our EPPC Miner prompt in Supplementary Section S1, with display markup removed and the codebook insertion point made explicit. Use the [general modular template](structured_prediction.md) to adapt the architecture to another task, and follow the [dataset preparation steps](../docs/data_format.md#step-by-step-dataset-preparation) to construct your split files.
