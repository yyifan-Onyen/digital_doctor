# OCD ERP Clinical Skill

This bundle defines the domain policy used by the Digital Doctor execution
harness. It is an executable clinical policy package, not merely a system
prompt.

## Inputs

- Latest patient turn and bounded dialogue memory
- Structured formulation and phase state
- Retrieved transcript and knowledge evidence
- Platform risk assessment result

## Outputs

- A structured state delta
- One action plan selected from `actions.json`
- Treatment-readiness authorization
- Prompt specifications for the configured model adapter
- A domain safety verdict for the drafted response

## Invariants

- Do not provide reassurance that functions as a compulsion.
- Do not treat ego-dystonic intrusive thoughts as intent without evidence of
  genuine desire, plan, or intent.
- Do not start behavioral treatment before the formulation, phase, and
  stability prerequisites are satisfied.
- Do not recommend physically dangerous exposure or medication changes.
- The harness owns persistent stop state, alert delivery, audit traces, and the
  final non-bypassable safety decision.

## Versioning

The harness pins `skill_id`, `version`, and a bundle checksum in every session
and turn trace. Changes to state, actions, phase policy, or safety behavior must
increment the manifest version.
