# Collaboration guide

## Source of truth

The project brief, experiment status, development results, and the SUB real-bird
rung are maintained in the **cmcert Research Wiki**. Keep this file as a short
handoff guide; add substantive research decisions and status updates to the
wiki rather than duplicating them here.

## SUB E4 handoff

SUB is the conditional real-bird read-path rung. It may begin only after E1
returns GO. Before that decision, collaborators may complete only the licence
check, per-cell data audit, and matched-baseline investigation.

Every SUB result must be labelled **conditional on edit fidelity**. Do not tune
thresholds on SUB and do not edit `configs/thresholds.yaml` from SUB outcomes.

## Update protocol

When adding SUB progress to the wiki, record:

1. Source and licence evidence.
2. Attribute-by-class counts and the E0-derived power threshold.
3. Matched-baseline availability.
4. Fidelity spot-check numerator, denominator, and criteria.
5. Conditional per-concept decisions alongside SUB detection accuracy.

Keep the repository's fail-loud TODO convention intact until a decision is
explicitly fixed and documented.

## Certificate scaffold

`src/cmcert/certificate.py` is the executable specification scaffold for
three-way read-path and write-path outcomes. It is not the final method: its
bootstrap confidence bounds retain `TODO(spec)` until Theorem 2 supplies valid
finite-sample bounds. Use its `CertificateSpec` to declare the intervention
unit, affected and unaffected detector sets, family-wise budget, and margins.
